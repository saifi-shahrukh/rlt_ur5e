"""Dataset recording script for robot teleoperation.

Collects demonstration data by recording robot observations and actions during
teleoperation sessions. Supports optional policy-assisted recording where a
pretrained model suggests actions that can be accepted or overridden.

Key Controls:
    SPACE       → Start recording current episode (robot moves but data is NOT saved until you press this)
    → (Right)   → End episode (save and move to next)
    ← (Left)    → Discard current episode and re-record
    ESC         → Stop recording entirely
"""

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

from lerobot.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
)
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import (
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
try:
    from lerobot.utils.import_utils import register_third_party_devices
except ImportError:
    from lerobot.utils.import_utils import register_third_party_plugins as register_third_party_devices
try:
    from lerobot.utils.robot_utils import busy_wait
except ImportError:
    from lerobot.utils.robot_utils import precise_sleep as busy_wait
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# Ensure third-party devices are discoverable by lerobot
from lerobot_camera_zmq import ZMQCameraConfig  # noqa: F401
from lerobot_robot_ur5e import UR5EConfig, UR5EDualCamConfig  # noqa: F401
from lerobot_camera_kinect import KinectCameraConfig  # noqa: F401
from lerobot_teleoperator_gello import GelloConfig  # noqa: F401
from lerobot_teleoperator_keyboard_ur5e import KeyboardUR5eConfig  # noqa: F401  # ← ADD


# ============================================================
# Custom keyboard listener with SPACE-to-start support
# ============================================================

def init_keyboard_listener_with_start():
    """Initialize keyboard listener with SPACE key to start recording.

    Key Controls:
        SPACE       → Start recording (begin saving frames for this episode)
        → (Right)   → End episode (save and move to next)
        ← (Left)    → Discard current episode and re-record
        ESC         → Stop recording entirely

    Returns:
        (listener, events) where events dict includes 'start_recording' flag.
    """
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False
    events["start_recording"] = False  # NEW: wait for SPACE to start

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        # In headless mode, auto-start recording
        events["start_recording"] = True
        return None, events

    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.space:
                if not events["start_recording"]:
                    print("\n  ● SPACE pressed → Recording STARTED! (press → to save, ← to discard)")
                    events["start_recording"] = True
            elif key == keyboard.Key.right:
                if events["start_recording"]:
                    # Only save if we are actually recording
                    print("\n  → Right arrow pressed → Ending episode (saving)...")
                    events["exit_early"] = True
                else:
                    print("\n  ⚠️  Right arrow ignored — press SPACE first to start recording!")
            elif key == keyboard.Key.left:
                if events["start_recording"]:
                    print("\n  ← Left arrow pressed → Discarding episode (re-record)...")
                    events["rerecord_episode"] = True
                    events["exit_early"] = True
                else:
                    print("\n  ⚠️  Left arrow ignored — press SPACE first to start recording!")
            elif key == keyboard.Key.esc:
                print("\n  ESC pressed → Stopping all recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


@dataclass
class DatasetRecordConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # A short but accurate description of the task performed during the recording (e.g. "Pick the Lego block and drop it in the box on the right.")
    single_task: str
    # Base directory for datasets. The final path will be <root>/<repo_id>.
    # Default: project workspace datasets folder (easy to find).
    # Set to None to use default HuggingFace cache (~/.cache/huggingface/lerobot/<repo_id>).
    root: str | Path | None = "/home/robolab-2/ur5e_hande_workspace/openpi_ur5e/datasets"
    # Limit the frames per second.
    fps: int = 60
    # Number of seconds for data recording for each episode.
    episode_time_s: int | float = 60
    # Number of seconds for resetting the environment after each episode.
    reset_time_s: int | float = 60
    # Number of episodes to record.
    num_episodes: int = 50
    # Encode frames in the dataset into video
    video: bool = True
    # Upload dataset to Hugging Face hub.
    push_to_hub: bool = True
    # Upload on private repository on the Hugging Face hub.
    private: bool = False
    # Add tags to your dataset on the hub.
    tags: list[str] | None = None
    # Number of subprocesses handling the saving of frames as PNG. Set to 0 to use threads only;
    # set to ≥1 to use subprocesses, each using threads to write images. The best number of processes
    # and threads depends on your system. We recommend 4 threads per camera with 0 processes.
    # If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses.
    num_image_writer_processes: int = 0
    # Number of threads writing the frames as png images on disk, per camera.
    # Too many threads might cause unstable teleoperation fps due to main thread being blocked.
    # Not enough threads might cause low camera fps.
    num_image_writer_threads_per_camera: int = 5
    # Number of episodes to record before batch encoding videos
    # Set to 1 for immediate encoding (default behavior), or higher for batched encoding
    video_encoding_batch_size: int = 20
    # Rename map for the observation to override the image and state keys
    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    # Whether to control the robot with a teleoperator
    teleop: TeleoperatorConfig | None = None
    # Whether to control the robot with a policy
    policy: PreTrainedConfig | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Resume recording on an existing dataset.
    resume: bool = False

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.teleop is None and self.policy is None:
            raise ValueError("Choose a policy, a teleoperator or both to control the robot")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


""" --------------- record_loop() data flow --------------------------
       [ Robot ]
           V
     [ robot.get_observation() ] ---> raw_obs
           V
     [ robot_observation_processor ] ---> processed_obs
           V
     .-----( ACTION LOGIC )------------------.
     V                                       V
     [ From Teleoperator ]                   [ From Policy ]
     |                                       |
     |  [teleop.get_action] -> raw_action    |   [predict_action]
     |          |                            |          |
     |          V                            |          V
     | [teleop_action_processor]             |          |
     |          |                            |          |
     '---> processed_teleop_action           '---> processed_policy_action
     |                                       |
     '-------------------------.-------------'
                               V
                  [ robot_action_processor ] --> robot_action_to_send
                               V
                    [ robot.send_action() ] -- (Robot Executes)
                               V
                    ( Save to Dataset )
                               V
                  ( Rerun Log / Loop Wait )
"""


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs after teleop
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs before robot
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],  # runs after robot
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    wait_for_start: bool = False,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (so100_leader.SO100Leader | so101_leader.SO101Leader | koch_leader.KochLeader),
                )
            ),
            None,
        )

        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. Currently only supported for LeKiwi robot."
            )

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    # ─── WAIT FOR SPACE KEY TO START RECORDING ───
    if wait_for_start and dataset is not None:
        print("\n  ○ WAITING... Press SPACE to start recording this episode")
        print("    (You can move the robot to the start position now)")
        while not events["start_recording"]:
            if events["exit_early"] or events["stop_recording"]:
                return
            # Keep robot responsive (read + send teleop) but don't record
            start_idle_t = time.perf_counter()
            obs = robot.get_observation()
            if policy is None and isinstance(teleop, Teleoperator):
                act = teleop.get_action()
                act_processed = teleop_action_processor((act, obs))
                robot_action_to_send = robot_action_processor((act_processed, obs))
                robot.send_action(robot_action_to_send)
            dt_s = time.perf_counter() - start_idle_t
            if dt_s < 1 / fps:
                busy_wait(1 / fps - dt_s)
        print("  ● Recording started!")
    # ─── END WAIT ───

    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # During reset phase (no dataset), allow SPACE to skip the wait
        # This lets the user press SPACE as soon as they're ready for next episode
        if dataset is None and events.get("start_recording", False):
            break

        # Get robot observation
        obs = robot.get_observation()

        # Applies a pipeline to the raw robot observation, default is IdentityProcessor
        obs_processed = robot_observation_processor(obs)

        if policy is not None or dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        # Get action from either policy or teleop
        if policy is not None and preprocessor is not None and postprocessor is not None:
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
            )

            act_processed_policy: RobotAction = make_robot_action(action_values, dataset.features)

        elif policy is None and isinstance(teleop, Teleoperator):
            act = teleop.get_action()

            # Applies a pipeline to the raw teleop action, default is IdentityProcessor
            act_processed_teleop = teleop_action_processor((act, obs))

        elif policy is None and isinstance(teleop, list):
            arm_action = teleop_arm.get_action()
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)
            act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
            act_processed_teleop = teleop_action_processor((act, obs))
        else:
            logging.info(
                "No policy or teleoperator provided, skipping action generation."
                "This is likely to happen when resetting the environment without a teleop device."
                "The robot won't be at its rest position at the start of the next episode."
            )
            continue

        # Applies a pipeline to the action, default is IdentityProcessor
        if policy is not None and act_processed_policy is not None:
            action_values = act_processed_policy
            robot_action_to_send = robot_action_processor((act_processed_policy, obs))
        else:
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

        # Send action to robot
        _sent_action = robot.send_action(robot_action_to_send)

        # Write to dataset
        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            log_rerun_data(observation=obs_processed, action=action_values)

        dt_s = time.perf_counter() - start_loop_t
        if dt_s < 1 / fps:
            busy_wait(1 / fps - dt_s)
        elif dt_s > 2.5 / fps:
            logging.warning(f"Loop took {dt_s} seconds, which is longer than the expected {1 / fps} seconds.")

        timestamp = time.perf_counter() - start_episode_t


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="recording")

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(
                action=robot.action_features
            ),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    # Resolve dataset root: combine base root + repo_id for organized storage
    dataset_root = None
    if cfg.dataset.root is not None:
        dataset_root = Path(cfg.dataset.root) / cfg.dataset.repo_id
        dataset_root.parent.mkdir(parents=True, exist_ok=True)

    # If dataset already exists and not resuming, remove it to start fresh
    if not cfg.resume and dataset_root is not None and dataset_root.exists():
        import shutil
        logging.warning(
            f"Dataset directory already exists: {dataset_root}\n"
            f"  Removing it to start fresh. Use --resume to continue an existing dataset."
        )
        shutil.rmtree(dataset_root)

    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=dataset_root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
    else:
        # Create empty dataset or load existing saved episodes
        sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=dataset_root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

    # Load pretrained policy
    policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)
    preprocessor = None
    postprocessor = None
    if cfg.policy is not None:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
                "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
            },
        )

    robot.connect()
    if teleop is not None:
        teleop.connect()

    # Use our custom keyboard listener with SPACE-to-start support
    listener, events = init_keyboard_listener_with_start()

    # Print control instructions
    print("\n" + "=" * 60)
    print("  UR5e Demo Collection - Key Controls")
    print("=" * 60)
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │  WORKFLOW:                                              │")
    print("  │  1. Move robot to start position (no data saved)       │")
    print("  │  2. Press SPACE → recording STARTS                     │")
    print("  │  3. Perform the task (data is being saved)             │")
    print("  │  4. Press → (Right) → episode SAVED                    │")
    print("  │  5. Reposition robot → press SPACE for next episode    │")
    print("  │                                                         │")
    print("  │  KEYS:                                                  │")
    print("  │  SPACE        → Start recording / skip to next episode │")
    print("  │  → (Right)    → End & SAVE current episode             │")
    print("  │  ← (Left)     → DISCARD current episode (re-record)    │")
    print("  │  G            → Toggle gripper open/close              │")
    print("  │  ESC          → Stop all recording & exit              │")
    print("  │                                                         │")
    print("  │  MOVEMENT (during recording):                           │")
    print("  │  Arrow Keys   → Move XY  |  PgUp/PgDn → Move Z        │")
    print("  │  Home/End     → Rotate wrist                           │")
    print("  └─────────────────────────────────────────────────────────┘")
    print(f"  Dataset: {cfg.dataset.repo_id}")
    print(f"  Save path: {dataset_root}")
    print(f"  Task: {cfg.dataset.single_task}")
    print(f"  FPS: {cfg.dataset.fps}")
    print(f"  Episodes to record: {cfg.dataset.num_episodes}")
    print("=" * 60 + "\n")

    with VideoEncodingManager(dataset):
        recorded_episodes = 0
        while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
            # Reset the start_recording flag for each new episode
            events["start_recording"] = False

            log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
            print(f"\n  ╔══════════════════════════════════════╗")
            print(f"  ║  Episode {recorded_episodes + 1}/{cfg.dataset.num_episodes}  ")
            print(f"  ║  Move robot to start position...    ║")
            print(f"  ║  Then press SPACE to begin          ║")
            print(f"  ╚══════════════════════════════════════╝")

            record_loop(
                robot=robot,
                events=events,
                fps=cfg.dataset.fps,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset=dataset,
                control_time_s=cfg.dataset.episode_time_s,
                single_task=cfg.dataset.single_task,
                display_data=cfg.display_data,
                wait_for_start=True,  # ← NEW: wait for SPACE
            )

            # Execute a few seconds without recording to give time to manually reset the environment
            # Skip reset for the last episode to be recorded
            if not events["stop_recording"] and (
                (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
            ):
                log_say("Reset the environment", cfg.play_sounds)
                print("\n  ⟳ Reset phase - reposition robot for next episode...")
                print("    Press SPACE when ready to start next episode (skips wait)")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    control_time_s=cfg.dataset.reset_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    wait_for_start=False,  # No wait during reset
                )

            if events["rerecord_episode"]:
                log_say("Re-record episode", cfg.play_sounds)
                print("\n  ✗ Episode discarded. Re-recording...")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            if not events["stop_recording"]:
                # Safety check: only save if frames were actually recorded
                if dataset.episode_buffer["size"] > 0:
                    dataset.save_episode()
                    recorded_episodes += 1
                    print(f"\n  ✓ Episode saved! ({recorded_episodes}/{cfg.dataset.num_episodes} complete)")
                else:
                    print("\n  ⚠️  Empty episode (0 frames) — skipping save.")
                    dataset.clear_episode_buffer()

    log_say("Stop recording", cfg.play_sounds, blocking=True)

    robot.disconnect()
    if teleop is not None:
        teleop.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()

    if cfg.dataset.push_to_hub:
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    print("\n" + "=" * 60)
    print(f"  ✓ Recording complete! Total episodes: {recorded_episodes}")
    print(f"  Dataset: {dataset_root}")
    print("=" * 60 + "\n")

    log_say("Exiting", cfg.play_sounds)
    return dataset


def main():
    register_third_party_devices()
    record()


if __name__ == "__main__":
    main()
