"""Remote policy inference for single-camera UR5e setup.

Maps the single RealSense D435 wrist camera to all image inputs
expected by the pi0_ur_tasks_merged_lora model.
"""

import logging
from collections import deque
from dataclasses import asdict, dataclass, field
from pprint import pformat
import numpy as np
import cv2
import time

from lerobot.robots import RobotConfig, make_robot_from_config, Robot
from lerobot.configs import parser
try:
    from lerobot.utils.import_utils import register_third_party_devices
except ImportError:
    from lerobot.utils.import_utils import register_third_party_plugins as register_third_party_devices
from lerobot.utils.utils import init_logging
from lerobot.utils.control_utils import is_headless
try:
    from lerobot.utils.robot_utils import busy_wait
except ImportError:
    from lerobot.utils.robot_utils import precise_sleep as busy_wait
from openpi_client.websocket_client_policy import WebsocketClientPolicy

# Register all plugins
from lerobot_camera_zmq import ZMQCameraConfig  # noqa: F401
from lerobot_robot_ur5e import UR5EConfig  # noqa: F401
from lerobot_teleoperator_gello import GelloConfig  # noqa: F401
from lerobot_teleoperator_keyboard_ur5e import KeyboardUR5eConfig  # noqa: F401

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    ip: str
    port: int
    prompt: str
    robot: RobotConfig = field(default_factory=lambda: UR5EConfig(ip="172.22.1.139"))
    fps: int = 30


def resize_image(img: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    """Resize image to target size expected by the policy model."""
    if img.shape[:2] == target_size:
        return img
    return cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)


def init_keyboard_listener():
    events = {"stop_inference": False}
    if is_headless():
        return None, events
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.esc:
                logger.info("Escape pressed. Stopping...")
                events["stop_inference"] = True
        except Exception:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener, events


def inference_loop(client: WebsocketClientPolicy, robot: Robot, events: dict, fps: int, prompt: str):
    action_queue = deque([])
    step = 0

    while True:
        start_time = time.perf_counter()

        if events["stop_inference"]:
            break

        if len(action_queue) == 0:
            obs = robot.get_observation()

            # Get wrist camera image and resize to model's expected size
            wrist_img = obs["wrist_cam"]  # (480, 640, 3) RGB
            wrist_224 = resize_image(wrist_img, (224, 224))

            # Map single camera to all 3 model inputs
            # The model was trained with 3 cameras — we duplicate our wrist image
            obs_dict = {
                "observation/joint_position": np.array(
                    [obs[f"joint_{i}"] for i in range(6)], dtype=np.float32
                ),
                "observation/gripper_position": np.array(
                    [obs["gripper"]], dtype=np.float32
                ),
                "prompt": prompt,
                "observation/exterior_image_1_left": wrist_224,
                "observation/wrist_image_left": wrist_224,
                "observation/wrist_image_right": wrist_224,
            }

            t0 = time.perf_counter()
            try:
                result = client.infer(obs_dict)
                action_chunk = result["actions"]
            except Exception as e:
                logger.error(f"Inference error: {e}")
                time.sleep(0.1)
                continue
            t1 = time.perf_counter()

            if not isinstance(action_chunk, np.ndarray):
                action_chunk = np.array(action_chunk)
            if action_chunk.ndim == 1:
                action_chunk = action_chunk.reshape(1, -1)

            action_queue.extend(action_chunk)

            if step % 5 == 0:
                tcp = robot.rtde_rec.getActualTCPPose()
                logger.info(
                    f"Step {step} | Inference {(t1-t0)*1000:.0f}ms | "
                    f"Chunk size {len(action_chunk)} | "
                    f"TCP [{tcp[0]:.3f} {tcp[1]:.3f} {tcp[2]:.3f}] | "
                    f"Gripper {obs['gripper']:.2f}"
                )

        # Execute one action from the chunk
        action = action_queue.popleft()
        action_dict = {
            "joint_0": float(action[0]),
            "joint_1": float(action[1]),
            "joint_2": float(action[2]),
            "joint_3": float(action[3]),
            "joint_4": float(action[4]),
            "joint_5": float(action[5]),
            "gripper": float(action[6]),
        }

        robot.send_action(action_dict)
        step += 1

        # Pace the loop
        dt_s = time.perf_counter() - start_time
        busy_wait(1 / fps - dt_s)


@parser.wrap()
def run_inference(cfg: InferenceConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    client = None
    listener = None

    try:
        client = WebsocketClientPolicy(
            host=cfg.ip,
            port=cfg.port,
            api_key=None,
        )
        logger.info(f"Connected to policy server at {cfg.ip}:{cfg.port}")
        logger.info(f"Server metadata: {client.get_server_metadata()}")

        listener, events = init_keyboard_listener()

        logger.info("Starting inference loop. Press ESC to stop.")
        inference_loop(client, robot, events, cfg.fps, cfg.prompt)

    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e
    finally:
        if client:
            client.close()
        robot.disconnect()
        if listener:
            listener.stop()


def main():
    register_third_party_devices()
    run_inference()


if __name__ == "__main__":
    main()