import os
import sys
import signal
import warnings

# Suppress noisy warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")

def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    # classifier=False during demo recording: the human decides success,
    # not the learned classifier (which may not exist yet)
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)
    
    obs, info = env.reset()
    print("Reset done. Begin guiding!")
    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0
    
    try:
        while success_count < success_needed:
            actions = np.zeros(env.action_space.sample().shape) 
            next_obs, rew, done, truncated, info = env.step(actions)
            returns += rew
            if "intervene_action" in info:
                actions = info["intervene_action"]
            transition = copy.deepcopy(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=1.0 - done,
                    dones=done,
                    infos=info,
                )
            )
            trajectory.append(transition)
            
            pbar.set_description(f"Return: {returns}")

            obs = next_obs
            if done:
                if info["succeed"]:
                    for transition in trajectory:
                        transitions.append(copy.deepcopy(transition))
                    success_count += 1
                    pbar.update(1)
                trajectory = []
                returns = 0
                obs, info = env.reset()

    except KeyboardInterrupt:
        print(f"\nInterrupted. Got {success_count}/{success_needed} demos.")

    finally:
        pbar.close()
        # Save whatever we have
        if transitions:
            if not os.path.exists("./demo_data"):
                os.makedirs("./demo_data")
            uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = f"./demo_data/{FLAGS.exp_name}_{len(transitions)}_transitions_{uuid}.pkl"
            with open(file_name, "wb") as f:
                pkl.dump(transitions, f)
            print(f"Saved {len(transitions)} transitions ({success_count} successes) to {file_name}")
        else:
            print("No successful demos recorded.")

        # Clean shutdown
        try:
            env.close()
        except Exception:
            pass
        os._exit(0)


if __name__ == "__main__":
    app.run(main)