import gym
from gym.envs.box2d import CarRacing
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np

class CustomCallBack(BaseCallback):
    """Custom CallBack Class."""
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(CustomCallBack, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        """Init callback function."""
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _on_training_start(self) -> None:
        print("Training begun.")
        pass

    def _on_training_end(self) -> None:
        print("Training end.")
        pass

    def _on_step(self) -> bool:
        """Return False to abort training early."""
        # self.locals - gives local variables in a dictionary
        print("On step called.")
        return True

if __name__=='__main__':
    # env = lambda :  CarRacing(
    #     grayscale=1,
    #     show_info_panel=0,
    #     discretize_actions="hard",
    #     frames_per_state=4,
    #     num_lanes=1,
    #     num_tracks=1,
    #     )
    # env = DummyVecEnv([env])
    env = gym.make('CarRacing-v0')
    # env.render(mode='rgb_array')
    # model = DQN("CnnPolicy", env, verbose=1)
    model = PPO("CnnPolicy", env, verbose=0)
    callback = CustomCallBack(check_freq=1000, log_dir="LOGDIR")
    model.learn(total_timesteps=100000, callback=callback)
    
    

