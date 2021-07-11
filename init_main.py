# SPDX-License-Identifier: Apache-2.0
import gym
from delayed_env import DelayedEnv
import wandb
import warnings

from keras import backend as K
import tensorflow as tf


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


def init_main():
    hyperparameter_defaults = dict(
        is_delayed_agent=False,
        double_q=True,
        delay_value=5,
        epsilon_decay=0.999, # Cartpole: 0.999, Acrobot: 0.9999, MountainCar: 0.99999
        epsilon_min=0.001, #0.001
        learning_rate=0.005, # Cartpole & Acrobot: 0.005, #mountainCar: 0.0001
        seed=1,
        epsilon=1.0,
        use_m_step_reward=False,
        use_latest_reward=False,
        use_reward_shaping=True,
        physical_noise_std_ratio=0.1, # default: 0.1
        env_name='CartPole-v1', #'CartPole-v1', 'Acrobot-v1', 'MountainCar-v0'
        train_freq=1,
        target_network_update_freq=300,
        use_learned_forward_model=True,
        agent_type='delayed', #'delayed', 'augmented', 'oblivious'
        # total_steps=3000, replaced with a delay-dependent function
    )
    # Pass your defaults to wandb.init
    wandb.init(config=hyperparameter_defaults)
    config = wandb.config
    if 'CartPole' in config.env_name or 'Acrobot' in config.env_name:
        try:
            orig_env = gym.make(config.env_name, physical_noise_std_ratio=config.physical_noise_std_ratio)
        except TypeError as e:
            warnings.warn('{} gym env has not been modified as needed to support added noise. See README.md for '
                          'instructions.\nRunning original noiseless version instead.'.format(config.env_name))
            orig_env = gym.make(config.env_name)
    else:
        orig_env = gym.make(config.env_name)
    # orig_env = DiscretizeActions(orig_env) # for mujoco envs
    delayed_env = DelayedEnv(orig_env, config.delay_value)
    state_size = orig_env.observation_space.shape#[0]
    if not delayed_env.is_atari_env:
        state_size = state_size[0]
    action_size = orig_env.action_space.n
    done = False
    batch_size = 32
    return config, delayed_env, state_size, action_size, done, batch_size





