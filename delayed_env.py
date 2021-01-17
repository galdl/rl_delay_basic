# SPDX-License-Identifier: Apache-2.0

from collections import deque
from dqn_agents import DQNAgent
import numpy as np
from dqn_agents import reshape_state
from numpy import sin, cos, pi

CARTPOLE_TRAINED_NON_DELAYED_AGENT_PATH = './saved_agents/i06rfoxy_cartpole_ddqn_no_delay.h5'

class DelayedEnv:
    def __init__(self, orig_env, delay_value):
        self.orig_env = orig_env
        self.env_name = str(self.orig_env)
        self.is_atari_env = 'AtariEnv' in self.env_name
        self.pending_actions = deque()
        self.delay_value = delay_value
        self.state_size = orig_env.observation_space.shape#[0]
        if not self.is_atari_env:
            self.state_size = self.state_size[0]
        self.action_size = orig_env.action_space.n
        self.stored_init_state = None
        self.trained_non_delayed_agent = DQNAgent(state_size=self.state_size,
                                                  action_size=self.action_size, is_delayed_agent=False,
                                                  delay_value=0, epsilon=0, is_atari_env=self.is_atari_env)
        self.pretrained_agent_loaded = False

        if 'CartPole' in self.env_name:
            self.trained_non_delayed_agent.load(CARTPOLE_TRAINED_NON_DELAYED_AGENT_PATH)
            self.pretrained_agent_loaded = True

    def step(self, action):
        if self.delay_value > 0:
            self.pending_actions.append(action)
            if len(self.pending_actions) - 1 >= self.delay_value:
                executed_action = self.pending_actions.popleft()
            else:
                curr_state = reshape_state(self.get_curr_state(), self.is_atari_env, self.state_size)
                executed_action = self.trained_non_delayed_agent.act(curr_state)
        else:
            executed_action = action
        return self.orig_env.step(executed_action)

    def reset(self):
        self.pending_actions.clear()
        return self.orig_env.reset()

    def get_shaped_reward(self, state, orig_reward):
        reward = orig_reward
        if 'CartPole' in self.env_name:
            x, x_dot, theta, theta_dot = state
            r1 = (self.orig_env.x_threshold - abs(x)) / self.orig_env.x_threshold - 0.8
            r2 = (self.orig_env.theta_threshold_radians - abs(
                theta)) / self.orig_env.theta_threshold_radians - 0.5
            reward = r1 + r2
        if 'MountainCar' in self.env_name:
            # # Adjust reward based on car position
            # reward = state[0] + 0.5
            # # Adjust reward for task completion
            # if state[0] >= 0.5:
            #     reward += 1
            position = state[0]
            reward = (position - self.orig_env.goal_position) / ((self.orig_env.max_position - self.orig_env.min_position) * 10)
            # print(position, self.goal_position)
            if position >= 0.1:
                reward += 10
            elif position >= 0.25:
                reward += 50
            elif position >= 0.5:
                reward += 100
        return reward

    def get_pending_actions(self):
        if len(self.pending_actions) == 0 and self.delay_value > 0:
            # reconstruct anticipated trajectory using the oracle
            self.store_initial_state()
            curr_state = self.get_curr_state()
            for i in range(self.delay_value):
                curr_state = reshape_state(curr_state, self.is_atari_env, self.state_size)
                estimated_action = self.trained_non_delayed_agent.act(curr_state)
                self.pending_actions.append(estimated_action)
                curr_state = self.get_next_state(state=None, action=estimated_action)
            self.restore_initial_state()

        return self.pending_actions

    def store_initial_state(self):
        if self.is_atari_env:
            self.stored_init_state = self.orig_env.clone_state()
        else:
            self.stored_init_state = self.orig_env.unwrapped.state

    def restore_initial_state(self):
        if self.is_atari_env:
            self.orig_env.restore_state(self.stored_init_state)
        else:
            self.orig_env.unwrapped.state = self.stored_init_state

    def get_curr_state(self):
        if self.is_atari_env:
            curr_state = self.orig_env.ale.getScreenRGB2()
        else:
            curr_state = self.orig_env.unwrapped.state
        if 'Acrobot' in self.env_name:
            curr_state = np.array([cos(curr_state[0]), sin(curr_state[0]), cos(curr_state[1]), sin(curr_state[1]),
                                   curr_state[2], curr_state[3]])
        return curr_state

    def get_next_state(self, state, action):
        next_state, _, _, _ = self.orig_env.step(action)
        self.orig_env._elapsed_steps -= 1
        return next_state

    def reset_to_state(self, state):
        self.orig_env.unwrapped.state = state
#