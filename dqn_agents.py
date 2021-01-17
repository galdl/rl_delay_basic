# SPDX-License-Identifier: Apache-2.0

from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from copy import deepcopy
import random
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import numpy as np

def reshape_state(state, is_atari_env, state_size):
    reshaped = state
    if not is_atari_env:
        reshaped = np.reshape(state, [1, state_size])
    else:
        if len(state.shape) < 4:
            reshaped = np.expand_dims(state, axis=0)
    return reshaped

def update_loss(loss, sample_loss):
    if loss is not None and sample_loss is not None:
        for key, val in sample_loss.items():
            if key in loss:
                loss[key] += val
            else:
                loss[key] = val

def concatenate_state_action(state, action):
    out = np.concatenate((state[0], [action]))
    out = np.reshape(out, [1, len(out)])
    return out

class DQNAgent:
    def __init__(self, state_size, action_size, is_atari_env, is_delayed_agent=False, delay_value=0, epsilon_min=0.001,
                epsilon_decay=0.999, learning_rate=0.001, epsilon=1.0, use_m_step_reward=False, use_latest_reward=True,
                 loss='mse', **kwargs):
        self.state_size = state_size
        self.action_size = action_size
        self.is_atari_env = is_atari_env
        mem_len = 50000 if self.is_atari_env else 2000
        self.memory = deque(maxlen=mem_len)
        self.gamma = 0.95    # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay #0.995
        self.learning_rate = learning_rate
        self.sample_buffer = deque()
        self.is_delayed_agent = is_delayed_agent
        self.delay_value = delay_value
        self.model = self._build_model(loss=loss)
        self.use_m_step_reward = use_m_step_reward
        self.use_latest_reward = use_latest_reward


    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        """Huber loss for Q Learning
        References: https://en.wikipedia.org/wiki/Huber_loss
                    https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
        """
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self, loss=None, input_size=None, output_size=None):
        loss = self._huber_loss if loss is 'huber' else loss
        input_size = self.state_size if input_size is None else input_size
        output_size = self.action_size if output_size is None else output_size

        # Neural Net for Deep-Q learning Model
        model = Sequential()
        if self.is_atari_env:
            model.add(Conv2D(32, 8, strides=(4,4), input_shape=input_size, activation='relu'))
            model.add(MaxPool2D())
            model.add(Conv2D(64, 4, strides=(2,2), activation='relu'))
            model.add(MaxPool2D())
            model.add(Conv2D(64, 3, strides=(1,1), activation='relu'))
            model.add(MaxPool2D())
            model.add(Flatten())
            model.add(Dense(64, activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(output_size, activation='linear'))
        else:
            model.add(Dense(24, input_dim=input_size, activation='relu'))
            model.add(Dense(24, activation='relu'))
            model.add(Dense(output_size, activation='linear'))

        model.compile(loss=loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        if self.is_delayed_agent:
            # for earlier time than delay_value, the data is problematic (non-delayed response)
            # Construct modified tuple by keeping old s_t with new a_{t+m}, r_{t+m} s_{t+m+1}
            new_tuple = (state, action, reward, next_state, done)
            self.sample_buffer.append(new_tuple)
            if len(self.sample_buffer) - 1 >= self.delay_value:
                old_tuple = self.sample_buffer.popleft()
                modified_tuple = list(deepcopy(old_tuple))
                modified_tuple[1] = action
                modified_tuple[2] = self.m_step_reward(first_reward=old_tuple[2])
                # trying to use s_{t+1} instead of s_{t+m} as in the original ICML2020 submission
                # modified_tuple[3] = next_state
                modified_tuple = tuple(modified_tuple)
                self.memory.append(modified_tuple)
        else:
            self.memory.append((state, action, reward, next_state, done))

    def act(self, state, eval=False):
        if not eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def m_step_reward(self, first_reward):
        if not self.use_m_step_reward:
            if self.use_latest_reward:
                return self.sample_buffer[-1][2]
            else:
                return first_reward
        else:
            discounted_rew = first_reward
            for i in range(self.delay_value):
                discounted_rew += self.gamma ** (i + 1) * self.sample_buffer[i][2]
            return discounted_rew

    def effective_gamma(self):
        return self.gamma if not self.use_m_step_reward else (self.gamma ** (self.delay_value + 1))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.effective_gamma() *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            # self.model.fit(state, target_f, epochs=1, verbose=0,
            #                callbacks=[WandbCallback()])
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def clear_action_buffer(self):
        self.sample_buffer.clear()


class DDQNAgent(DQNAgent):
    def __init__(self, state_size, action_size, is_atari_env, is_delayed_agent=False, delay_value=0, epsilon_min=0.001,
                epsilon_decay=0.999, learning_rate=0.001, epsilon=1.0, use_m_step_reward=False, use_latest_reward=True):
        super().__init__(state_size, action_size, is_atari_env=is_atari_env, is_delayed_agent=is_delayed_agent, delay_value=delay_value,
                         epsilon_min=epsilon_min, epsilon_decay=epsilon_decay, learning_rate=learning_rate,
                         epsilon=epsilon, use_m_step_reward=use_m_step_reward, use_latest_reward=use_latest_reward,
                         loss='huber')
        # self.model = self._build_model()
        self.target_model = self._build_model(loss='huber')
        self.update_target_model()


    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def train_model(self, batch):
        state_vec, action_vec, reward_vec, next_state_vec, done_vec = batch
        target = self.model.predict(state_vec)
        # a = self.model.predict(next_state)[0]
        t = self.target_model.predict(next_state_vec)#[0]
        not_done_arr = np.invert(np.asarray(done_vec))
        new_targets = reward_vec + not_done_arr * self.effective_gamma() * np.amax(t, axis=1)
        for i in range(len(batch[0])):
            target[i][action_vec[i]] = new_targets[i]
        # target[0][action] = reward + self.gamma * t[np.argmax(a)]
        train_history = self.model.fit(state_vec, target, epochs=1, verbose=0)
        q_loss = train_history.history['loss'][0]
        loss_dict = {'q_loss': q_loss}
        return loss_dict

    def _create_batch(self, indices):
        state_vec, action_vec, reward_vec, next_state_vec, done_vec = [], [], [], [], []
        for i in indices:
            data = self.memory[i]
            state, action, reward, next_state, done = data
            state_vec.append(np.array(state, copy=False))
            action_vec.append(action)
            reward_vec.append(reward)
            next_state_vec.append(np.array(next_state, copy=False))
            done_vec.append(done)
        return np.concatenate(state_vec, axis=0), action_vec, reward_vec, np.concatenate(next_state_vec, axis=0), done_vec

    def replay(self, batch_size):
        loss = {}
        indices = np.random.choice(len(self.memory), batch_size)
        batch = self._create_batch(indices)
        sample_loss = self.train_model(batch)
        update_loss(loss, sample_loss)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

class DDQNPlanningAgent(DDQNAgent):
    def __init__(self, state_size, action_size, is_atari_env, is_delayed_agent=False, delay_value=0, epsilon_min=0.001,
                 epsilon_decay=0.999, learning_rate=0.001, epsilon=1.0, use_m_step_reward=False,
                 use_latest_reward=True, env=None, use_learned_forward_model=True):
        super().__init__(state_size, action_size, is_atari_env=is_atari_env, is_delayed_agent=is_delayed_agent, delay_value=delay_value,
                         epsilon_min=epsilon_min, epsilon_decay=epsilon_decay, learning_rate=learning_rate,
                         epsilon=epsilon, use_m_step_reward=use_m_step_reward, use_latest_reward=use_latest_reward)
        self.use_learned_forward_model = use_learned_forward_model
        if self.use_learned_forward_model:
            keras_forward_model = self._build_model(loss='mse', input_size=self.state_size + 1, output_size=self.state_size)
            self.forward_model = ForwardModel(keras_forward_model)
        else:
            self.forward_model = env

    def train_model(self, batch):
        loss_dict = super().train_model(batch)
        if self.use_learned_forward_model and self.delay_value > 0:
            state_vec, action_vec, _, next_state_vec, _ = batch
            act_t = np.asarray([action_vec]).transpose()
            concat_vec = np.concatenate((state_vec, act_t), axis=1)
            train_history = self.forward_model.keras_model.fit(concat_vec, next_state_vec, epochs=1, verbose=0)
            f_model_loss = train_history.history['loss'][0]
            loss_dict['f_model_loss'] = f_model_loss
        return loss_dict

    def act(self, state, pending_actions, eval):
        if not eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        last_state = state
        if self.delay_value > 0:
            if not self.use_learned_forward_model:
                self.forward_model.store_initial_state()
                # initial_state = deepcopy(state)
            for curr_action in pending_actions:
                last_state = self.forward_model.get_next_state(state=last_state, action=curr_action)
            if not self.use_learned_forward_model:
                self.forward_model.restore_initial_state()
        last_state_r = reshape_state(last_state, self.is_atari_env, self.state_size)
        act_values = self.model.predict(last_state_r)
        return np.argmax(act_values[0])  # returns best action for last state

    def memorize(self, state, action, reward, next_state, done):
        # for earlier time than delay_value, the data is problematic (non-delayed response)
        # Construct modified tuple by keeping old s_t with new a_{t+m}, r_{t+m} s_{t+m+1}
        new_tuple = (state, action, reward, next_state, done)
        self.sample_buffer.append(new_tuple)
        if len(self.sample_buffer) - 1 >= self.delay_value:
            old_tuple = self.sample_buffer.popleft()
            modified_tuple = list(deepcopy(old_tuple))
            # build time-coherent tuple from new tuple and old action
            modified_tuple[0] = state
            # modified_tuple[1] = action
            modified_tuple[2] = reward #self.m_step_reward(first_reward=old_tuple[2])
            modified_tuple[3] = next_state
            modified_tuple = tuple(modified_tuple)
            self.memory.append(modified_tuple)

class ForwardModel:
    def __init__(self, keras_model):
        self.keras_model = keras_model

    def get_next_state(self, state, action):
        input = concatenate_state_action(state, action)
        return self.keras_model.predict(input)

    def reset_to_state(self, state):
        # not necessary here. Only used if the forwrad_model is the actual env instance
        pass