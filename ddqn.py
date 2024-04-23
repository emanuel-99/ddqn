import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np
import random
from collections import deque
import cv2
import os
from glob import glob
from datetime import datetime

from tensorflow import keras
from tensorflow.keras import layers
from keras.initializers import HeUniform, Zeros, Constant
from keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt

FLAPPY_ENV = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False, background=None)
FLAPPY_ENV.reset()

BATCH_SIZE = 32
TOTAL_STEPS = 500000
OBSERVE_STEPS = 100000
REPLAY_BUFFER_SIZE = 100000
NUM_STEPS = 1
REPLAY_FREQUENCY = 30
TARGET_MODEL_UPDATE_FREQ = 30
GAMMA = 0.99

REDUCTION_PERCENTAGE = 50
original_height, original_width, channels = FLAPPY_ENV.render().shape
ASPECT_RATIO = original_width / original_height
TARGET_WIDTH = int(original_width * (1 - REDUCTION_PERCENTAGE / 100))
TARGET_HEIGHT = int(TARGET_WIDTH / ASPECT_RATIO)

episode_rewards = []
loss_values = []
epsilon_values = []
episode_lengths = []


def process_image(image):
    resized_image = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
    normalized_image = grayscale_image / 255.0
    return normalized_image


def create_nn(action_size, stack_size):
    image_input = layers.Input(shape=(stack_size, TARGET_HEIGHT, TARGET_WIDTH), name='image_input')
    action_input = layers.Input(shape=(stack_size - 1,), name='action_input')

    x = layers.Conv2D(filters=32, kernel_size=8, strides=4, padding='same', kernel_initializer=HeUniform(),
                      bias_initializer=Zeros())(image_input)
    x = BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=2, padding='same')(x)

    x = layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same', kernel_initializer=HeUniform(),
                      bias_initializer=Zeros())(x)
    x = BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=2, padding='same')(x)

    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer=HeUniform(),
                      bias_initializer=Zeros())(x)
    x = BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=2, padding='same')(x)

    x = layers.Flatten()(x)

    x = keras.Model(inputs=[image_input], outputs=x)

    combined = layers.concatenate([x.output, action_input])

    combined = layers.Dense(512, kernel_initializer=HeUniform(), bias_initializer=Zeros())(combined)
    combined = BatchNormalization()(combined)
    combined = layers.Activation('relu')(combined)
    combined = layers.Dropout(0.15)(combined)

    combined = layers.Dense(action_size, activation='linear', kernel_initializer=HeUniform(),
                            bias_initializer=Constant(0.1))(combined)

    model = keras.Model(inputs=[x.input, action_input], outputs=combined)
    return model


ACTION_SIZE = FLAPPY_ENV.action_space.n
STACK_SIZE = 4
POLICY_NETWORK = create_nn(ACTION_SIZE, STACK_SIZE)
TARGET_NETWORK = create_nn(ACTION_SIZE, STACK_SIZE)

weights_dir = './checkpoints/'
weights_files = glob(os.path.join(weights_dir, 'flappy_*.weights.h5'))
weights_files.sort(key=os.path.getmtime, reverse=True)

POLICY_NETWORK.compile(optimizer=Adam(), loss=MeanSquaredError())
TARGET_NETWORK.compile(optimizer=Adam(), loss=MeanSquaredError())
TARGET_NETWORK.set_weights(POLICY_NETWORK.get_weights())

EPSILON = 0.1
EPSILON_FINAL = 0.0001
REWARD_TARGET = 10
STEPS_TO_TAKE = REWARD_TARGET
REWARD_INCREMENT = 1
REWARD_THRESHOLD = 0
EPSILON_DELTA = (EPSILON - EPSILON_FINAL) / STEPS_TO_TAKE


def decay_epsilon(step_no, observe_steps, episode_reward):
    global EPSILON
    global REWARD_THRESHOLD

    if step_no < observe_steps:
        return EPSILON
    if EPSILON > EPSILON_FINAL and episode_reward > REWARD_THRESHOLD:
        EPSILON = max(EPSILON_FINAL, EPSILON - EPSILON_DELTA)
        REWARD_THRESHOLD += REWARD_INCREMENT
    return EPSILON


def epsilon_greedy(state_stack, action_stack):
    if random.random() > EPSILON:
        Q = POLICY_NETWORK.predict([state_stack, action_stack], batch_size=1, verbose=0)
        action = np.argmax(Q)
    else:
        action = random.choice(np.arange(ACTION_SIZE))
    return action


def sample(buffer, batch_size, num_steps):
    sample_data = []

    sample_indexes = np.random.randint(0, len(buffer) - num_steps, size=batch_size)

    for index in sample_indexes:

        states_frame = []
        actions_frame = []
        rewards_frame = []
        next_states_frame = []
        dones_frame = []

        for i in range(num_steps):
            state, action, reward, next_state, done = buffer[index + i]
            states_frame.append(state)
            actions_frame.append(action)
            rewards_frame.append(reward)
            next_states_frame.append(next_state)
            dones_frame.append(done)

        sample_data.append((states_frame, actions_frame, rewards_frame, next_states_frame, dones_frame))

    return sample_data


def replay(policy_network, target_network, replay_buffer, num_steps, gamma, batch_size):
    if len(replay_buffer) < batch_size + num_steps + STACK_SIZE:
        return

    sample_data = sample(replay_buffer, batch_size, num_steps)

    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*sample_data)

    state_batch_arr = np.array(state_batch)[:, 0]
    next_state_batch_arr = np.array(next_state_batch)[:, -1]

    action_batch_arr = np.array(action_batch)[:, 0]
    sized_action_batch_arr = action_batch_arr[:, :-1]

    next_state_action_batch_arr = np.array(action_batch)[:, -1]
    next_sized_action_batch_arr = next_state_action_batch_arr[:, 1:]

    done_batch_arr = np.array(done_batch)
    reward_batch_arr = np.array(reward_batch)

    Q_policy = policy_network.predict_on_batch([state_batch_arr, sized_action_batch_arr])
    Q_policy_next = policy_network.predict_on_batch([next_state_batch_arr, next_sized_action_batch_arr])
    next_policy_actions = np.argmax(Q_policy_next, axis=1)
    Q_target = target_network.predict_on_batch([next_state_batch_arr, next_sized_action_batch_arr])

    for i, reward in enumerate(reward_batch_arr):
        discounts = np.power(gamma, np.arange(len(reward)))
        n_step_return = np.sum(reward * discounts)
        if done_batch_arr[i, -1]:
            estimated_q = Q_target[[i], next_policy_actions[i]]
            n_step_return += np.power(gamma, num_steps) * estimated_q
        Q_policy[i, action_batch_arr[:, -1][i]] = n_step_return

    loss = policy_network.train_on_batch([state_batch_arr, sized_action_batch_arr], Q_policy)
    loss_values.append(loss)


def train(env, policy_network, target_network, total_steps, observe_steps, num_steps, gamma, batch_size,
          target_model_update_freq, replay_frequency):
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    total_parameter_updates = 0
    i_step = 0
    max_reward = 0
    max_steps = 0

    while i_step < total_steps:

        env.reset()
        state = process_image(env.render())
        state_stack = deque(maxlen=STACK_SIZE)
        action_stack = deque(maxlen=STACK_SIZE)
        state_stack.extend([state] * STACK_SIZE)
        action_stack.extend([0] * STACK_SIZE)

        step = 0
        total_reward = 0
        action = action_stack[-1]
        skipped_steps = 4

        while True:

            if step % skipped_steps == 0:
                if i_step < observe_steps:
                    action = np.random.randint(ACTION_SIZE)
                else:
                    actions = np.array([action_stack])
                    action = epsilon_greedy(np.array([state_stack]), actions[:, 1:])

            action_stack.append(action)

            next_state, reward, terminated, truncated, info = env.step(action)
            if reward == 0.1:
                reward = 0.001

            total_reward += reward

            next_state = process_image(env.render())
            next_state_stack = state_stack + deque([next_state])
            replay_buffer.append((state_stack, action_stack, reward, next_state_stack, terminated or truncated))

            if i_step > observe_steps and i_step % replay_frequency == 0:
                total_parameter_updates += 1
                replay(policy_network, target_network, replay_buffer, num_steps, gamma, batch_size)

            if total_parameter_updates % target_model_update_freq == 0:
                target_network.set_weights(policy_network.get_weights())

            step += 1
            i_step += 1
            state_stack = next_state_stack

            if terminated or truncated:
                print(
                    "Episode finished after {} timesteps, total reward: {}, gamma: {}, epsilon: {}, step count: {}".format(
                        step,
                        total_reward,
                        gamma,
                        EPSILON,
                        i_step))

                epsilon_values.append(EPSILON)
                decay_epsilon(i_step, observe_steps, total_reward)
                episode_rewards.append(total_reward)
                episode_lengths.append(step)
                if total_reward > max_reward:
                    max_reward = total_reward
                    max_steps = step

                break

    current_date = datetime.now().strftime('%Y-%m-%d')
    video_path = f'flappy_bird_{current_date}.avi'
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), 30, (TARGET_WIDTH, TARGET_HEIGHT),
                            isColor=False)
    for experience in replay_buffer:
        state, action, reward, next_state, done = experience
        video.write((state[-1] * 255.0).astype(np.uint8))
    video.release()

    weights_file_path = f'./checkpoints/flappy_{current_date}.weights.h5'

    policy_network.save_weights(weights_file_path, overwrite=True)

    env.close()

    print('Max reward: ', max_reward, ', max steps: ', max_steps)


train(FLAPPY_ENV, POLICY_NETWORK, TARGET_NETWORK, TOTAL_STEPS, OBSERVE_STEPS, NUM_STEPS, GAMMA, BATCH_SIZE,
      TARGET_MODEL_UPDATE_FREQ, REPLAY_FREQUENCY)

current_date = datetime.now().strftime('%Y-%m-%d')
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Reward over Episodes')
plt.savefig(f'reward_plot_{current_date}.png')
plt.show()

plt.plot(epsilon_values)
plt.xlabel('Training Step')
plt.ylabel('Epsilon Value')
plt.title('Epsilon Value over Training Steps')
plt.savefig(f'epsilon_plot_{current_date}.png')
plt.show()

plt.plot(episode_lengths)
plt.xlabel('Episode')
plt.ylabel('Episode Length')
plt.title('Episode Length over Episodes')
plt.savefig(f'episode_length_plot_{current_date}.png')
plt.show()

plt.plot(loss_values)
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('Training Loss over Steps')
plt.savefig(f'loss_plot_{current_date}.png')
plt.show()
