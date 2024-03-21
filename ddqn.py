import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np
import sys
import random
from collections import deque
import cv2
# from google.colab import files

import matplotlib.pyplot as plt
from keras.src.initializers import HeUniform, Zeros, Constant
from keras.src.layers import BatchNormalization

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.initializers import RandomUniform

FLAPPY_ENV = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False, background=None)
SEED = 42
FLAPPY_ENV.action_space.seed(SEED)
FLAPPY_ENV.observation_space.seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

FLAPPY_ENV.reset(seed=SEED)
REDUCTION_PERCENTAGE = 25
original_height, original_width, channels = FLAPPY_ENV.render().shape
ASPECT_RATIO = original_width / original_height
TARGET_WIDTH = int(original_width * (1 - REDUCTION_PERCENTAGE / 100))
TARGET_HEIGHT = int(TARGET_WIDTH / ASPECT_RATIO)

BATCH_SIZE = 32
NUM_STEPS = 10
NUM_EPISODES = 1000
TARGET_MODEL_UPDATE = 5
MAX_STEPS_EPISODE = 500
REPLAY_BUFFER_SIZE = 100000

GAMMA_START = 0.95
GAMMA_FINAL = 0.5
GAMMA_STEP = 0.01
THRESHOLD = 10


def process_image(image):
    resized_image = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
    normalized_image = grayscale_image / 255.0
    return normalized_image


def create_nn(action_size):
    image_input = layers.Input(shape=(TARGET_HEIGHT, TARGET_WIDTH, 1), name='image_input')
    x = layers.Conv2D(filters=32, kernel_size=8, strides=4, padding='same', kernel_initializer=HeUniform(),
                      bias_initializer=Zeros())(image_input)
    x = BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same', kernel_initializer=HeUniform(),
                      bias_initializer=Zeros())(x)
    x = BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(64, kernel_initializer=HeUniform(), bias_initializer=Zeros())(x)
    x = BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.20)(x)

    x = layers.Dense(64, kernel_initializer=HeUniform(), bias_initializer=Zeros())(x)
    x = BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.20)(x)
    output = layers.Dense(action_size, activation='linear', kernel_initializer=HeUniform(),
                          bias_initializer=Constant(0.1))(x)

    model = keras.Model(inputs=[image_input], outputs=output)
    model.compile(optimizer=Adam(), loss=MeanSquaredError())
    return model


ACTION_SIZE = FLAPPY_ENV.action_space.n
POLICY_NETWORK = create_nn(ACTION_SIZE)
TARGET_NETWORK = create_nn(ACTION_SIZE)
TARGET_NETWORK.set_weights(POLICY_NETWORK.get_weights())

EPSILON_START = 1
EPSILON_FINAL = 0.1
EPSILON_DECAY_STEPS = 800


def decay_epsilon(episode):
    slope = (EPSILON_START - EPSILON_FINAL) / EPSILON_DECAY_STEPS
    epsilon = max(EPSILON_FINAL, EPSILON_START - slope * episode)
    return epsilon


def epsilon_greedy(Q, eps):
    action_size = Q.shape[0]
    if random.random() > eps:
        return np.argmax(Q)
    else:
        return random.choice(np.arange(action_size))


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
            if done:
                break

        if len(states_frame) < num_steps:
            continue

        sample_data.append((states_frame, actions_frame, rewards_frame, next_states_frame, dones_frame))

    return sample_data


def replay(policy_network, target_network, replay_buffer, num_steps, gamma, batch_size):
    if len(replay_buffer) < batch_size + num_steps:
        return

    sample_data = sample(replay_buffer, batch_size, num_steps)

    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*sample_data)

    # take first state & action from each frame and last state from each frame => batch size array of first states from each frame and last states from each frame
    state_batch_arr = np.array(state_batch)[:, 0]
    action_batch_arr = np.array(action_batch)[:, 0]
    next_state_batch_arr = np.array(next_state_batch)[:, -1]
    done_batch_arr = np.array(done_batch)
    reward_batch_arr = np.array(reward_batch)

    Q_policy = policy_network.predict_on_batch(next_state_batch_arr)
    Q_target = target_network.predict_on_batch(state_batch_arr)

    for i, reward in enumerate(reward_batch_arr):
        discounts = np.power(gamma, np.arange(len(reward)))
        n_step_return = np.sum(reward * discounts)
        if not done_batch_arr[i, -1]:
            next_state_value = np.max(Q_policy[i][0])
            n_step_return += np.power(gamma, num_steps) * next_state_value
        Q_target[i, action_batch_arr[i]] = n_step_return

    policy_network.train_on_batch(state_batch_arr, Q_target)


def train(env, policy_network, target_network, num_episodes, batch_size, num_steps, gamma, target_model_update,
          plot_every=10):
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    tmp_scores = deque(maxlen=plot_every)
    avg_scores = deque(maxlen=num_episodes // plot_every)

    total_step = 0

    for i_episode in range(1, num_episodes + 1):
        epsilon = decay_epsilon(i_episode)

        env.reset()
        state = process_image(env.render())
        step = 0
        total_reward = 0

        while True:
            temp_state = np.expand_dims(state, axis=0)
            Q = policy_network.predict([temp_state], verbose=0)
            action = epsilon_greedy(Q[0], epsilon)

            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            next_state = process_image(env.render())
            replay_buffer.append((state, action, reward, next_state, terminated or truncated))

            replay(policy_network, target_network, replay_buffer, num_steps, gamma, batch_size)

            if terminated or truncated:
                print("Episode ({}/{}) finished after {} timesteps, total reward: {}, gamma: {}, epsilon: {}".format(
                    i_episode,
                    num_episodes,
                    step,
                    total_reward,
                    gamma,
                    epsilon))
                tmp_scores.append(total_reward)
                break

            if step > MAX_STEPS_EPISODE:
                print("Episode OVER finished after {} timesteps".format(step))
                break

            state = next_state
            step += 1

        total_step += step

        if i_episode % target_model_update == 0:
            target_network.set_weights(policy_network.get_weights())

        if i_episode % plot_every == 0:
            avg_scores.append(np.mean(tmp_scores))

    video_path = "flappy_bird_video.avi"
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), 1, (TARGET_WIDTH, TARGET_HEIGHT),
                            isColor=False)
    for experience in replay_buffer:
        state, action, reward, next_state, done = experience
        video.write(state)
    video.release()
    # files.download('flappy_bird_video.avi')

    plt.plot(np.linspace(0, num_episodes, len(avg_scores), endpoint=False), np.asarray(avg_scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(avg_scores))

    print(('Total steps for %d episodes: ' % num_episodes), num_steps)

    env.close()


train(FLAPPY_ENV, POLICY_NETWORK, TARGET_NETWORK, NUM_EPISODES, BATCH_SIZE, NUM_STEPS, GAMMA_START, TARGET_MODEL_UPDATE,
      plot_every=10)
