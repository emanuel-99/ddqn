import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np
import random
from collections import deque
import cv2
import os
from glob import glob
from datetime import datetime

import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from keras.initializers import HeUniform, Zeros, Constant
from keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

FLAPPY_ENV = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False, background=None)
SEED = 55
FLAPPY_ENV.action_space.seed(SEED)
FLAPPY_ENV.observation_space.seed(SEED)
np.random.seed(SEED)
FLAPPY_ENV.reset(seed=SEED)

BATCH_SIZE = 32
TOTAL_STEPS = 80000
OBSERVE_STEPS = 10000
REPLAY_BUFFER_SIZE = 40000
SKIP_STEPS = 5
NUM_STEPS = SKIP_STEPS
REPLAY_FREQUENCY = 5
TARGET_MODEL_UPDATE_FREQ = 20
GAMMA = 0.95

REDUCTION_PERCENTAGE = 50
original_height, original_width, channels = FLAPPY_ENV.render().shape
ASPECT_RATIO = original_width / original_height
TARGET_WIDTH = int(original_width * (1 - REDUCTION_PERCENTAGE / 100))
TARGET_HEIGHT = int(TARGET_WIDTH / ASPECT_RATIO)


def process_image(image):
    resized_image = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
    normalized_image = grayscale_image / 255.0
    return normalized_image

def create_nn(action_size, stack_size):
    image_input = layers.Input(shape=(stack_size - 1, TARGET_HEIGHT, TARGET_WIDTH), name='image_input')
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

    x = layers.Flatten()(x)

    x = keras.Model(inputs=[image_input], outputs=x)

    combined = layers.concatenate([x.output, action_input])

    combined = layers.Dense(64, kernel_initializer=HeUniform(), bias_initializer=Zeros())(combined)
    combined = BatchNormalization()(combined)
    combined = layers.Activation('relu')(combined)
    combined = layers.Dropout(0.15)(combined)

    combined = layers.Dense(64, kernel_initializer=HeUniform(), bias_initializer=Zeros())(combined)
    combined = BatchNormalization()(combined)
    combined = layers.Activation('relu')(combined)
    combined = layers.Dropout(0.15)(combined)
    combined = layers.Dense(action_size, activation='linear', kernel_initializer=HeUniform(),
                            bias_initializer=Constant(0.1))(combined)

    model = keras.Model(inputs=[x.input[0], action_input], outputs=combined)
    return model


ACTION_SIZE = FLAPPY_ENV.action_space.n
STACK_SIZE = 5
POLICY_NETWORK = create_nn(ACTION_SIZE, STACK_SIZE)
tensorflow.keras.utils.plot_model(POLICY_NETWORK,  show_shapes=True, show_layer_names=True, to_file='model.png')
exit(0)
TARGET_NETWORK = create_nn(ACTION_SIZE, STACK_SIZE)

REPLAY_BUFFER = deque(maxlen=REPLAY_BUFFER_SIZE)

weights_dir = './checkpoints/'
weights_files = glob(os.path.join(weights_dir, 'flappy_*.weights.h5'))
weights_files.sort(key=os.path.getmtime, reverse=True)
if weights_files:
    most_recent_weights_file = weights_files[0]
    POLICY_NETWORK.load_weights(most_recent_weights_file)

lr = 0.0001
clipnorm = 1.0
clipvalue = 0.5
POLICY_NETWORK.compile(optimizer=Adam(learning_rate=lr, clipnorm=clipnorm, clipvalue=clipvalue),
                       loss=MeanSquaredError())
TARGET_NETWORK.set_weights(POLICY_NETWORK.get_weights())

EPSILON = 1
EPSILON_FINAL = 0.01
EPSILON_DECREMENT = (EPSILON / EPSILON_FINAL) ** 1 / TOTAL_STEPS
REWARD_TARGET = 10
STEPS_TO_TAKE = REWARD_TARGET
REWARD_INCREMENT = 1
REWARD_THRESHOLD = 0
EPSILON_DELTA = (EPSILON - EPSILON_FINAL) / STEPS_TO_TAKE


def decay_epsilon(step_no, observe_steps, transition_reward):
    global EPSILON
    global REWARD_THRESHOLD

    if step_no < observe_steps:
        return EPSILON
    if EPSILON > EPSILON_FINAL and transition_reward > REWARD_THRESHOLD:
        EPSILON = max(EPSILON_FINAL, EPSILON - EPSILON_DELTA)
        REWARD_THRESHOLD += REWARD_INCREMENT
    return EPSILON


def epsilon_greedy(state_stack, action_stack):
    if random.random() > EPSILON:
        state_array = np.array([state_stack])
        action_array = np.array([action_stack])
        Q = POLICY_NETWORK.predict([state_array[:, 1:], action_array[:, 1:]], batch_size=1, verbose=0)
        action = np.argmax(Q)
    else:
        action = random.choice(np.arange(ACTION_SIZE))
    return action


def sample():
    sample_data = []

    sample_indexes = np.random.choice(len(REPLAY_BUFFER) - NUM_STEPS, BATCH_SIZE, replace=False)

    for index in sample_indexes:
        states_frame = []
        actions_frame = []
        rewards_frame = []
        dones_frame = []

        for i in range(NUM_STEPS):
            state, action, reward, done = REPLAY_BUFFER[index + i]
            states_frame.append(state)
            actions_frame.append(action)
            rewards_frame.append(reward)
            dones_frame.append(done)

        sample_data.append((states_frame, actions_frame, rewards_frame, dones_frame))

    return sample_data


def replay():
    if len(REPLAY_BUFFER) < BATCH_SIZE:
        return

    sample_data = sample()

    states, actions, rewards, dones = zip(*sample_data)

    current_action = np.array(actions)[:, 0][:, :-1]
    next_action = np.array(actions)[:, 0][:, 1:]

    current_state = np.array(states)[:, 0][:, :-1]
    next_state = np.array(states)[:, -1][:, 1:]
    dones_array = np.array(dones)
    Q_policy = POLICY_NETWORK.predict_on_batch([current_state, current_action])
    Q_next = POLICY_NETWORK.predict_on_batch([next_state, next_action])
    Q_target = TARGET_NETWORK.predict_on_batch([next_state, next_action])

    next_policy_actions = np.argmax(Q_next, axis=1)  #max actions
    batch_indices = np.arange(BATCH_SIZE, dtype=np.int32)

    for i, reward in enumerate(np.array(rewards)):
        discounts = np.power(GAMMA, np.arange(len(reward)))
        n_step_return = np.sum(reward * discounts)
        result = True in dones_array[i]
        if not result:
            estimated_q = Q_target[i, next_policy_actions[i]]
            n_step_return += np.power(GAMMA, NUM_STEPS) * estimated_q
        Q_policy[i, next_action[:, -1]] = n_step_return

    POLICY_NETWORK.train_on_batch([current_state, current_action], Q_policy)


def train():
    global EPSILON
    i_step = 0
    total_parameter_updates = 0

    while i_step < TOTAL_STEPS:

        step = 0
        total_reward = 0

        FLAPPY_ENV.reset()
        ACTION_STACK = deque(maxlen=STACK_SIZE)
        STATE_STACK = deque(maxlen=STACK_SIZE)
        state = process_image(FLAPPY_ENV.render())
        STATE_STACK.extend([state] * STACK_SIZE)
        ACTION_STACK.extend([0] * STACK_SIZE)
        action = STATE_STACK[-1]

        transition_reward = 0

        while True:

            if step % SKIP_STEPS == 0:
                if i_step < OBSERVE_STEPS:
                    action = random.choice(np.arange(ACTION_SIZE))
                else:
                    action = epsilon_greedy(STATE_STACK, ACTION_STACK)

                ACTION_STACK.append(action)

                next_state, reward, terminated, truncated, info = FLAPPY_ENV.step(action)
                total_reward += reward
                transition_reward += reward

                next_state = process_image(FLAPPY_ENV.render())
                STATE_STACK.append(next_state)

                REPLAY_BUFFER.append(
                    (np.array(STATE_STACK), np.array(ACTION_STACK), transition_reward, terminated or truncated))

                if i_step > OBSERVE_STEPS and i_step % REPLAY_FREQUENCY == 0:
                    total_parameter_updates += 1
                    replay()
                    if EPSILON > EPSILON_FINAL:
                        EPSILON = EPSILON - EPSILON_DECREMENT

                if total_parameter_updates % TARGET_MODEL_UPDATE_FREQ == 0:
                    TARGET_NETWORK.set_weights(POLICY_NETWORK.get_weights())

                if terminated or truncated:
                    print(
                        "Episode finished after {} timesteps, total reward: {}, gamma: {}, epsilon: {}, step count: {}".format(
                            step,
                            total_reward,
                            GAMMA,
                            EPSILON,
                            i_step))
                    break

                transition_reward = 0
            else:
                next_state, reward, terminated, truncated, info = FLAPPY_ENV.step(action)
                total_reward += reward
                transition_reward += reward
                if terminated or truncated:
                    ACTION_STACK.append(action)
                    next_state = process_image(FLAPPY_ENV.render())
                    STATE_STACK.append(next_state)

                    REPLAY_BUFFER.append(
                        (np.array(STATE_STACK), np.array(ACTION_STACK), transition_reward, terminated or truncated))

                    print(
                        "Episode finished after {} timesteps, total reward: {}, gamma: {}, epsilon: {}, step count: {}".format(
                            step,
                            total_reward,
                            GAMMA,
                            EPSILON,
                            i_step))
                    break

            i_step += 1
            step += 1

    current_date = datetime.now().strftime('%Y-%m-%d')
    video_path = f'flappy_bird_{current_date}.avi'
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), 30, (TARGET_WIDTH, TARGET_HEIGHT),
                            isColor=False)
    for experience in REPLAY_BUFFER:
        state, action, reward, done = experience
        video.write((state[-1] * 255.0).astype(np.uint8))
    video.release()

    weights_file_path = f'./checkpoints/flappy_{current_date}.weights.h5'

    POLICY_NETWORK.save_weights(weights_file_path, overwrite=True)

    FLAPPY_ENV.close()


train()
