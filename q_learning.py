from gym.version import VERSION
print(VERSION) # make sure the new version of gym is loaded

import gym
import numpy as np
import tensorflow as tf

class Agent:
    def __init__(self, obs_shape, act_size):
        self.obs_shape = obs_shape
        self.act_size = act_size

    def network(self, train=True):
        inputs = tf.keras.Input(shape=(self.obs_shape,), name="input")
        x = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(), name="dense_1")(inputs)
        x = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(), name="dense_2")(x)
        outputs = tf.keras.layers.Dense(self.act_size, name="output")(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name="nn", trainable=train)
        return model

class Util:
    def __init__(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=50, decay_rate=0.9))
        self.history = []

    def record_history(self, current_state, action, reward, next_state):
        self.history.append([current_state, action, reward, next_state])

    def td_loss(self, nn, discount=0.99):
        loss = []
        for current_state, action, reward, next_state in self.history:
            binary_action = [0.0] * nn.output.shape[1]
            binary_action[action] = 1.0
            binary_action = tf.constant([binary_action])
            q_current = nn(tf.convert_to_tensor([current_state]))
            max_q_next = tf.math.reduce_max(nn(tf.convert_to_tensor([next_state])))
            loss.append(tf.math.square((reward + discount * max_q_next - q_current) * binary_action))
        return tf.math.reduce_mean(loss, axis=0)

    def update_model(self, nn):
        with tf.GradientTape() as tape:
            loss = self.td_loss(nn)
        grads = tape.gradient(loss, nn.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, nn.trainable_variables))
        self.history = []

env = gym.make("CartPole-v1")
agent = Agent(4, 2).network()
utility = Util()

# train
epsilon = 0.3
i, early_stop = 0, 0
n_game = 2000
while i < n_game:
    current_state = env.reset()
    step = 0
    while True:
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = tf.math.argmax(tf.reshape(agent(tf.convert_to_tensor([current_state])), [-1])).numpy()
        next_state, reward, done, info = env.step(action)
        step += 1
        utility.record_history(current_state, action, reward, next_state)
        current_state = next_state
        if len(utility.history) == 50:
            utility.update_model(agent)
        epsilon = max(epsilon * 0.99, 0.05)
        if done:
            print(i, step)
            i += 1
            if step >= 500:
                early_stop += 1
            else:
                early_stop = 0
            if early_stop >= 10:
                i = n_game
            break

# test
for i in range(10):
    env.close()
    env = gym.make("CartPole-v1")
    state = env.reset()
    step = 0
    while True:
        action = tf.math.argmax(tf.reshape(agent(tf.convert_to_tensor([state])), [-1])).numpy()
        state, reward, done, info = env.step(action)
        step += 1
        if done:
            print(i, step)
        break

env.close()

# save agent
# agent.save("cartpole_dql")
