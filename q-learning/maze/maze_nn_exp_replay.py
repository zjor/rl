"""
DQL Experience replay
"""

import numpy as np
import tensorflow as tf
from dqn import ReplayMemory
from maze import MAPS, Environment, AbstractAgent


def avg(a, n):
    return np.convolve(a, np.ones(n) / n, mode='valid')[:-1]


class Agent(AbstractAgent):
    actions = ['←', '→', '↑', '↓']

    def __init__(self, env, model, lr=0.8, y=0.95, step_cost=.0, living_cost=.0, episode_length=100,
                 memory_capacity=100, batch_size=10, eps=0.5, eps_decay=0.999):
        AbstractAgent.__init__(self, eps, eps_decay)
        self.env = env
        self.model = model
        self.lr = lr
        self.y = y
        self.step_cost = step_cost
        self.living_cost = living_cost
        self.s0 = env.field.index('s')
        self.episode_length = episode_length
        self.rewards = []
        self.losses = []
        self.memory = ReplayMemory(memory_capacity)
        self.batch_size = batch_size

    def step(self, state, action):
        return self.env.step(state, action)

    def print_policy(self):
        for y in range(self.env.height):
            for x in range(self.env.width):
                s = y * self.env.width + x
                cell = self.env.field[s]
                if not (cell == '.' or cell == 's'):
                    print(cell, end='')
                    continue
                q_predicted = self.predict_q(s)
                a = np.argmax(q_predicted)
                print(Agent.actions[a], end='')
            print()

    def _encode_state(self, s):
        z = np.zeros(self.env.length)
        z[s] = 1.0
        return np.array([z])

    def predict_q(self, s):
        return self.model.predict(self._encode_state(s))[0]

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        for s, a, s1, r in transitions:
            q_predicted = self.predict_q(s)
            q_target = q_predicted
            q_target[a] = r + self.y * self.predict_q(s1).max()

            history = self.model.fit(
                x=self._encode_state(s),
                y=np.array([q_target]),
                epochs=1,
                verbose=False)
            self.losses.append(history.history["loss"][-1])

    def run_episode(self):
        AbstractAgent.run_episode(self)
        s = self.s0
        self.rewards.append(.0)
        for j in range(self.episode_length):
            q_predicted = self.predict_q(s)
            a = np.argmax(q_predicted)
            a = self.select_action(a)
            s1, r, over = self.step(s, Agent.actions[a])
            if s != s1:
                r -= self.step_cost
            r -= self.living_cost
            self.memory.push(s, a, s1, r)
            s = s1
            self.optimize()
            self.rewards[-1] += r
            if over:
                break


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.random.seed(42)
    tf.random.set_seed(42)
    env = Environment(world=MAPS["classic"], win_reward=5.0, death_reward=-5.0)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=env.length, input_shape=[env.length], activation='linear'),
        tf.keras.layers.Dense(units=4)
    ])
    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(0.1)
    )
    agent = Agent(env=env, model=model, step_cost=0.01, episode_length=100, memory_capacity=500)
    agent.print_policy()
    num_episodes = 50
    for i in range(num_episodes):
        agent.run_episode()
        if i % 10 == 0:
            print(f"[Episode: {i}] rewards: {agent.rewards[-1]:.4f}; losses: {agent.losses[-1]:.4f}")

    agent.print_policy()

    plt.plot(avg(agent.rewards, num_episodes // 10))
    plt.grid(True)
    plt.show()
