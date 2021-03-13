"""
DQL Experience replay
"""

import numpy as np
import torch
from dqn import Model, ReplayMemory
from maze import MAPS, Environment, AbstractAgent
from torch import nn


def avg(a, n):
    return np.convolve(a, np.ones(n) / n, mode='valid')[:-1]


class Agent(AbstractAgent):
    actions = ['←', '→', '↑', '↓']

    def __init__(self, env, p=1.0, lr=0.8, y=0.95, step_cost=.0, living_cost=.0, episode_length=100,
                 memory_capacity=100, batch_size=10, eps=0.5, eps_decay=0.999):
        AbstractAgent.__init__(self, eps, eps_decay)
        self.env = env
        self.lr = lr
        self.y = y
        self.step_cost = step_cost
        self.living_cost = living_cost
        q = (1.0 - p) / 2
        self.stochastic_actions = {
            '←': [[0, 2, 3], [p, q, q]],
            '→': [[1, 2, 3], [p, q, q]],
            '↑': [[2, 0, 1], [p, q, q]],
            '↓': [[3, 0, 1], [p, q, q]]
        }
        self.s0 = env.field.index('s')
        self.episode_length = episode_length
        self.rewards = []
        self.losses = []
        self.state_len = env.width * env.height
        self.nn = Model(
            in_features=self.state_len,
            hidden=[],
            out_features=len(Agent.actions))
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=0.05)
        self.memory = ReplayMemory(memory_capacity)
        self.batch_size = batch_size

    def step(self, state, action):
        # simulating Markov Process, desired action happens with probability p
        # but with the probability (1-p) / 2 the agent goes sideways
        sa = self.stochastic_actions[action]
        mp_action = np.random.choice(sa[0], p=sa[1])
        action = Agent.actions[mp_action]
        return self.env.step(state, action)

    def print_policy(self):
        for y in range(self.env.height):
            for x in range(self.env.width):
                s = y * self.env.width + x
                cell = self.env.field[s]
                if not (cell == '.' or cell == 's'):
                    print(cell, end='')
                    continue
                q_predicted = self._predict_q(s)
                a = torch.argmax(q_predicted, 0).item()
                print(Agent.actions[a], end='')
            print()

    def _encode_state(self, s):
        z = np.zeros(self.state_len)
        z[s] = 1
        return torch.tensor(z, dtype=torch.float)

    def _predict_q(self, s):
        return self.nn.forward(self._encode_state(s))

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        for s, a, s1, r in transitions:
            q_predicted = self._predict_q(s)
            q_target = q_predicted.clone().detach()
            q_target[a] = r + self.y * self._predict_q(s1).max().item()

            loss = self.criterion(q_predicted, q_target)
            self.losses.append(loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def run_episode(self):
        AbstractAgent.run_episode(self)
        s = self.s0
        self.rewards.append(.0)
        for j in range(self.episode_length):
            q_predicted = self._predict_q(s)
            a = torch.argmax(q_predicted, 0).item()
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

    torch.manual_seed(42)
    np.random.seed(42)
    env = Environment(world=MAPS["classic"], win_reward=5.0, death_reward=-5.0)
    agent = Agent(env=env, p=1.0, step_cost=0.01, episode_length=100, memory_capacity=500)
    agent.print_policy()
    num_episodes = 5000
    for i in range(num_episodes):
        agent.run_episode()
        if i % 100 == 0:
            print(f"Episode: {i}")
            print(agent.rewards[-1])
            print(agent.losses[-1].detach().numpy())

    agent.print_policy()

    plt.plot(avg(agent.rewards, num_episodes // 10))
    plt.grid(True)
    plt.show()
