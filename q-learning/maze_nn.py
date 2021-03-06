import torch
import numpy as np
from torch import nn
from maze import MAPS, Environment


class Model(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        layer_sizes = [in_features] + hidden
        layers = []

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(layer_sizes[-1], out_features))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Agent:
    actions = ['←', '→', '↑', '↓']

    def __init__(self, env, p=1.0, lr=0.8, y=0.95, step_cost=.0, living_cost=.0, episode_length=100):
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

    def _e_greedy_action(self, a, episode):
        eps = (1.0 / (episode + 1))
        if eps < np.random.rand():
            return np.random.choice(range(len(Agent.actions)))
        else:
            return a

    def run_episode(self):
        s = self.s0
        episode_number = len(self.rewards)
        self.rewards.append(.0)
        for j in range(self.episode_length):
            q_predicted = self._predict_q(s)
            a = torch.argmax(q_predicted, 0).item()
            a = self._e_greedy_action(a, episode_number)
            s1, r, over = self.step(s, Agent.actions[a])
            if s != s1:
                r -= self.step_cost
            r -= self.living_cost

            q_target = q_predicted.clone().detach()
            q_target[a] = r + self.y * self._predict_q(s1).max().item()

            loss = self.criterion(q_predicted, q_target)
            self.losses.append(loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            s = s1
            self.rewards[-1] += r
            if over:
                break


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    env = Environment(world=MAPS["classic"], win_reward=5.0, death_reward=-10.0)
    agent = Agent(env=env, p=1.0, step_cost=0.2, episode_length=100)
    agent.print_policy()
    for i in range(10000):
        agent.run_episode()
        if i % 100 == 0:
            print(agent.rewards[-1])
            print(agent.losses[-1].detach().numpy())

    agent.print_policy()
