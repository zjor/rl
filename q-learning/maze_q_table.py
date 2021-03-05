import numpy as np
from maze import MAPS, Environment


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
        self.Q = np.zeros((env.width * env.height, len(Agent.actions)))
        self.s0 = env.field.index('s')
        self.episode_length = episode_length
        self.rewards = []

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
                ix = np.argmax(self.Q[s, :])
                print(Agent.actions[ix], end='')
            print()

    def run_episode(self):
        s = self.s0
        episode_number = len(self.rewards)
        self.rewards.append(.0)
        for j in range(self.episode_length):
            a = np.argmax(self.Q[s, :] + np.random.randn(1, len(Agent.actions)) * (1 / (episode_number + 1)))
            s1, r, over = self.step(s, Agent.actions[a])
            if s != s1:
                r -= self.step_cost
            r -= self.living_cost
            self.Q[s, a] = self.Q[s, a] + self.lr * (r + self.y * np.max(self.Q[s1, :]) - self.Q[s, a])
            s = s1
            self.rewards[-1] += r
            if over:
                break


np.random.seed(42)
env = Environment(world=MAPS["classic"])
agent = Agent(env=env, p=0.9, step_cost=.01)

num_episodes = 500
for i in range(num_episodes):
    agent.run_episode()

env.print_game()
print()
agent.print_policy()

print(agent.rewards)
