# see: https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py

import numpy as np

MAPS = {
    "4x4": {
        "width": 4,
        "field": [
            "s...",
            ".x.x",
            "...x",
            "x..f"
        ]
    },
    "8x8": {
        "width": 8,
        "field": [
            "s.......",
            "........",
            "...x....",
            ".....x..",
            "...x....",
            ".xx...x.",
            ".x..x.x.",
            "...x...f"
        ]},
    "classic": {
        "width": 4,
        "field": [
            "...f",
            ".o.x",
            "s..."
        ]}
}


class Environment:
    def __init__(self, world, win_reward=1.0, death_reward=-1.0):
        self.field = ''.join(world["field"])
        self.height = len(world["field"])
        self.width = world["width"]
        self.win_reward = win_reward
        self.death_reward = death_reward

    def print_game(self):
        for y in range(self.height):
            for x in range(self.width):
                print(self.field[y * self.width + x], end='')
            print()

    def step(self, state, action):
        w, h = self.width, self.height

        x, y = state % w, state // w
        if action == '→':
            x += 1
        elif action == '←':
            x -= 1
        elif action == '↑':
            y -= 1
        else:
            y += 1
        if (0 <= x < w) and (0 <= y < h):
            ix = y * w + x
            if self.field[ix] == 'f':
                return ix, self.win_reward, True
            elif self.field[ix] == 'x':
                return ix, self.death_reward, True,
            elif self.field[ix] == 'o':
                return state, 0, False
            else:
                return ix, 0, False
        else:
            return state, 0, False


class Agent:
    actions = ['←', '→', '↑', '↓']

    def __init__(self, env, p=1.0, lr=0.8, y=0.95, step_cost=.0, living_cost=.0):
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
        for j in range(100):
            a = np.argmax(self.Q[s, :] + np.random.randn(1, len(Agent.actions)) * (1 / (i + 1)))
            s1, r, over = self.step(s, Agent.actions[a])
            if s != s1:
                r -= self.step_cost
            r -= self.living_cost
            self.Q[s, a] = self.Q[s, a] + self.lr * (r + self.y * np.max(self.Q[s1, :]) - self.Q[s, a])
            s = s1
            if over:
                break


np.random.seed(42)
env = Environment(world=MAPS["classic"])
agent = Agent(env=env, p=1.0, step_cost=.0)

num_episodes = 500
for i in range(num_episodes):
    agent.run_episode()

env.print_game()
print()
agent.print_policy()
