import numpy as np
from maze import MAPS, Environment, AbstractAgent


class Agent(AbstractAgent):
    actions = ['←', '→', '↑', '↓']

    def __init__(self, env, p=1.0, lr=0.8, y=0.95, step_cost=.0, living_cost=.0, episode_length=100, eps=0.5,
                 eps_decay=0.999):
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
                cell = self.env.field[s]
                if not (cell == '.' or cell == 's'):
                    print(cell, end='')
                    continue
                ix = np.argmax(self.Q[s, :])
                print(Agent.actions[ix], end='')
            print()

    def run_episode(self):
        AbstractAgent.run_episode(self)
        s = self.s0
        self.rewards.append(.0)
        for j in range(self.episode_length):
            a = np.argmax(self.Q[s, :])
            a = self.select_action(a)
            s1, r, over = self.step(s, AbstractAgent.actions[a])
            if s != s1:
                r -= self.step_cost
            r -= self.living_cost
            self.Q[s, a] = self.Q[s, a] + self.lr * (r + self.y * np.max(self.Q[s1, :]) - self.Q[s, a])
            s = s1
            self.rewards[-1] += r
            if over:
                break


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.random.seed(42)
    env = Environment(world=MAPS["classic"], win_reward=5.0, death_reward=-5.0)
    agent = Agent(env=env, p=1.0, step_cost=0.01, episode_length=100)

    num_episodes = 1000
    for i in range(num_episodes):
        agent.run_episode()

    env.print_game()
    print()
    agent.print_policy()

    plt.plot(agent.rewards)
    plt.show()
