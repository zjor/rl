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


# TODO: move step logic to the environment
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
