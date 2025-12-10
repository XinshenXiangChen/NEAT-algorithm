board_size = 10
board = []
for i in range(board_size):
    board.append([])
    for j in range(board_size):
        board[i].append(".")

initial_pos = (2, 4)
directions = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}


class Snake:
    def __init__(self):
        self.direction = "right"
        self.positions = [initial_pos]



    def step(self):
        self.update_position()

    def update_position(self):
        positions_size = len(self.positions)


        for i in range(positions_size - 1):
            self.positions[positions_size - i - 1] = self.positions[positions_size - i - 2]

        self.positions[0] = (
            self.positions[0][0] + directions.get(self.direction)[0],
            self.positions[0][1] + directions.get(self.direction)[1]
        )

    def check_collisions(self):
        if

