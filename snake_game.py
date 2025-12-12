import random

board_size = 10
initial_pos = (2, 4)
directions = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
direction_list = ["up", "down", "left", "right"]


class SnakeGame:
    def __init__(self):
        self.direction = "right"
        self.positions = [initial_pos]

        self.apple_pos = (5, 5)
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.max_steps = 200  # Prevent infinite games
        self.died_by_collision = False

    def reset(self):
        """Reset the game to initial state."""
        self.direction = "right"
        self.positions = [initial_pos]
        self.apple_pos = self._spawn_apple()
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.died_by_collision = False

    def _spawn_apple(self):
        """Spawn a new apple at a random position not occupied by the snake."""
        while True:
            x = random.randint(0, board_size - 1)
            y = random.randint(0, board_size - 1)
            pos = (x, y)
            if pos not in self.positions:
                return pos

    def step(self):
        """Move the snake one step forward."""
        if self.game_over:
            return
        
        self.steps += 1
        last_pos = self.update_position()
        
        # Check if apple is eaten
        if self.positions[0] == self.apple_pos:
            self.score += 1
            self.apple_pos = self._spawn_apple()
            self.positions.append(last_pos)
            # Don't remove tail when eating apple (snake grows)
        else:
            pass
        
        # Check collisions
        self.check_collisions()

    def update_position(self):
        """Update snake position based on current direction."""
        positions_size = len(self.positions)
        last_pos = self.positions[positions_size-1]
        # Move body segments
        for i in range(positions_size - 1):
            self.positions[positions_size - i - 1] = self.positions[positions_size - i - 2]

        # Move head
        dx, dy = directions.get(self.direction, (0, 0))
        self.positions[0] = (
            self.positions[0][0] + dx,
            self.positions[0][1] + dy
        )

        return last_pos

    def check_collisions(self):
        """Check for wall collisions and self collisions."""
        head_x, head_y = self.positions[0]
        
        # Check wall collisions
        if head_x < 0 or head_x >= board_size or head_y < 0 or head_y >= board_size:
            self.game_over = True
            self.died_by_collision = True
            return
        
        # Check self collision
        if self.positions[0] in self.positions[1:]:
            self.game_over = True
            self.died_by_collision = True
            return
        
        # Check max steps
        if self.steps >= self.max_steps:
            self.game_over = True
            # Not a collision, so died_by_collision stays False

    def set_direction(self, new_direction):
        """Set the direction, but prevent reversing into itself."""
        opposite = {
            "up": "down",
            "down": "up",
            "left": "right",
            "right": "left"
        }
        if new_direction in directions and new_direction != opposite.get(self.direction):
            self.direction = new_direction

    def get_state(self):
        """Get game state as input for neural network.
        Returns: [distance_to_top, distance_to_right, distance_to_bottom, distance_to_left, distance_to_apple]
        Matches config input_nodes = 5
        """
        head_x, head_y = self.positions[0]
        apple_x, apple_y = self.apple_pos

        distance_to_top = head_y / board_size
        distance_to_right = (board_size - 1 - head_x) / board_size
        distance_to_bottom = (board_size - 1 - head_y) / board_size
        distance_to_left = head_x / board_size


        dx = apple_x - head_x
        dy = apple_y - head_y
        max_distance = (board_size - 1) * (2 ** 0.5)
        distance_to_apple = ((dx ** 2 + dy ** 2) ** 0.5) / max_distance
        
        return [distance_to_top, distance_to_right, distance_to_bottom, distance_to_left, distance_to_apple]

    def get_fitness(self):
        """Calculate fitness score based on score and steps."""
        fitness = self.score * 2000 + self.steps
        
        # Penalty for dying by collision
        if self.died_by_collision:
            fitness -= self.score * 100
        
        return fitness

    def play_with_network(self, genotype, max_steps):
        """Play the game using a neural network genotype."""
        self.reset()
        self.max_steps = max_steps
        
        while not self.game_over:
            # Get current state
            state = self.get_state()

            output = genotype.forward(state)

            # Use the output with highest value among first 4 outputs
            if len(output) >= 4:
                direction_outputs = output[:4]
                direction_index = direction_outputs.index(max(direction_outputs))
                self.set_direction(direction_list[direction_index])
            
            # Step the game
            self.step()
        
        return self.get_fitness()

    def print_board(self):
        for i in range(board_size):
            for j in range(board_size):
                if (i,j) in self.positions:
                    if (i, j) == self.positions[0]:
                        print("O", end="")
                    else:
                        print("x", end="")
                elif (i, j) == self.apple_pos:
                    print("A", end="")
                else:
                    print(".", end="")

            print()
        print("===" * 3)

    def replay(self, genotype, max_steps=10000):

        self.reset()
        self.max_steps = max_steps

        while not self.game_over:

            state = self.get_state()


            output = genotype.forward(state)


            if len(output) >= 4:
                direction_outputs = output[:4]
                direction_index = direction_outputs.index(max(direction_outputs))
                self.set_direction(direction_list[direction_index])
                self.print_board()
            # Step the game
            self.step()


if __name__ == "__main__":
    from NEAT.NEAT import NEAT
    

    game = SnakeGame()
    
    def evaluate_genotype(genotype):
        fitness = game.play_with_network(genotype, max_steps=256)
        return fitness
    
    # Initialize NEAT
    neat = NEAT(fn_fitness=evaluate_genotype)

    print("Starting NEAT training for Snake Game...")
    print("=" * 60)
    best_genotype, generation_best_list = neat.evolve(evaluate_genotype)
    
    game.reset()
    game.replay(best_genotype)
    print("=" * 60)
    print("Training completed!")
    print(f"Best overall fitness: {best_genotype.fitness_score:.2f}")
    print(f"Best genotype connections: {len(best_genotype.connections)}")
    print(f"Best genotype hidden layers: {len(best_genotype.hidden_layers)}")