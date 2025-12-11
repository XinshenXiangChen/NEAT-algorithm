# NEAT-algorithm
Implementation of the NEAT algorithm

Video that explains the concepts of the NEAT algorithm
https://youtu.be/yVtdp1kF0I4?si=gYBFmYmKzPnFTyjC

Structure of the evaluate_genotype to call the NEAT function
def evaluate_genotype(genotype):
    done = False
    total_reward = 0
    reset_game()
    while not done:
        inputs = get_input_values()          # from the game state
        outputs = genotype.forward(inputs)   # run the network
        action = pick_action(outputs)        # e.g., argmax or sample
        done, reward = step_game(action)     # advance the game
        total_reward += reward
    return total_reward