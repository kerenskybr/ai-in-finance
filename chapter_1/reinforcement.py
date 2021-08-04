import numpy as np

# Coin toss

# The state space (1 = heads, 0 = tails)
ssp = [1, 1, 1, 1, 0]
asp = [1, 0]

def epoch():
    tr = 0
    for _ in range(100):
        # An action is randomly chosen from the action space
        a = np.random.choice(asp)
        # A state is randomly chosen from the state space
        s = np.random.choice(ssp)
        if a == s:
            # The total reward tr is increased by one if the bet is correct
            tr += 1
    return tr

# The game is played for a number of epochs; each epoch is 100 bets
rl = np.array([epoch() for _ in range(15)])

# The average total reward of the epochs played is calculated
print(rl.mean())

# Reinforcement learning tries to learn from what is observed after an action is taken,
# usually based on a reward

def epoch():
    tr = 0
    # Resets the action space before starting (over)
    asp = [0, 1]
    for _ in range(100):
        a = np.random.choice(asp)
        s = np.random.choice(ssp)
        if a == s:
            tr += 1
        # Adds the observed state to the action space
        asp.append(s)
    return tr

rl = np.array([epoch() for _ in range(15)])

print(rl.mean())