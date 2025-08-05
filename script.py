import numpy as np

# Define states and actions
states = ['S1', 'S2', 'S3']
actions = ['A1', 'A2']

# Define a transition model with probabilities
transition_model = {
    'S1': {'A1': {'S1': 0.7, 'S2': 0.3}, 'A2': {'S1': 0.4, 'S3': 0.6}},
    'S2': {'A1': {'S1': 0.5, 'S2': 0.5}, 'A2': {'S2': 0.8, 'S3': 0.2}},
    'S3': {'A1': {'S2': 0.6, 'S3': 0.4}, 'A2': {'S1': 0.3, 'S3': 0.7}},
}

# Initialize transition counts
transition_counts = {state: {action: {s: 0 for s in states} for action in actions} for state in states}

def sample_next_state(current_state, action):
    """Sample the next state based on the current state and action using the transition model."""
    next_states = list(transition_model[current_state][action].keys())
    probabilities = list(transition_model[current_state][action].values())
    return np.random.choice(next_states, p=probabilities)

def simulate_mdp(start_state, num_steps):
    """Simulate the MDP for a given number of steps starting from a specified state."""
    current_state = start_state
    for _ in range(num_steps):
        action = np.random.choice(actions)  # Randomly choose an action
        next_state = sample_next_state(current_state, action)  # Sample next state based on transition model
        transition_counts[current_state][action][next_state] += 1  # Update transition counts
        current_state = next_state  # Move to the next state

def estimate_transition_probabilities():
    """Estimate transition probabilities based on the collected transition counts."""
    transition_probabilities = {state: {action: {} for action in actions} for state in states}

    for state in states:
        for action in actions:
            total_transitions = sum(transition_counts[state][action].values())
            for next_state in states:
                if total_transitions > 0:
                    transition_probabilities[state][action][next_state] = transition_counts[state][action][next_state] / total_transitions
                else:
                    transition_probabilities[state][action][next_state] = 0  # No transitions observed

    return transition_probabilities

# Simulate the MDP for 1000 steps starting from 'S1'
simulate_mdp('S1', 1000)

# Estimate the transition probabilities
transition_probabilities = estimate_transition_probabilities()
print("Estimated Transition Probabilities:")
print(transition_probabilities)

# pretty print
def pretty_print_transition_probabilities(probabilities):
    """Pretty print the transition probabilities."""
    for state, actions in probabilities.items():
        print(f"State: {state}")
        for action, next_states in actions.items():
            print(f"  Action: {action}")
            for next_state, prob in next_states.items():
                print(f"    -> {next_state}: {prob:.2f}")
        print()


pretty_print_transition_probabilities(transition_probabilities);
