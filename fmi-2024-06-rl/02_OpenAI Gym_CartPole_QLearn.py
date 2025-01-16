import gymnasium as gym
import numpy as np
import warnings
import math

if __name__=='__main__':
    # Import library for visualization
    import matplotlib.pyplot as plt

    # Load the environment with render mode specified
    env = gym.make('CartPole-v1', render_mode="human")

    # Initialize the environment to get the initial state
    state = env.reset()

    # Print the state space and action space
    print("State space:", env.observation_space)
    print("Action space:", env.action_space)

    ## Q-Learning

    # Set hyperparameters
    alpha = 0.7  # Learning rate
    alpha_decay = 0.999  # Learning rate decay
    alpha_min = 0.1
    gamma = 0.95 # Discount factor
    epsilon = 0.3  # Exploration rate
    epsilon_decay = 0.995
    epsilon_min = 0.01
    num_episodes = 1000

    # Discretization parameters
    num_buckets = (5, 3, 6, 12)  # Number of discrete buckets for each state parameter
    state_bounds = list(zip(env.observation_space.low * 0.6, env.observation_space.high * 0.6))

    # Adjust the state bounds for CartPole to avoid infinite ranges
    state_bounds[1] = [-0.5, 0.5]
    state_bounds[3] = [-np.radians(50), np.radians(50)]

    # Calculate the width of each bucket
    bucket_width = [(state_bounds[i][1] - state_bounds[i][0]) / num_buckets[i] for i in range(len(state_bounds))]

    # Initialize the Q-table
    q_table = np.zeros(num_buckets + (env.action_space.n,))

    def discretize_state(state):
        discrete_state = []
        for i in range(len(state)):
            if state[i] <= state_bounds[i][0]:
                bucket_index = 0
            elif state[i] >= state_bounds[i][1]:
                bucket_index = num_buckets[i] - 1
            else:
                bucket_index = int((state[i] - state_bounds[i][0]) / bucket_width[i])
            discrete_state.append(bucket_index)
        return tuple(discrete_state)


    # Q-Learning algorithm
    for episode in range(num_episodes):
        state, info = env.reset()
        state = discretize_state(state)
        done = False
        total_reward = 0

        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit

            # Take a step in the environment
            step_result = env.step(action)

            # Check the number of values returned and unpack accordingly
            if len(step_result) == 4:
                next_state, reward, done, info = step_result
                terminated = False
            else:
                next_state, reward, done, truncated, info = step_result
                terminated = done or truncated

            cart_pos, cart_v, pole_ang, pole_ang_vel = next_state
            reward -= math.fabs(cart_pos) * 4 / env.observation_space.high[0] + 4 * math.fabs(pole_ang) # penalize cart moving off centre & pole off center
            print(f"{reward:.2f}", end="")

            next_state = discretize_state(next_state)

            # Update Q-value
            q_value = q_table[state][action]
            print(f"|q{q_value:.2f}, ", end="")
            max_q_value_next = np.max(q_table[next_state])
            new_q_value = q_value + alpha * (reward + gamma * max_q_value_next - q_value)
            q_table[state][action] = new_q_value

            state = next_state
            total_reward += reward

            if terminated:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        alpha = max(alpha_min, alpha * alpha_decay)
        print(f"\nEpisode {episode + 1}: Total Reward: {total_reward:.4f} ==> alpha {alpha:.2f}, epsilon {epsilon:.2f}")

        # Render the environment
        env.render()

    env.close()  # Close the environment when done


