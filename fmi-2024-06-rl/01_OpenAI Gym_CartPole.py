import gymnasium as gym
import numpy as np
import warnings

if __name__=='__main__':
    # Suppress specific deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Load the environment with render mode specified
    env = gym.make('CartPole-v1', render_mode="human")

    # Initialize the environment to get the initial state
    state = env.reset()

    # Print the state space and action space
    print("State space:", env.observation_space)
    print("Action space:", env.action_space)

    # Run a few steps in the environment with random actions
    for _ in range(200):
        env.render()  # Render the environment for visualization
        action = env.action_space.sample()  # Take a random action

        # Take a step in the environment
        step_result = env.step(action)

        # Check the number of values returned and unpack accordingly
        if len(step_result) == 4:
            next_state, reward, done, info = step_result
            terminated = False
        else:
            next_state, reward, done, truncated, info = step_result
            terminated = done or truncated

        print(f"Action: {action}, Reward: {reward}, Next State: {next_state}, Done: {done}, Info: {info}")

        if terminated:
            state = env.reset()  # Reset the environment if the episode is finished
            print("\nState space:", env.observation_space)

    env.close()  # Close the environment when done