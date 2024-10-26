import numpy as np
import pygame
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import random
from FlappyBirdEnv import FlappyBirdEnv

def compute_action(state, weights):
    action_value = np.dot(weights, state)
    return [1, action_value] if action_value > 0 else [0, action_value]

REWARD_LIMIT = 1000
def simulate_episode(env, weights):
    state, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = compute_action(state, weights)[0]
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        if total_reward >= REWARD_LIMIT:
            break
    return total_reward

def simulated_annealing(env, initial_weights):
    initial_temperature = 100000
    cooling_rate = 0.995
    min_temperature = 1.0
    max_restarts = 10
    
    best_overall_reward = -float('inf')
    best_overall_weights = initial_weights.copy()
    
    for restart in range(max_restarts):
        temperature = initial_temperature
        current_weights = 2 * np.random.rand(5) - 1.0
        best_weights = current_weights.copy()
        best_reward = -float('inf')
        noise_scale = 0.1
        count = 0
        
        while temperature > min_temperature:
            count += 1
            current_reward = simulate_episode(env, current_weights)
            if current_reward == REWARD_LIMIT:
                break
            new_weights = current_weights + np.random.normal(0, noise_scale, size=current_weights.shape)
            new_reward = simulate_episode(env, new_weights)
            
            if new_reward > current_reward:
                current_weights = new_weights
                noise_scale = max(noise_scale / 2, 0.01)
            else:
                p = np.exp((new_reward - current_reward) / temperature)
                if np.random.rand() < p:
                    current_weights = new_weights
                noise_scale = min(noise_scale * 2, 1000.0)
            
            if new_reward > best_reward:
                print(f'({restart+1}/{max_restarts}) Best Reward: {new_reward}')
                best_reward = new_reward
                best_weights = new_weights

            temperature *= cooling_rate

        if best_reward > best_overall_reward:
            best_overall_reward = best_reward
            best_overall_weights = best_weights.copy()

        print(f"Restart {restart+1}/{max_restarts}, Best Reward so far: {best_overall_reward}")

    return best_overall_weights, best_overall_reward

# Initialize the environment and weights
env = FlappyBirdEnv()
initial_weights = 2 * np.random.rand(5) - 1.0

# Run simulated annealing
best_weights, best_reward = simulated_annealing(env, initial_weights)
print(f"Best Weights: {best_weights}, Best Reward: {best_reward}")

# Run one rendered simulation with the best weights
env = FlappyBirdEnv()

state, _ = env.reset()
done = False
total_reward = 0

import os

while not done:
    env.render()  # Render the environment

    computed_action = compute_action(state, best_weights)  # Use the best weights
    action = computed_action[0]
    action_value = computed_action[1]
    state, reward, done, _, _ = env.step(action)
    total_reward += reward
    os.system('clear')
    print(f'state = {state} | | Weights = {best_weights}')
    print(f'state = [bird y, bird y velocity, dist to next pipe, next lower pipe height difference, next higher pipe height difference ]')
    print(f'dot( state * weights ) = {action_value:.2f}   ({action})')

print(f"Best episode finished with total reward: {total_reward}")

# Close the environment when done
env.close()