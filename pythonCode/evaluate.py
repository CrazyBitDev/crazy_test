from env.robotic_navigation import RoboticNavigation
from alg.DDQN import DDQN
import time, sys, argparse
import tensorflow as tf
import config
import numpy as np
import traceback

from stable_baselines3 import PPO

# TF Setup (Keep existing setup)
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

def evaluate(env, args, model_path, n_eval_episodes=20):
    """
    Evaluates the loaded model and tracks success/failure rates.
    """
    print(f"Loading model from: {model_path}")
    
    # Load the trained agent
    # Ensure you are loading the zip file (e.g., "BoatSimulator_PPO_static.zip")
    model = PPO.load(model_path, env=env)

    # Metrics
    success_count = 0
    failure_count = 0
    timeout_count = 0
    total_rewards = []
    
    try:
        for episode in range(1, n_eval_episodes + 1):
            obs = env.reset()
            done = False
            episode_reward = 0
            step_count = 0
            
            print(f"--- Starting Episode {episode} ---")

            while not done:
                # Use deterministic=True for evaluation to get the best action
                action, _states = model.predict(obs, deterministic=True)
                
                # Execute action
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                step_count += 1
                
                # OPTIONAL: Render if supported
                # env.render() 

                if done:
                    # ANALYZE THE INFO DICT
                    # Note: You must verify the specific keys your Unity wrapper returns.
                    # Common keys in these wrappers are 'is_success', 'success', 'crash', etc.
                    
                    is_success = info.get('is_success', False) or info.get('success', False)
                    # Assuming failure is collision or negative condition
                    is_crash = info.get('crash', False) or info.get('collision', False) 
                    
                    outcome = "UNKNOWN"

                    if is_success:
                        success_count += 1
                        outcome = "SUCCESS"
                    elif is_crash:
                        failure_count += 1
                        outcome = "FAILURE (Crash)"
                    else:
                        # If done but neither success nor crash, likely a timeout
                        timeout_count += 1
                        outcome = "TIMEOUT / OTHER"

                    total_rewards.append(episode_reward)
                    
                    print(f"Episode {episode} finished.")
                    print(f"  Outcome: {outcome}")
                    print(f"  Steps: {step_count}")
                    print(f"  Reward: {episode_reward:.4f}")
                    print(f"  Info: {info}") # Print info to help debug keys

    except Exception as e:
        print("An error occurred during evaluation:")
        print(e)
        print(traceback.format_exc())

    finally:
        env.close()
        
        # Final Statistics
        if n_eval_episodes > 0:
            avg_reward = np.mean(total_rewards) if total_rewards else 0.0
            print("\n" + "="*30)
            print("EVALUATION RESULTS")
            print("="*30)
            print(f"Episodes: {n_eval_episodes}")
            print(f"Success Rate: {success_count}/{n_eval_episodes} ({(success_count/n_eval_episodes)*100:.2f}%)")
            print(f"Failure Rate: {failure_count}/{n_eval_episodes}")
            print(f"Timeout Rate: {timeout_count}/{n_eval_episodes}")
            print(f"Average Reward: {avg_reward:.4f}")
            print("="*30)

def generate_environment(editor_build, env_type, render=False):
    worker_id = int(round(time.time() % 1, 4)*10000)
    # Ensure env_type is set correctly for inference if your wrapper supports it
    return RoboticNavigation(editor_build=editor_build, worker_id=worker_id, env_type=env_type, render=render)

if __name__ == "__main__":
    # Default parameters
    args = config.parse_args()
    
    # PARAMETERS FOR EVALUATION
    editor_build = False
    env_type = "training" # Or "inference" if your Unity build distinguishes the two
    render = True         # Usually True for evaluation to see what's happening
    
    # PATH TO YOUR SAVED MODEL
    # Update this to point to the specific .zip file generated during training
    model_to_load = "./models/BoatSimulator_PPO_static.zip" 

    print("Mobile Robotics - PPO Evaluation Mode \n")
    
    env = generate_environment(editor_build, env_type, render=render)
    
    evaluate(env, args, model_path=model_to_load, n_eval_episodes=100)