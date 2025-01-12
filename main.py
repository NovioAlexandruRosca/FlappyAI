import flappy_bird_gymnasium
import gymnasium as gym
import torch
from agent import DQNAgent
from datetime import datetime
import numpy as np
import cv2
from processing_and_ploting import preprocess_frame, plot_training_stats


def main():

    log_file = open("models/training_log.txt", "a")
    start_time = datetime.now().strftime("%H:%M:%S")
    log_file.write(f"\n\n{start_time} ===\n")

    env = gym.make("FlappyBird-v0", render_mode="rgb_array")
    observation, info = env.reset()
    agent = DQNAgent((64, 64), 2)
    best_reward = -1000
    episode_rewards = []
    epsilon_values = []

    for episode in range(25000):
        observation, info = env.reset()
        episode_reward = 0

        while True:
            frame = env.render()
            processed_frame = preprocess_frame(frame)

            action = agent.select_action(processed_frame)
            observation, reward, terminated, truncated, metadata = env.step(action)

            if terminated:
                reward = -1.0
            elif metadata.get("score", 0) > 0:
                reward = 1.0
            else:
                reward = 0.1

            next_frame = preprocess_frame(env.render())
            agent.memory.push(processed_frame, action, reward, next_frame, terminated)
            agent.train()

            episode_reward += reward

            if terminated or truncated:
                episode_rewards.append(episode_reward)
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    torch.save(agent.policy_net.state_dict(), "models/best_model.pth")

                if episode % 200 == 0:
                    checkpoint_path = f"models/model_checkpoint_ep{episode}.pth"
                    torch.save(agent.policy_net.state_dict(), checkpoint_path)
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    log_file.write(
                        f"[{timestamp}] Saved history of episode {episode}\n"
                    )
                    log_file.flush()
                break

        if episode % 10 == 0:
            timestamp = datetime.now().strftime("%H:%M:%S")
            stats = f"[{timestamp}] Episode {episode}, Epsilon: {agent.epsilon:.5f}, Avg Reward:\
             {np.mean(episode_rewards[-100:]):.2f}"
            print(stats)
            log_file.write(stats + "\n")
            log_file.flush()
            epsilon_values.append(agent.epsilon)
            plot_training_stats(episode_rewards, epsilon_values)

        if episode_reward >= 5000:
            print(f"Target was achived the training has been stopped")
            torch.save(agent.policy_net.state_dict(), "models/final_model.pth")
            epsilon_values.append(agent.epsilon)
            plot_training_stats(episode_rewards, epsilon_values)
            break

    log_file.close()
    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
