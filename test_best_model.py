import flappy_bird_gymnasium
import gymnasium as gym
import torch
import cv2
import numpy as np
from model import DQN
from main import preprocess_frame  # Reusing the preprocess_frame function


def play_flappy_bird():
    # Set up the Flappy Bird environment
    game_env = gym.make('FlappyBird-v0', render_mode='rgb_array')

    # Prepare the model parameters
    input_shape = (64, 64)
    num_actions = 2
    compute_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the pre-trained neural network model
    ai_agent = DQN(input_shape, num_actions).to(compute_device)
    ai_agent.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=compute_device, weights_only=True))

    ai_agent.eval()  # Set the model to evaluation mode

    while True:
        obs, metadata = game_env.reset()
        cumulative_reward = 0

        while True:
            # Render and preprocess the current frame
            raw_frame = game_env.render()
            processed_frame = preprocess_frame(raw_frame)

            # Visualize the game and preprocessed frame
            cv2.imshow('Game Window', cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR))
            upscaled_frame = (processed_frame * 255).astype(np.uint8)
            upscaled_frame = cv2.resize(upscaled_frame, (256, 256), interpolation=cv2.INTER_NEAREST)
            cv2.imshow('Preprocessed Frame', upscaled_frame)

            # Predict the action using the AI model
            with torch.no_grad():
                input_tensor = torch.FloatTensor(processed_frame).unsqueeze(0).unsqueeze(0).to(compute_device)
                predicted_q_values = ai_agent(input_tensor)
                chosen_action = predicted_q_values.argmax().item()

            # Perform the chosen action in the environment
            obs, reward, done, truncated, metadata = game_env.step(chosen_action)

            # Adjust rewards based on game events
            if done:
                reward = -1.0
            elif metadata.get('score', 0) > 0:
                reward = 1.0
            else:
                reward = 0.1

            cumulative_reward += reward

            # Exit loop if the 'q' key is pressed
            if cv2.waitKey(20) & 0xFF == ord('q'):
                game_env.close()
                cv2.destroyAllWindows()
                return

            if done or truncated:
                print(f"Final Score: {cumulative_reward}")
                break


if __name__ == "__main__":
    play_flappy_bird()
