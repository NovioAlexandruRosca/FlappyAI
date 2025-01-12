import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np
from model import DQN
from main import preprocess_frame
import torch
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"


def test():
    game_env = gym.make("FlappyBird-v0", render_mode="rgb_array")

    ai_agent = DQN((64, 64), 2).to(device)
    ai_agent.load_state_dict(
        torch.load(
            "models/best_model.pth", map_location=device, weights_only=True
        )
    )

    ai_agent.eval()

    while True:
        obs, metadata = game_env.reset()
        reward = 0

        while True:
            raw_frame = game_env.render()
            processed_frame = preprocess_frame(raw_frame)

            cv2.imshow("Flappyu Birdu", cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR))
            upscaled_frame = (processed_frame * 255).astype(np.uint8)
            upscaled_frame = cv2.resize(
                upscaled_frame, (256, 256), interpolation=cv2.INTER_NEAREST
            )
            cv2.imshow("Preprocessed Frame", upscaled_frame)

            with torch.no_grad():
                input_tensor = (
                    torch.FloatTensor(processed_frame)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(device)
                )
                predicted_q_values = ai_agent(input_tensor)
                chosen_action = predicted_q_values.argmax().item()

            obs, reward, done, truncated, metadata = game_env.step(chosen_action)

            if done:
                reward = -1.0
            elif metadata.get("score", 0) > 0:
                reward = 1.0
            else:
                reward = 0.1

            reward += reward

            if cv2.waitKey(20) & 0xFF == ord("q"):
                game_env.close()
                cv2.destroyAllWindows()
                return

            if done or truncated:
                print(f"Final Score: {reward}")
                break


if __name__ == "__main__":
    test()
