import matplotlib.pyplot as plt
import cv2
import numpy as np


def plot_training_stats(episode_rewards, epsilon_values, save=True):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 2)
    plt.plot(range(len(episode_rewards)), episode_rewards)
    plt.title(f"Rewards")
    plt.xlabel("E")
    plt.ylabel("Reward")
    plt.grid(True)

    plt.subplot(1, 2, 1)
    plt.plot(range(0, len(epsilon_values) * 10, 10), epsilon_values, "r-")
    plt.title("E Decay")
    plt.xlabel("Ep")
    plt.ylabel("E")
    plt.grid(True)

    plt.tight_layout()
    if save:
        plt.savefig("training_stats.png")
    plt.close()


def preprocess_frame(frame):
    frame = frame[:400, :]

    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    masks = [
        cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255])),
        cv2.inRange(hsv, np.array([50, 100, 100]), np.array([75, 255, 255])),
        cv2.inRange(hsv, np.array([40, 0, 130]), np.array([100, 120, 255])),
        cv2.inRange(hsv, np.array([20, 50, 50]), np.array([30, 150, 150])),
    ]

    background_mask = masks[0]
    for mask in masks[1:]:
        background_mask = cv2.bitwise_or(background_mask, mask)
    objects_mask = cv2.bitwise_not(background_mask)
    result = np.zeros_like(frame[:, :, 0])
    result[objects_mask > 0] = 255
    resized = cv2.resize(result, (64, 64), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized