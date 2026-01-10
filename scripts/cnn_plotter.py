import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="training_log.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    fig, axs = plt.subplots(5, 1, figsize=(12, 6), sharex=True)

    axs[0].plot(df["episode"], df["actor_loss"], label="Actor Loss")
    axs[0].set_ylabel("Actor Loss")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(df["episode"], df["critic_loss"], label="Critic Loss", color="orange")
    axs[1].set_ylabel("Critic Loss")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(df["episode"], df["entropy"], label="Entropy", color="green")
    axs[2].set_ylabel("Entropy")
    axs[2].set_xlabel("Episode")
    axs[2].legend()
    axs[2].grid(True)

    axs[3].plot(df["episode"], df["reward_ratio"], label="Reward Ratio", color="purple")
    axs[3].set_ylabel("Reward Ratio")
    axs[3].set_xlabel("Episode")
    axs[3].legend()
    axs[3].grid(True)

    axs[4].plot(df["episode"], df["regret"], label="Regret", color="purple")
    axs[4].set_ylabel("Regret")
    axs[4].set_xlabel("Episode")
    axs[4].legend()
    axs[4].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

