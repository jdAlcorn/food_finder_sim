import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import sys
import signal
import csv
import os
from datetime import datetime

# =========================
# Args
# =========================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num-rays", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--save-path", type=str, default="model.pt")
    return parser.parse_args()

# =========================
# Checkpointing
# =========================

def save_checkpoint(network, optimizer, path):
    torch.save({
        "model": network.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
    }, path)
    print(f"\n✓ Saved checkpoint to {path}")

def load_checkpoint(network, optimizer, path):
    ckpt = torch.load(path, map_location="cpu")
    network.load_state_dict(ckpt["model"])
    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"✓ Loaded checkpoint from {path}")

def register_interrupt_handler(network, optimizer, save_path):
    def handler(sig, frame):
        print("\nInterrupted — saving model...")
        save_checkpoint(network, optimizer, save_path)
        sys.exit(0)
    signal.signal(signal.SIGINT, handler)

class TrainingLogger:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.file_exists = os.path.exists(csv_path)

        self.csv_file = open(csv_path, "a", newline="")
        self.writer = csv.writer(self.csv_file)

        if not self.file_exists:
            self.writer.writerow([
                "episode",
                "episode_length",
                "actor_loss",
                "critic_loss",
                "entropy",
                "reward_ratio",
                "regret"
            ])
            self.csv_file.flush()

    def log(self, ep, ep_len, actor_loss, critic_loss, entropy, reward_ratio, regret):
        # Print (same style as before)
        print(
            f"ep {ep:05d} | "
            f"len={ep_len:03d} | "
            f"actor={actor_loss:+.3f} | "
            f"critic={critic_loss:+.3f} | "
            f"entropy={entropy:.3f} | "
            f"reward_ratio={reward_ratio:.3f} | "
            f"regret={regret:.3f}"
        )

        # Write CSV
        self.writer.writerow([
            ep,
            ep_len,
            actor_loss,
            critic_loss,
            entropy,
            reward_ratio,
            regret
        ])
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()

# =========================
# Network
# =========================

class TestCnnNetwork(nn.Module):
    def __init__(self, num_rays=128, init_seed=42):
        super().__init__()
        torch.manual_seed(init_seed)

        self.vision_encoder = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.Flatten()
        )

        self.actor = nn.Linear(64 * num_rays, 3)   # LEFT / STAY / RIGHT
        self.critic = nn.Linear(64 * num_rays, 1)

    def forward(self, obs):
        feat = self.vision_encoder(obs)
        logits = self.actor(feat)
        value = self.critic(feat).squeeze(-1)
        return logits, value

    def act(self, obs):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, logits

# =========================
# Environment
# =========================

def make_vision(num_rays, food_idx):
    v = torch.zeros(1, 1, num_rays)
    v[0, 0, food_idx % num_rays] = 1.0
    return v

def step_environment(food_idx, action, num_rays):
    # 0 = LEFT, 1 = STAY, 2 = RIGHT
    if action == 0:
        food_idx += 1
    elif action == 2:
        food_idx -= 1
    return food_idx % num_rays

def circular_distance(a, b, n):
    d = abs(a - b)
    return min(d, n - d)

def compute_reward(food_idx, num_rays):
    center = num_rays // 2
    dist = circular_distance(food_idx, center, num_rays)
    if dist == 0:
        return +1.0
    return -dist - 0.1   # step penalty prevents spinning

# =========================
# Training utils
# =========================

def entropy_coef(step, start=0.2, end=0.0, decay_steps=500):
    t = min(step / decay_steps, 1.0)
    return start * (1 - t) + end * t

def compute_returns(rewards, gamma):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns)

def run_episode_and_collect(network, start_idx, num_rays, max_steps=100):
    food_idx = start_idx

    log_probs = []
    values = []
    rewards = []
    logits_list = []

    for _ in range(max_steps):
        obs = make_vision(num_rays, food_idx)
        action, log_prob, value, logits = network.act(obs)

        food_idx = step_environment(food_idx, action.item(), num_rays)
        reward = compute_reward(food_idx, num_rays)

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        logits_list.append(logits)

    return log_probs, values, rewards, logits_list

def train_episode(log_probs, values, rewards, logits, optimizer, gamma, step):
    # --- compute Monte Carlo returns ---
    returns = compute_returns(rewards, gamma)

    # 🔧 CRITICAL FIX: normalize returns
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    log_probs = torch.cat(log_probs)
    values = torch.cat(values)
    logits = torch.cat(logits)

    # --- advantages ---
    advantages = returns - values.detach()

    # --- losses ---
    actor_loss = -(log_probs * advantages).mean()
    critic_loss = (values - returns).pow(2).mean()

    # --- entropy bonus ---
    dist = torch.distributions.Categorical(logits=logits)
    entropy = dist.entropy().mean()

    loss = actor_loss + 0.5 * critic_loss - entropy_coef(step) * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return actor_loss.item(), critic_loss.item(), entropy.item()

# Used for plotting improvement over time
def optimal_episode_return(start_idx, num_rays, max_steps, gamma):
    center = num_rays // 2
    dist = circular_distance(start_idx, center, num_rays)

    # steps to reach center optimally
    steps_to_center = dist

    rewards = []

    # move toward center
    for _ in range(steps_to_center):
        rewards.append(-1.0)  # approx: -dist change per step

    # arrive at center
    rewards.append(+1.0)

    # stay at center
    for _ in range(max_steps - steps_to_center - 1):
        rewards.append(+1.0)

    # discounted return
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
    return G

# Used for plotting improvement over time
def discounted_return(rewards, gamma):
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
    return G

# =========================
# Interactive mode
# =========================

def render_vision(food_idx, num_rays, width=60):
    idx = int(food_idx / num_rays * width)
    bar = ["-"] * width
    bar[idx] = "F"
    return "|" + "".join(bar) + "|"

def action_name(a):
    return ["LEFT", "STAY", "RIGHT"][a]

def run_interactive(network, num_rays):
    while True:
        food_idx = int(input(f"\nStart food index (0–{num_rays-1}): "))
        print("\n--- INTERACTIVE ---")
        for t in range(300):
            obs = make_vision(num_rays, food_idx)
            with torch.no_grad():
                action, _, value, _ = network.act(obs)
            food_idx = step_environment(food_idx, action.item(), num_rays)

            print(
                f"t={t:03d} | "
                f"{action_name(action.item()):5s} | "
                f"value={value.item():+.2f} | "
                f"{render_vision(food_idx, num_rays)}"
            )

# =========================
# Main
# =========================

MAX_STEPS = 100
if __name__ == "__main__":
    args = parse_args()

    net = TestCnnNetwork(args.num_rays)
    optimizer = None if args.interactive else optim.Adam(net.parameters(), lr=3e-4)

    if args.checkpoint:
        load_checkpoint(net, optimizer, args.checkpoint)

    if args.interactive:
        net.eval()
        run_interactive(net, args.num_rays)
        sys.exit(0)

    register_interrupt_handler(net, optimizer, args.save_path)
    logger = TrainingLogger("training_log.csv")

    global_step = 0

    for ep in range(args.episodes):
        start_idx = torch.randint(0, args.num_rays, ()).item()

        lp, v, r, logits = run_episode_and_collect(net, start_idx, args.num_rays, MAX_STEPS)
        a_loss, c_loss, ent = train_episode(lp, v, r, logits, optimizer, args.gamma, global_step)
        optimal_rewards = optimal_episode_return(start_idx, args.num_rays, len(r), args.gamma)
        actual_rewards = discounted_return(r, args.gamma)
        normalized_reward_ratio = actual_rewards/optimal_rewards
        regret = optimal_rewards - actual_rewards
        global_step += 1

        if ep % 10 == 0:
            logger.log(
                ep=ep,
                ep_len=len(r),
                actor_loss=a_loss,
                critic_loss=c_loss,
                entropy=ent,
                reward_ratio=normalized_reward_ratio,
                regret=regret
            )
            save_checkpoint(net, optimizer, args.save_path)

    save_checkpoint(net, optimizer, args.save_path)

