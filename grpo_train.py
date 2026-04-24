"""
GRPO Training Script — Single-Turn Per-Step (matches inference.py)
==================================================================
Each training example is one (system + observation) prompt. GRPO generates
G completions online, each is parsed into an action, the env is replayed
to that step and stepped once, and the real environment reward is returned.

The model sees exactly what it sees at inference time: system prompt +
current observation. No history. notes_to_self is the only memory.

Pipeline:
    1. generate_grpo_data.py -> grpo_data.jsonl (observation prompts + replay actions)
    2. This script loads the dataset, GRPO generates completions online, scores via env

Usage:
    python grpo_train.py

Env vars:
    SFT_MODEL_DIR    — path to SFT model (default: ./sft_model_merged)
    GRPO_DATA_FILE   — path to GRPO dataset (default: grpo_data.jsonl)
    OUTPUT_DIR       — where to save model (default: ./grpo_model)
"""

import os
import json
import time
import logging

from dotenv import load_dotenv
load_dotenv()

from unsloth import FastLanguageModel

from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

from server.inventory_env import InventoryEnvironment
from models import InventoryAction
from inference import SYSTEM_PROMPT, format_observation, parse_action

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("grpo_train")

# --- Config ---
SFT_MODEL_DIR = os.getenv("SFT_MODEL_DIR", "./sft_model_merged")
GRPO_DATA_FILE = os.getenv("GRPO_DATA_FILE", "grpo_data.jsonl")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./grpo_model")
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "1"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "4"))
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "4096"))
MAX_COMPLETION = int(os.getenv("MAX_COMPLETION", "1024"))
NUM_GENERATIONS = int(os.getenv("NUM_GENERATIONS", "4"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "5e-6"))
LORA_RANK = int(os.getenv("LORA_RANK", "32"))

# Track rewards across training for plotting
_reward_log = []
_component_log = []  # Track reward sub-components
_step_counter = [0]


def replay_to_step(task_name, prior_actions_json):
    """Replay baseline actions to reconstruct environment state at a given step."""
    prior_actions = json.loads(prior_actions_json)
    env = InventoryEnvironment(task_name)
    env.reset()
    for action_dict in prior_actions:
        action = InventoryAction(**action_dict)
        env.step(action)
    return env


def step_reward(completions, task_name, prior_actions, **kwargs):
    """Score each completion by parsing it, replaying env, stepping once.

    Args:
        completions: list of model-generated text (G per prompt)
        task_name: list of task names (dataset column, repeated for G completions)
        prior_actions: list of JSON-encoded action histories (dataset column)

    Returns:
        list of float rewards from the real environment
    """
    rewards = []
    batch_components = []
    for completion, tname, actions_json in zip(completions, task_name, prior_actions):
        try:
            # Replay to get env at this step
            env = replay_to_step(tname, actions_json)

            # Conversational format: completion is [{"role": "assistant", "content": "..."}]
            # Standard format: completion is a string
            if isinstance(completion, list):
                text = completion[0]["content"] if completion else ""
            else:
                text = str(completion)

            action = parse_action(text)

            # Step once and get real reward
            obs = env.step(action)
            rewards.append(obs.reward)
            batch_components.append(getattr(env, "reward_components", {}))
        except Exception as e:
            log.debug(f"Reward computation failed: {e}")
            rewards.append(-1.0)
            batch_components.append({})

    # Log batch stats
    _step_counter[0] += 1
    avg_r = sum(rewards) / len(rewards) if rewards else 0
    min_r = min(rewards) if rewards else 0
    max_r = max(rewards) if rewards else 0
    _reward_log.append({
        "step": _step_counter[0],
        "mean": avg_r,
        "min": min_r,
        "max": max_r,
        "rewards": rewards,
        "tasks": list(task_name),
    })

    # Log reward sub-components
    if batch_components and any(batch_components):
        comp_means = {}
        for key in ["R_directives", "R_planning", "R_revenue", "R_fulfillment", "R_waste",
                     "milestone_bonus", "directive_penalty", "hard_penalty"]:
            vals = [c.get(key, 0.0) for c in batch_components if c]
            comp_means[key] = sum(vals) / len(vals) if vals else 0.0
        comp_means["step"] = _step_counter[0]
        _component_log.append(comp_means)

    if _step_counter[0] % 5 == 0 or _step_counter[0] <= 3:
        log.info(
            f"[GRPO step {_step_counter[0]}] "
            f"reward: mean={avg_r:.3f} min={min_r:.3f} max={max_r:.3f} "
            f"(G={len(rewards)})"
        )

    return rewards


def load_grpo_dataset(filepath):
    """Load GRPO dataset from JSONL. Each row becomes a prompt for online generation."""
    examples = []
    task_counts = {}
    with open(filepath) as f:
        for line in f:
            row = json.loads(line.strip())
            examples.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["observation"]},
                ],
                "task_name": row["task_name"],
                "prior_actions": row["prior_actions"],
            })
            task_counts[row["task_name"]] = task_counts.get(row["task_name"], 0) + 1

    log.info(f"Loaded {len(examples)} prompts from {filepath}")
    for task, count in task_counts.items():
        log.info(f"  {task}: {count} prompts")
    return Dataset.from_list(examples)


def _smooth(values, window):
    """Rolling average smoothing."""
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window)
        smoothed.append(sum(values[start:i+1]) / (i - start + 1))
    return smoothed


def plot_grpo_curves(output_dir, trainer):
    """Plot comprehensive GRPO training curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)

        if not _reward_log:
            log.warning("No reward data to plot")
            return

        steps = [r["step"] for r in _reward_log]
        means = [r["mean"] for r in _reward_log]
        mins = [r["min"] for r in _reward_log]
        maxs = [r["max"] for r in _reward_log]
        window = max(5, len(means) // 15) if len(means) > 10 else 1

        # Extract trainer log history
        history = trainer.state.log_history if trainer else []
        loss_steps = [h["step"] for h in history if "loss" in h]
        losses = [h["loss"] for h in history if "loss" in h]
        kl_steps = [h["step"] for h in history if "kl" in h]
        kls = [h["kl"] for h in history if "kl" in h]
        lr_steps = [h["step"] for h in history if "learning_rate" in h]
        lrs = [h["learning_rate"] for h in history if "learning_rate" in h]
        entropy_steps = [h["step"] for h in history if "entropy" in h]
        entropies = [h["entropy"] for h in history if "entropy" in h]

        # === Figure 1: Reward Dashboard (2x2) ===
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("GRPO Training — QuarterMaster (Rewards)", fontsize=16, fontweight="bold")

        # Top-left: Mean reward (raw + smoothed)
        ax = axes[0, 0]
        ax.plot(steps, means, color="#3b82f6", linewidth=1, alpha=0.4, label="Raw mean")
        if len(means) > 10:
            ax.plot(steps, _smooth(means, window), color="#ef4444", linewidth=2.5,
                    label=f"Smoothed (w={window})")
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean Reward")
        ax.set_title("Mean Reward per Step")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Top-right: Min/Max band
        ax = axes[0, 1]
        ax.fill_between(steps, mins, maxs, alpha=0.2, color="#8b5cf6", label="Min-Max range")
        ax.plot(steps, means, color="#3b82f6", linewidth=1.5, label="Mean")
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.set_title("Reward Range (Min/Max Band)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Bottom-left: Reward distribution histogram
        ax = axes[1, 0]
        all_rewards = []
        for r in _reward_log:
            all_rewards.extend(r["rewards"])
        ax.hist(all_rewards, bins=50, color="#22c55e", alpha=0.7, edgecolor="#000")
        mean_all = sum(all_rewards) / len(all_rewards) if all_rewards else 0
        ax.axvline(x=mean_all, color="#ef4444", linestyle="--", linewidth=2,
                   label=f"Mean={mean_all:.3f}")
        ax.set_xlabel("Reward")
        ax.set_ylabel("Count")
        ax.set_title("Reward Distribution (All Completions)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Bottom-right: Cumulative mean reward
        ax = axes[1, 1]
        cumulative = []
        running = 0
        for i, m in enumerate(means):
            running += m
            cumulative.append(running / (i + 1))
        ax.plot(steps, cumulative, color="#f59e0b", linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative Mean Reward")
        ax.set_title("Cumulative Mean Reward")
        ax.grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(output_dir, "grpo_reward_dashboard.png"), dpi=150)
        plt.close(fig)
        log.info(f"Reward dashboard saved to {output_dir}/grpo_reward_dashboard.png")

        # === Figure 2: Loss + KL + LR + Entropy (2x2) ===
        fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
        fig2.suptitle("GRPO Training — QuarterMaster (Optimization)", fontsize=16, fontweight="bold")

        # Top-left: Loss curve
        ax = axes2[0, 0]
        if loss_steps:
            ax.plot(loss_steps, losses, color="#3b82f6", linewidth=1, alpha=0.4, label="Raw loss")
            if len(losses) > 10:
                ax.plot(loss_steps, _smooth(losses, max(5, len(losses)//20)),
                        color="#ef4444", linewidth=2.5, label="Smoothed")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No loss data logged", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Policy Loss")
        ax.grid(True, alpha=0.3)

        # Top-right: KL divergence
        ax = axes2[0, 1]
        if kl_steps:
            ax.plot(kl_steps, kls, color="#a855f7", linewidth=1, alpha=0.5, label="KL")
            if len(kls) > 10:
                ax.plot(kl_steps, _smooth(kls, max(5, len(kls)//20)),
                        color="#c084fc", linewidth=2.5, label="Smoothed")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No KL data logged", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
        ax.set_xlabel("Step")
        ax.set_ylabel("KL Divergence")
        ax.set_title("KL Divergence from Reference")
        ax.grid(True, alpha=0.3)

        # Bottom-left: Learning rate schedule
        ax = axes2[1, 0]
        if lr_steps:
            ax.plot(lr_steps, lrs, color="#06b6d4", linewidth=2)
            ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
        else:
            ax.text(0.5, 0.5, "No LR data logged", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.3)

        # Bottom-right: Entropy
        ax = axes2[1, 1]
        if entropy_steps:
            ax.plot(entropy_steps, entropies, color="#f59e0b", linewidth=1, alpha=0.5, label="Entropy")
            if len(entropies) > 10:
                ax.plot(entropy_steps, _smooth(entropies, max(5, len(entropies)//20)),
                        color="#ef4444", linewidth=2.5, label="Smoothed")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No entropy data logged", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
        ax.set_xlabel("Step")
        ax.set_ylabel("Entropy")
        ax.set_title("Policy Entropy")
        ax.grid(True, alpha=0.3)

        fig2.tight_layout(rect=[0, 0, 1, 0.96])
        fig2.savefig(os.path.join(output_dir, "grpo_optimization_dashboard.png"), dpi=150)
        plt.close(fig2)
        log.info(f"Optimization dashboard saved to {output_dir}/grpo_optimization_dashboard.png")

        # === Figure 3: Per-task reward breakdown ===
        task_rewards_by_step = {}
        for entry in _reward_log:
            tasks_in_batch = entry.get("tasks", [])
            for task, reward in zip(tasks_in_batch, entry["rewards"]):
                task_rewards_by_step.setdefault(task, []).append(reward)

        if task_rewards_by_step:
            task_colors = {"easy": "#22c55e", "medium": "#f59e0b", "hard": "#ef4444"}
            n_tasks = len(task_rewards_by_step)
            fig3, axes3 = plt.subplots(1, max(n_tasks, 1) + 1, figsize=(6 * (n_tasks + 1), 5))
            fig3.suptitle("GRPO Training — Per-Task Reward Analysis", fontsize=14, fontweight="bold")
            if n_tasks + 1 == 1:
                axes3 = [axes3]

            # One histogram per task
            for i, (task, rewards) in enumerate(sorted(task_rewards_by_step.items())):
                ax = axes3[i]
                color = task_colors.get(task, "#888")
                ax.hist(rewards, bins=30, color=color, alpha=0.7, edgecolor="#000")
                mean_r = sum(rewards) / len(rewards)
                ax.axvline(x=mean_r, color="#000", linestyle="--", linewidth=2,
                           label=f"Mean={mean_r:.3f}")
                ax.set_xlabel("Reward")
                ax.set_ylabel("Count")
                ax.set_title(f"{task.title()} (n={len(rewards)})")
                ax.legend()
                ax.grid(True, alpha=0.3)

            # Summary bar chart
            ax = axes3[-1]
            labels, bar_means, bar_colors = [], [], []
            for task in ["easy", "medium", "hard"]:
                if task in task_rewards_by_step:
                    r = task_rewards_by_step[task]
                    labels.append(task)
                    bar_means.append(sum(r) / len(r))
                    bar_colors.append(task_colors.get(task, "#888"))
            ax.bar(labels, bar_means, color=bar_colors, width=0.5)
            ax.set_ylabel("Mean Reward")
            ax.set_title("Mean Reward Comparison")
            ax.grid(True, alpha=0.3, axis="y")

            fig3.tight_layout(rect=[0, 0, 1, 0.94])
            fig3.savefig(os.path.join(output_dir, "grpo_per_task_rewards.png"), dpi=150)
            plt.close(fig3)
            log.info(f"Per-task rewards saved to {output_dir}/grpo_per_task_rewards.png")

            # === Figure 3b: Per-task reward curves over training steps ===
            fig3b, axes3b = plt.subplots(1, 2, figsize=(14, 5))
            fig3b.suptitle("GRPO Training — Per-Task Reward Curves", fontsize=14, fontweight="bold")

            # Left: Rolling mean reward per task over time
            ax = axes3b[0]
            for task in ["easy", "medium", "hard"]:
                if task in task_rewards_by_step:
                    rewards = task_rewards_by_step[task]
                    color = task_colors.get(task, "#888")
                    # Plot raw as faint dots
                    ax.scatter(range(len(rewards)), rewards, color=color, alpha=0.1, s=4)
                    # Smoothed curve
                    if len(rewards) > 10:
                        w = max(5, len(rewards) // 15)
                        ax.plot(range(len(rewards)), _smooth(rewards, w),
                                color=color, linewidth=2.5, label=f"{task} (smoothed)")
                    else:
                        ax.plot(range(len(rewards)), rewards,
                                color=color, linewidth=1.5, label=task)
            ax.set_xlabel("Sample Index (within task)")
            ax.set_ylabel("Reward")
            ax.set_title("Reward Over Time (per task)")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Right: Cumulative mean per task
            ax = axes3b[1]
            for task in ["easy", "medium", "hard"]:
                if task in task_rewards_by_step:
                    rewards = task_rewards_by_step[task]
                    color = task_colors.get(task, "#888")
                    cum = []
                    running = 0
                    for i, r in enumerate(rewards):
                        running += r
                        cum.append(running / (i + 1))
                    ax.plot(range(len(cum)), cum, color=color, linewidth=2, label=task)
            ax.set_xlabel("Sample Index (within task)")
            ax.set_ylabel("Cumulative Mean Reward")
            ax.set_title("Cumulative Mean Reward (per task)")
            ax.legend()
            ax.grid(True, alpha=0.3)

            fig3b.tight_layout(rect=[0, 0, 1, 0.94])
            fig3b.savefig(os.path.join(output_dir, "grpo_per_task_curves.png"), dpi=150)
            plt.close(fig3b)
            log.info(f"Per-task curves saved to {output_dir}/grpo_per_task_curves.png")

        # === Figure 3c: Reward Sub-Component Decomposition ===
        if _component_log and len(_component_log) > 5:
            dense_keys = ["R_directives", "R_planning", "R_revenue", "R_fulfillment", "R_waste"]
            sparse_keys = ["milestone_bonus", "directive_penalty", "hard_penalty"]
            comp_steps = [c["step"] for c in _component_log]
            colors_dense = {"R_directives": "#ef4444", "R_planning": "#8b5cf6",
                            "R_revenue": "#22c55e", "R_fulfillment": "#3b82f6", "R_waste": "#f59e0b"}
            weights = {"R_directives": 0.40, "R_planning": 0.20, "R_revenue": 0.15,
                       "R_fulfillment": 0.15, "R_waste": 0.10}

            fig3c, axes3c = plt.subplots(1, 3, figsize=(20, 6))
            fig3c.suptitle("GRPO Training — Reward Decomposition", fontsize=14, fontweight="bold")

            # Left: Dense reward components over time (smoothed)
            ax = axes3c[0]
            for key in dense_keys:
                vals = [c.get(key, 0.0) for c in _component_log]
                w = max(5, len(vals) // 15)
                ax.plot(comp_steps, _smooth(vals, w), color=colors_dense[key],
                        linewidth=2, label=f"{key} ({weights[key]:.0%})")
            ax.set_xlabel("Step")
            ax.set_ylabel("Component Value (raw, [-1, +1])")
            ax.set_title("Dense Reward Components")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color="#000", linewidth=0.5, alpha=0.5)

            # Middle: Weighted contribution over time
            ax = axes3c[1]
            for key in dense_keys:
                vals = [c.get(key, 0.0) * weights[key] for c in _component_log]
                w = max(5, len(vals) // 15)
                ax.plot(comp_steps, _smooth(vals, w), color=colors_dense[key],
                        linewidth=2, label=f"{key}")
            ax.set_xlabel("Step")
            ax.set_ylabel("Weighted Contribution")
            ax.set_title("Weighted Dense Contributions")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color="#000", linewidth=0.5, alpha=0.5)

            # Right: Mean component values bar chart
            ax = axes3c[2]
            all_keys = dense_keys + sparse_keys
            bar_means = []
            bar_colors = []
            sparse_colors = {"milestone_bonus": "#22c55e", "directive_penalty": "#ef4444",
                             "hard_penalty": "#f97316"}
            for key in all_keys:
                vals = [c.get(key, 0.0) for c in _component_log]
                bar_means.append(sum(vals) / len(vals))
                bar_colors.append(colors_dense.get(key, sparse_colors.get(key, "#888")))
            short_labels = [k.replace("R_", "").replace("_", "\n") for k in all_keys]
            bars = ax.bar(short_labels, bar_means, color=bar_colors, width=0.6)
            for bar, val in zip(bars, bar_means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f"{val:.3f}", ha="center", va="bottom" if val >= 0 else "top",
                        fontsize=7, fontweight="bold")
            ax.set_ylabel("Mean Value")
            ax.set_title("Average Component Values")
            ax.grid(True, alpha=0.3, axis="y")
            ax.axhline(y=0, color="#000", linewidth=0.5)
            ax.tick_params(axis='x', labelsize=7)

            fig3c.tight_layout(rect=[0, 0, 1, 0.94])
            fig3c.savefig(os.path.join(output_dir, "grpo_reward_decomposition.png"), dpi=150)
            plt.close(fig3c)
            log.info(f"Reward decomposition saved to {output_dir}/grpo_reward_decomposition.png")

            # Save component log
            comp_path = os.path.join(output_dir, "grpo_component_log.json")
            with open(comp_path, "w") as f:
                json.dump(_component_log, f, indent=2)
            log.info(f"Component log saved to {comp_path}")

        # === Figure 4: Improvement (first half vs second half) ===
        if len(means) >= 10:
            fig4, ax4 = plt.subplots(1, 1, figsize=(8, 5))
            mid = len(means) // 2
            first_half = means[:mid]
            second_half = means[mid:]
            fh_mean = sum(first_half) / len(first_half)
            sh_mean = sum(second_half) / len(second_half)
            pct_change = ((sh_mean - fh_mean) / abs(fh_mean) * 100) if fh_mean != 0 else 0
            bars = ax4.bar(["First Half", "Second Half"], [fh_mean, sh_mean],
                           color=["#ef4444", "#22c55e"], width=0.5)
            ax4.set_ylabel("Mean Reward")
            ax4.set_title(f"Reward Improvement: {pct_change:+.1f}%")
            ax4.grid(True, alpha=0.3, axis="y")
            # Annotate bars
            for bar, val in zip(bars, [fh_mean, sh_mean]):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                         f"{val:.3f}", ha="center", va="bottom", fontweight="bold")
            fig4.tight_layout()
            fig4.savefig(os.path.join(output_dir, "grpo_improvement.png"), dpi=150)
            plt.close(fig4)
            log.info(f"Improvement chart saved to {output_dir}/grpo_improvement.png")

        # Save raw data
        data_path = os.path.join(output_dir, "grpo_training_log.json")
        with open(data_path, "w") as f:
            json.dump(_reward_log, f, indent=2)
        log.info(f"Training log saved to {data_path}")

        # Save trainer log history
        if history:
            trainer_log_path = os.path.join(output_dir, "grpo_trainer_log.json")
            with open(trainer_log_path, "w") as f:
                json.dump(history, f, indent=2)
            log.info(f"Trainer log saved to {trainer_log_path}")

    except ImportError:
        log.warning("matplotlib not installed — skipping curve plots. pip install matplotlib")
    except Exception as e:
        log.error(f"Failed to plot training curves: {e}")


def main():
    train_start = time.time()

    log.info(f"SFT model: {SFT_MODEL_DIR}")
    log.info(f"GRPO data: {GRPO_DATA_FILE}")
    log.info(f"Output: {OUTPUT_DIR}")
    log.info(f"Generations per prompt: {NUM_GENERATIONS}")
    log.info(f"Max seq length: {MAX_SEQ_LENGTH}")
    log.info(f"Max completion: {MAX_COMPLETION}")
    log.info(f"LoRA rank: {LORA_RANK}")
    log.info(f"Learning rate: {LEARNING_RATE}")

    dataset = load_grpo_dataset(GRPO_DATA_FILE)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=SFT_MODEL_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.9,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        max_completion_length=MAX_COMPLETION,
        num_generations=NUM_GENERATIONS,
        report_to="none",
        use_vllm=False,  # Unsloth's fast_inference handles vLLM internally

        # --- GRPO Training Tricks (see README) ---
        # DAPO: Asymmetric clipping prevents entropy collapse by allowing
        # exploration of low-probability tokens (wider upper bound)
        epsilon_low=0.2,
        epsilon_high=0.28,
        # Dr. GRPO: Fixed denominator normalization removes response-length
        # bias — longer wrong answers no longer get softer gradients
        loss_type="dr_grpo",
        # Mask truncated completions instead of assigning them negative
        # reward, which confuses the model about valid reasoning paths
        mask_truncated_completions=True,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=step_reward,
        args=training_args,
        train_dataset=dataset,
    )

    log.info("Starting single-turn GRPO training...")
    trainer.train()

    train_time = time.time() - train_start
    log.info(f"GRPO training finished in {train_time:.1f}s ({train_time/60:.1f}min)")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    log.info(f"Model saved to {OUTPUT_DIR}")

    # Plot training curves
    log.info("Generating training curves...")
    plot_grpo_curves(OUTPUT_DIR, trainer)

    # Final summary
    if _reward_log:
        all_means = [r["mean"] for r in _reward_log]
        log.info("=== GRPO TRAINING SUMMARY ===")
        log.info(f"  Total steps: {len(_reward_log)}")
        log.info(f"  Overall mean reward: {sum(all_means)/len(all_means):.3f}")
        log.info(f"  First 10% mean: {sum(all_means[:max(1,len(all_means)//10)])/max(1,len(all_means)//10):.3f}")
        log.info(f"  Last 10% mean: {sum(all_means[-max(1,len(all_means)//10):])/max(1,len(all_means)//10):.3f}")
        log.info(f"  Best step reward: {max(all_means):.3f}")
        log.info(f"  Training time: {train_time/60:.1f} minutes")

    log.info("GRPO training complete.")


if __name__ == "__main__":
    main()