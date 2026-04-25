"""
Generate GRPO training dataset by running inference and saving replay data.

For each step in each episode, saves:
    - observation: the formatted observation text (exactly what the model sees)
    - task_name: easy/medium/hard
    - prior_actions: JSON list of action dicts for days 1..N-1 (to replay env state)
    - day: current day number
    - episode: episode number

The GRPO trainer loads this dataset as prompts. For each prompt, the model
generates G completions (online). Each completion is parsed into an action,
the env is replayed to that step, stepped once, and the real reward is returned.

Usage:
    API_BASE_URL=... API_KEY=... MODEL_NAME=Qwen/Qwen3-32B python generate_grpo_data.py

Output: grpo_data.jsonl
"""

import os
import json
import time
import logging

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

from server.inventory_env import InventoryEnvironment
from models import InventoryAction
from inference import SYSTEM_PROMPT, format_observation, parse_action

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("generate_grpo_data")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen3-32B"
OUTPUT_FILE = os.getenv("OUTPUT_FILE") or "grpo_data.jsonl"
NUM_EPISODES = int(os.getenv("NUM_EPISODES") or "1")
TASKS_TO_RUN = os.getenv("TASKS_TO_RUN") or "easy,medium,hard"


def action_to_dict(action):
    """Convert InventoryAction to a minimal serializable dict."""
    d = {}
    if action.buy_quantities:
        d["buy_quantities"] = action.buy_quantities
    if action.delivery_methods:
        d["delivery_methods"] = action.delivery_methods
    if action.liquidate:
        d["liquidate"] = action.liquidate
    if action.price_multipliers:
        d["price_multipliers"] = action.price_multipliers
    if action.notes_to_self:
        d["notes_to_self"] = action.notes_to_self
    if action.weekly_plan is not None:
        d["weekly_plan"] = action.weekly_plan
    if action.take_loan:
        d["take_loan"] = True
    return d


def run_episode(client, task_name, episode_num):
    """Run one episode, collect GRPO training data for every step."""
    env = InventoryEnvironment(task_name)
    obs = env.reset()

    examples = []
    prior_actions = []
    rewards = []
    ep_start = time.time()

    for day in range(1, env.max_days + 1):
        if obs.done:
            break

        obs_text = format_observation(obs)

        # Save this step's training example
        examples.append({
            "observation": obs_text,
            "task_name": task_name,
            "prior_actions": json.dumps(prior_actions),
            "day": day,
            "episode": episode_num,
        })

        # Get action from baseline model (same as inference.py)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": obs_text},
        ]

        step_start = time.time()
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.6,
                max_completion_tokens=800,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            log.warning(f"[{task_name}] ep{episode_num} day{day}: API error: {exc}")
            response_text = "{}"

        action = parse_action(response_text)
        action_dict = action_to_dict(action)

        # Step environment
        obs = env.step(action)
        rewards.append(obs.reward)

        # Record this action for future replay
        prior_actions.append(action_dict)

        step_ms = (time.time() - step_start) * 1000
        log.info(
            f"[{task_name}] ep{episode_num} day{day:02d}: "
            f"reward={obs.reward:+.2f} profit=${obs.total_profit:.0f} "
            f"cash=${obs.total_cash:.0f} "
            f"violations={len(obs.directive_violations_last_step)} "
            f"({step_ms:.0f}ms)"
        )

    ep_time = time.time() - ep_start
    avg_reward = sum(rewards) / len(rewards) if rewards else 0
    log.info(
        f"[{task_name}] ep{episode_num} DONE: {len(examples)} steps, "
        f"avg_reward={avg_reward:.3f}, final_profit=${obs.total_profit:.0f}, "
        f"time={ep_time:.1f}s"
    )

    return examples, rewards


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    tasks = [t.strip() for t in TASKS_TO_RUN.split(",")]

    log.info(f"Model: {MODEL_NAME}")
    log.info(f"Episodes per task: {NUM_EPISODES}")
    log.info(f"Tasks: {tasks}")
    log.info(f"Output: {OUTPUT_FILE}")

    all_examples = []
    all_rewards = {}
    run_start = time.time()

    for task_name in tasks:
        task_rewards = []
        for ep in range(1, NUM_EPISODES + 1):
            log.info(f"--- [{task_name}] Episode {ep}/{NUM_EPISODES} ---")
            examples, rewards = run_episode(client, task_name, ep)
            all_examples.extend(examples)
            task_rewards.extend(rewards)

            # Write continuously after each episode so partial progress is saved
            with open(OUTPUT_FILE, "a") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")
            log.info(f"Flushed {len(examples)} prompts to {OUTPUT_FILE}")

        all_rewards[task_name] = task_rewards
        if task_rewards:
            log.info(
                f"[{task_name}] SUMMARY: {len(task_rewards)} steps, "
                f"avg_reward={sum(task_rewards)/len(task_rewards):.3f}, "
                f"min={min(task_rewards):.3f}, max={max(task_rewards):.3f}"
            )

    total_time = time.time() - run_start
    log.info(f"Total {len(all_examples)} prompts in {OUTPUT_FILE} ({total_time:.1f}s total)")

    # Save reward stats for analysis
    stats_file = OUTPUT_FILE.replace(".jsonl", "_stats.json")
    stats = {}
    for task_name in tasks:
        task_examples = [e for e in all_examples if e["task_name"] == task_name]
        if task_examples and all_rewards.get(task_name):
            stats[task_name] = {
                "count": len(task_examples),
                "mean_reward": sum(all_rewards[task_name]) / len(all_rewards[task_name]),
                "min_reward": min(all_rewards[task_name]),
                "max_reward": max(all_rewards[task_name]),
                "rewards_by_day": all_rewards[task_name],
            }
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Stats saved to {stats_file}")

    log.info("=== FINAL STATS ===")
    for task_name in tasks:
        task_examples = [e for e in all_examples if e["task_name"] == task_name]
        if task_examples:
            days = [e["day"] for e in task_examples]
            log.info(f"  {task_name}: {len(task_examples)} steps, days {min(days)}-{max(days)}")


if __name__ == "__main__":
    main()