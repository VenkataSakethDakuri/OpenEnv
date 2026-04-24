"""
Generate SFT training examples by running inference with a strong model.

Collects (system_prompt, observation, response) triples from successful episodes.
Only saves examples where JSON parsing succeeded without errors.
Saves ALL valid examples with metadata — filter by reward later for SFT.

Usage:
    API_BASE_URL=... API_KEY=... MODEL_NAME=gpt-5.4 python generate_sft_data.py

Output: sft_data.jsonl — one JSON object per line with fields:
    - system: system prompt
    - user: observation prompt
    - assistant: model response (reasoning + JSON action)
    - task: easy/medium/hard
    - day: which day in the episode
    - episode: episode number
    - reward: step reward received
"""

import os
import json
import time
import logging

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

from server.inventory_env import InventoryEnvironment
from server.constants import EXTRA_INVENTORY_COST, EVENT_DURATION
from models import InventoryAction
from inference import SYSTEM_PROMPT, format_observation, parse_action

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("generate_sft_data")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-5.4"
OUTPUT_FILE = os.getenv("OUTPUT_FILE") or "sft_data.jsonl"
NUM_EPISODES = int(os.getenv("NUM_EPISODES") or "1")


def run_episode_and_collect(client, task_name, episode_num):
    """Run one episode, return list of valid (prompt, response, metadata) examples."""
    env = InventoryEnvironment(task_name)
    obs = env.reset()
    examples = []
    parse_failures = 0
    rewards = []
    ep_start = time.time()

    for day in range(1, env.max_days + 1):
        if obs.done:
            break

        user_prompt = format_observation(obs)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
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
            obs = env.step(InventoryAction())
            parse_failures += 1
            continue

        # Try parsing — only keep if it succeeds cleanly
        parsed_ok = False
        try:
            text = response_text.strip()

            # Find JSON in response
            if "```" in text:
                parts = text.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        text = part
                        break

            start = text.find("{")
            if start != -1:
                depth = 0
                end = -1
                for i in range(start, len(text)):
                    if text[i] == "{":
                        depth += 1
                    elif text[i] == "}":
                        depth -= 1
                        if depth == 0:
                            end = i
                            break

                if end != -1:
                    json_str = text[start:end + 1]
                    data = json.loads(json_str)
                    valid_keys = {"buy_quantities", "delivery_methods", "liquidate",
                                  "price_multipliers", "notes_to_self", "weekly_plan",
                                  "take_loan"}
                    if any(k in data for k in valid_keys):
                        parsed_ok = True
        except (json.JSONDecodeError, ValueError):
            parsed_ok = False

        # Step the environment with the parsed action
        action = parse_action(response_text)
        obs = env.step(action)
        rewards.append(obs.reward)

        step_ms = (time.time() - step_start) * 1000
        log.info(
            f"[{task_name}] ep{episode_num} day{day:02d}: "
            f"reward={obs.reward:+.2f} profit=${obs.total_profit:.0f} "
            f"cash=${obs.total_cash:.0f} parsed={parsed_ok} "
            f"violations={len(obs.directive_violations_last_step)} "
            f"({step_ms:.0f}ms)"
        )

        if parsed_ok:
            examples.append({
                "system": SYSTEM_PROMPT,
                "user": user_prompt,
                "assistant": response_text,
                "task": task_name,
                "day": day,
                "episode": episode_num,
                "reward": obs.reward,
            })
        else:
            parse_failures += 1

    ep_time = time.time() - ep_start
    avg_reward = sum(rewards) / len(rewards) if rewards else 0
    log.info(
        f"[{task_name}] ep{episode_num} DONE: {len(examples)} valid, "
        f"{parse_failures} failures, avg_reward={avg_reward:.3f}, "
        f"final_profit=${obs.total_profit:.0f}, time={ep_time:.1f}s"
    )

    return examples, parse_failures


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    log.info(f"Model: {MODEL_NAME}")
    log.info(f"Episodes per task: {NUM_EPISODES}")
    log.info(f"Output: {OUTPUT_FILE}")

    all_examples = []
    run_start = time.time()

    for task_name in ["easy", "medium", "hard"]:
        task_total = 0
        task_failures = 0
        task_rewards = []

        for ep in range(1, NUM_EPISODES + 1):
            log.info(f"--- [{task_name}] Episode {ep}/{NUM_EPISODES} ---")

            examples, failures = run_episode_and_collect(client, task_name, ep)
            task_total += len(examples)
            task_failures += failures
            task_rewards.extend([e["reward"] for e in examples])
            all_examples.extend(examples)

        avg = sum(task_rewards) / len(task_rewards) if task_rewards else 0
        log.info(
            f"[{task_name}] SUMMARY: {task_total} valid, {task_failures} failures, "
            f"avg_reward={avg:.3f}"
        )

    # Append all examples
    with open(OUTPUT_FILE, "a") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    total_time = time.time() - run_start
    log.info(f"Appended {len(all_examples)} examples to {OUTPUT_FILE} ({total_time:.1f}s total)")

    # Final stats
    log.info("=== FINAL STATS ===")
    for task_name in ["easy", "medium", "hard"]:
        task_examples = [e for e in all_examples if e["task"] == task_name]
        if task_examples:
            avg_reward = sum(e["reward"] for e in task_examples) / len(task_examples)
            max_reward = max(e["reward"] for e in task_examples)
            min_reward = min(e["reward"] for e in task_examples)
            log.info(
                f"  {task_name}: {len(task_examples)} examples, "
                f"reward avg={avg_reward:.3f} min={min_reward:.3f} max={max_reward:.3f}"
            )

    # Save reward distribution for analysis
    stats_file = OUTPUT_FILE.replace(".jsonl", "_stats.json")
    stats = {}
    for task_name in ["easy", "medium", "hard"]:
        task_rewards = [e["reward"] for e in all_examples if e["task"] == task_name]
        if task_rewards:
            stats[task_name] = {
                "count": len(task_rewards),
                "mean": sum(task_rewards) / len(task_rewards),
                "min": min(task_rewards),
                "max": max(task_rewards),
                "rewards_by_day": task_rewards,
            }
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Stats saved to {stats_file}")


if __name__ == "__main__":
    main()