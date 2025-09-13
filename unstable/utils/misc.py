import csv, json
from unstable._types import GameInformation


def write_training_data_to_file(batch, filename: str):
    with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['pid', 'obs', 'act', 'reward', "env_id", "step_info"])  # header
        for step in batch: writer.writerow([step.pid, step.obs, step.act, step.reward, step.env_id, step.step_info])

def write_game_information_to_file(game_info: GameInformation, filename: str) -> None:
    """Write per-turn game information including full prompts.

    Added 'prompt' column to capture the exact input shown to the model
    (distinct from the raw environment observation) for better reproducibility.
    Backwards compatibility: existing readers can ignore the new column.
    """
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "game_idx","turn_idx","pid","name","obs","prompt","full_action","extracted_action","step_info",
                # rewards
                "final_reward","final_reward_transformed",
                "step_reward_raw_final","step_reward_env","step_reward_shaped",
                # eval meta
                "eval_model_pid","eval_opponent_name"
            ],
        )
        writer.writeheader()
        n_turns = game_info.num_turns or len(game_info.obs)
        for t in range(n_turns):
            pid = game_info.pid[t] if t < len(game_info.pid) else None
            row = {
                "game_idx": game_info.game_idx,
                "turn_idx": t,
                "pid": pid,
                "name": game_info.names.get(pid, ""),
                "obs": game_info.obs[t] if t < len(game_info.obs) else "",
                "prompt": game_info.prompts[t] if t < len(game_info.prompts) else "",
                "full_action": game_info.full_actions[t] if t < len(game_info.full_actions) else "",
                "extracted_action": game_info.extracted_actions[t] if t < len(game_info.extracted_actions) else "",
                "step_info": json.dumps(game_info.step_infos[t] if t < len(game_info.step_infos) else {}, ensure_ascii=False),
                "final_reward": game_info.final_rewards.get(pid, ""),
                "final_reward_transformed": (game_info.final_rewards_transformed.get(pid, "") if hasattr(game_info, "final_rewards_transformed") else ""),
                "step_reward_raw_final": (game_info.step_rewards_raw_final[t] if hasattr(game_info, "step_rewards_raw_final") and t < len(game_info.step_rewards_raw_final) else ""),
                "step_reward_env": (game_info.step_rewards_env[t] if hasattr(game_info, "step_rewards_env") and t < len(game_info.step_rewards_env) else ""),
                "step_reward_shaped": (game_info.step_rewards_shaped[t] if hasattr(game_info, "step_rewards_shaped") and t < len(game_info.step_rewards_shaped) else ""),
                "eval_model_pid": game_info.eval_model_pid,
                "eval_opponent_name": game_info.eval_opponent_name,
            }
            writer.writerow(row)
