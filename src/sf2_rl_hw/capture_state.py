from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any, Dict, Optional

from .agents import load_agent, predict_action
from .config import load_experiment_config
from .envs import build_retro_env, prepare_integration_assets
from .rollout import reset_env, step_env
from .utils.logging import dump_json, emit_section
from .utils.paths import checkpoint_step_from_path, resolve_checkpoint_path


def run_capture_state(
    config_path: Path,
    checkpoint_override: Optional[Path] = None,
    output_name: Optional[str] = None,
    max_steps: int = 5000,
    render: bool = False,
) -> None:
    if max_steps <= 0:
        raise ValueError("max_steps must be greater than 0")

    config, resolved = load_experiment_config(config_path)
    checkpoint_path = resolve_checkpoint_path(
        explicit_checkpoint=checkpoint_override,
        configured_checkpoint=config.evaluation.checkpoint_path,
        output_dir=config.runtime.output_dir,
        experiment_name=config.experiment_name,
    )
    state_name = normalize_state_name(output_name or f"{config.env.state}.Round2Start")
    state_path = state_output_path(config, state_name)

    config.env.reset_round = False
    prepare_integration_assets(config.env)
    env = build_retro_env(
        env_config=config.env,
        reward_config=config.reward,
        seed=config.runtime.seed,
        render=render,
        monitor=False,
        terminate_on_result=False,
    )

    transition_info: Dict[str, Any] = {}
    try:
        model = load_agent(
            checkpoint_path=checkpoint_path,
            env=env,
            config=config.ppo,
            device=config.runtime.device,
        )
        observation, _ = reset_env(env)
        full_hp = config.reward.full_hp
        saw_terminal = False

        for agent_step in range(1, max_steps + 1):
            action = predict_action(model, observation, deterministic=True)
            observation, reward, done, info = step_env(env, action)
            result = str(info.get("result", "ongoing"))
            agent_hp = int(info.get("agent_hp", 0))
            enemy_hp = int(info.get("enemy_hp", 0))

            if result in {"win", "lose", "draw"}:
                saw_terminal = True

            if saw_terminal and result == "ongoing" and agent_hp == full_hp and enemy_hp == full_hp:
                write_state_bytes(state_path, emulator_state_bytes(env))
                transition_info = {
                    "source_checkpoint": str(checkpoint_path),
                    "checkpoint_step": checkpoint_step_from_path(checkpoint_path),
                    "source_state": config.env.state,
                    "captured_state": state_name,
                    "captured_path": str(state_path),
                    "trigger_agent_step": agent_step,
                    "trigger_env_step": int(info.get("env_step", 0)),
                    "trigger_result": result,
                    "trigger_agent_hp": agent_hp,
                    "trigger_enemy_hp": enemy_hp,
                    "last_reward": float(reward),
                    "render": render,
                    "max_steps": max_steps,
                }
                dump_json(state_path.with_suffix(".json"), transition_info)
                emit_section(
                    "Capture State",
                    [
                        f"checkpoint={checkpoint_path}",
                        f"saved_state={state_path}",
                        f"agent_step={agent_step}",
                        f"env_step={info.get('env_step', 0)}",
                    ],
                )
                return

            if done:
                raise RuntimeError(
                    "Environment terminated before the next round started. "
                    "Try a different checkpoint or disable custom round reset."
                )
    finally:
        env.close()

    raise RuntimeError(
        f"Did not detect the next round within {max_steps} agent steps. "
        f"No state file was created for {state_name}."
    )


def normalize_state_name(name: str) -> str:
    trimmed = name.strip()
    if not trimmed:
        raise ValueError("output state name cannot be empty")
    return trimmed[:-6] if trimmed.endswith(".state") else trimmed


def state_output_path(config: Any, state_name: str) -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "retro_data"
        / config.env.game
        / f"{state_name}.state"
    )


def emulator_state_bytes(env: Any) -> bytes:
    retro_env = getattr(env, "unwrapped", env)
    emulator = getattr(retro_env, "em", None)
    if emulator is None or not hasattr(emulator, "get_state"):
        raise RuntimeError("The retro emulator does not expose get_state(); cannot capture a new state.")
    return bytes(emulator.get_state())


def write_state_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        handle.write(payload)
