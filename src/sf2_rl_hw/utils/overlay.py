from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SUPPORTED_OVERLAY_FIELDS = [
    "experiment_name",
    "checkpoint_step",
    "env_step",
    "action",
    "instant_reward",
    "episode_return",
    "agent_hp",
    "enemy_hp",
    "result",
]


def format_overlay_lines(overlay_state: Dict[str, Any]) -> List[str]:
    return format_overlay_lines_with_fields(overlay_state, SUPPORTED_OVERLAY_FIELDS)


def format_overlay_lines_with_fields(
    overlay_state: Dict[str, Any],
    fields: List[str],
) -> List[str]:
    label_map = {
        "experiment_name": lambda state: f"exp: {state['experiment_name']}",
        "checkpoint_step": lambda state: f"step: {state['checkpoint_step']}",
        "env_step": lambda state: f"env_step: {state['env_step']}",
        "action": lambda state: f"action: {state['action']}",
        "instant_reward": lambda state: f"reward: {state['instant_reward']:.4f}",
        "episode_return": lambda state: f"return: {state['episode_return']:.4f}",
        "agent_hp": lambda state: f"agent_hp: {state['agent_hp']}",
        "enemy_hp": lambda state: f"enemy_hp: {state['enemy_hp']}",
        "result": lambda state: f"result: {state['result']}",
    }
    return [label_map[field](overlay_state) for field in fields if field in label_map]


def draw_overlay(
    frame: np.ndarray,
    overlay_state: Dict[str, Any],
    fields: Optional[List[str]] = None,
) -> np.ndarray:
    image = Image.fromarray(_ensure_rgb(frame)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    lines = format_overlay_lines_with_fields(
        overlay_state,
        SUPPORTED_OVERLAY_FIELDS if fields is None else fields,
    )
    if not lines:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    margin = 6
    padding_x = 6
    padding_y = 5
    line_spacing = 2
    line_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    text_width = max((bbox[2] - bbox[0]) for bbox in line_boxes)
    line_heights = [bbox[3] - bbox[1] for bbox in line_boxes]
    text_height = sum(line_heights) + line_spacing * max(0, len(lines) - 1)
    available_width = frame.shape[1] - margin * 2
    available_height = frame.shape[0] - margin * 2
    if available_width <= 0 or available_height <= 0:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    box_width = max(1, min(available_width, text_width + padding_x * 2))
    box_height = max(1, min(available_height, text_height + padding_y * 2))

    draw.rounded_rectangle(
        (margin, margin, margin + box_width, margin + box_height),
        radius=6,
        fill=(0, 0, 0, 144),
    )

    y = margin + padding_y
    for line in lines:
        draw.text((margin + padding_x, y), line, font=font, fill=(255, 255, 255, 255))
        _, _, _, bottom = draw.textbbox((margin + padding_x, y), line, font=font)
        y = bottom + line_spacing

    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"), dtype=np.uint8)


def _ensure_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return np.repeat(frame[:, :, None], 3, axis=2)
    if frame.ndim == 3 and frame.shape[-1] == 1:
        return np.repeat(frame, 3, axis=2)
    if frame.ndim == 3 and frame.shape[-1] >= 3:
        return frame[:, :, :3]
    raise ValueError("Frame must be 2D grayscale or 3-channel RGB.")


def detect_black_border_crop_box(frame: np.ndarray, threshold: int = 0) -> Optional[tuple[int, int, int, int]]:
    rgb = _ensure_rgb(frame)
    active_mask = np.any(rgb > threshold, axis=-1)
    active_rows = np.where(np.any(active_mask, axis=1))[0]
    active_cols = np.where(np.any(active_mask, axis=0))[0]
    if active_rows.size == 0 or active_cols.size == 0:
        return None

    top = int(active_rows[0])
    bottom = int(active_rows[-1]) + 1
    left = int(active_cols[0])
    right = int(active_cols[-1]) + 1

    # Keep encoded video dimensions friendly to yuv420p by enforcing even width/height.
    if (bottom - top) % 2 != 0 and bottom - top > 1:
        bottom -= 1
    if (right - left) % 2 != 0 and right - left > 1:
        right -= 1

    if top == 0 and left == 0 and bottom == rgb.shape[0] and right == rgb.shape[1]:
        return None
    return (top, bottom, left, right)


def crop_frame(frame: np.ndarray, crop_box: Optional[tuple[int, int, int, int]]) -> np.ndarray:
    if crop_box is None:
        return frame
    top, bottom, left, right = crop_box
    return np.asarray(frame[top:bottom, left:right], dtype=np.uint8)
