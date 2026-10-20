"""Fail-fast checks for checkpoint/config compatibility at inference time."""
from __future__ import annotations

from typing import Any


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def validate_checkpoint_runtime(payload: dict[str, Any], cfg: Any, *, hierarchical: bool) -> tuple[int, ...] | None:
    """Validate spatial/temporal settings saved with a checkpoint, when available.

    Older checkpoints that did not persist these values remain loadable when
    their weights are otherwise compatible; current checkpoints fail early with
    a descriptive error rather than silently changing the observation contract.
    Returns the checkpoint's pulse durations for a hierarchical model.
    """
    saved_config = None
    if hierarchical:
        extra = payload.get("extra", {})
        if isinstance(extra, dict):
            saved_config = extra.get("config")
        if saved_config is None:
            # Staged hierarchy checkpoints store a compact top-level `config`.
            saved_config = payload.get("config")
    else:
        saved_config = payload.get("config")

    saved_mesen = _get(saved_config, "mesen") if saved_config is not None else None
    current_mesen = cfg.mesen
    runtime_fields = (
        ("observation_width", int(current_mesen.observation_width)),
        ("observation_height", int(current_mesen.observation_height)),
        ("frame_skip", int(current_mesen.frame_skip)),
    )
    for field, current in runtime_fields:
        saved = _get(saved_mesen, field) if saved_mesen is not None else None
        if saved is not None and int(saved) != current:
            raise ValueError(
                f"Checkpoint/config mismatch for mesen.{field}: checkpoint={saved}, "
                f"active config={current}. Use the same observation resolution and frame skip "
                "for recording, training, and inference, or retrain from scratch."
            )

    if not hierarchical:
        return None
    raw_pulses = payload.get("button_pulse_durations", cfg.hierarchical.button_pulse_durations)
    pulse_durations = tuple(int(x) for x in raw_pulses)
    if not pulse_durations or any(x < 1 for x in pulse_durations):
        raise ValueError(f"Checkpoint has invalid button pulse durations: {pulse_durations}")
    if max(pulse_durations) > int(current_mesen.frame_skip):
        raise ValueError(
            f"Checkpoint button pulse durations {pulse_durations} exceed active frame_skip="
            f"{current_mesen.frame_skip}. Match the checkpoint's frame skip or retrain with compatible pulses."
        )
    configured_pulses = tuple(int(x) for x in cfg.hierarchical.button_pulse_durations)
    if pulse_durations != configured_pulses:
        raise ValueError(
            f"Checkpoint button_pulse_durations={pulse_durations} do not match active config "
            f"{configured_pulses}. Use the training config or retrain with the desired pulse choices."
        )
    return pulse_durations
