"""Rule-based eco mask generation and optimization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import islice

import numpy as np

from .image_ops import binarize, distance_transform, to_float01
from .metrics import evaluate_tradeoff


@dataclass(frozen=True)
class RuleWeights:
    readability_weight: float = 2.0
    ink_weight: float = 0.4
    target_weight: float = 2.5
    topology_weight: float = 3.0


@dataclass(frozen=True)
class RuleParams:
    pattern: str
    spacing: float
    radius: float = 0.0
    width: float = 0.0
    angle: float = 0.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    min_distance: float = 1.5
    cutoff: float = 0.0
    region_width: float = 0.0


@dataclass(frozen=True)
class RuleResult:
    params: RuleParams
    remove_mask: np.ndarray
    eco: np.ndarray
    metrics: dict[str, float]
    loss: float

    def params_dict(self) -> dict[str, float | str]:
        return asdict(self.params)


def _dot_mask(shape: tuple[int, int], params: RuleParams) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    spacing = max(2.0, params.spacing)
    gx = np.mod(xx - params.offset_x + spacing / 2.0, spacing) - spacing / 2.0
    gy = np.mod(yy - params.offset_y + spacing / 2.0, spacing) - spacing / 2.0
    return ((gx * gx + gy * gy) <= params.radius * params.radius).astype(np.float32)


def _stripe_mask(shape: tuple[int, int], params: RuleParams) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    theta = np.deg2rad(params.angle)
    coord = xx * np.cos(theta) + yy * np.sin(theta)
    spacing = max(2.0, params.spacing)
    phase = np.mod(coord + params.offset_x, spacing)
    width = max(1.0, params.width)
    return ((phase < width) | (phase > spacing - width)).astype(np.float32)


def _center_cut_mask(dist: np.ndarray, params: RuleParams) -> np.ndarray:
    return (dist >= max(1.0, params.cutoff)).astype(np.float32)


def _thin_mask(dist: np.ndarray, params: RuleParams) -> np.ndarray:
    return ((dist > 0.0) & (dist <= max(0.5, params.cutoff))).astype(np.float32)


def _edge_region(dist: np.ndarray, params: RuleParams) -> np.ndarray:
    width = max(0.5, params.region_width)
    return ((dist > 0.0) & (dist <= width)).astype(np.float32)


def apply_rule(foreground: np.ndarray, params: RuleParams) -> tuple[np.ndarray, np.ndarray]:
    """Apply one rule and return remove mask plus eco foreground."""
    fg = to_float01(foreground)
    fg_binary = binarize(fg)
    dist = distance_transform(fg)
    interior = ((fg_binary > 0) & (dist >= params.min_distance)).astype(np.float32)
    foreground_region = (fg_binary > 0).astype(np.float32)

    if params.pattern == "dots":
        remove = _dot_mask(fg.shape, params) * interior
    elif params.pattern == "stripes":
        remove = _stripe_mask(fg.shape, params) * interior
    elif params.pattern == "center":
        remove = _center_cut_mask(dist, params) * interior
    elif params.pattern == "thin":
        remove = _thin_mask(dist, params) * foreground_region
    elif params.pattern == "edge_dots":
        remove = _dot_mask(fg.shape, params) * _edge_region(dist, params) * foreground_region
    elif params.pattern == "edge_stripes":
        remove = _stripe_mask(fg.shape, params) * _edge_region(dist, params) * foreground_region
    elif params.pattern == "slash":
        remove = _stripe_mask(fg.shape, params) * foreground_region
    else:
        raise ValueError(f"Unknown rule pattern: {params.pattern}")

    remove = np.clip(remove, 0.0, 1.0).astype(np.float32)
    eco = fg * (1.0 - remove)
    return remove, eco.astype(np.float32)


def candidate_params(target_saving: float, include_outline: bool = False) -> list[RuleParams]:
    """Generate a compact candidate set around the desired saving rate."""
    target = float(np.clip(target_saving, 0.05, 0.65))
    base_radius = 1.0 + target * 8.0
    base_spacing = 16.0 - target * 12.0
    params: list[RuleParams] = []

    for spacing in sorted({8.0, 10.0, 12.0, 14.0, round(base_spacing, 1)}):
        for radius in sorted({1.0, 1.5, 2.0, 2.5, round(base_radius, 1)}):
            if radius * 2.2 >= spacing:
                continue
            for ox, oy in [(0.0, 0.0), (spacing / 2.0, spacing / 2.0)]:
                params.append(
                    RuleParams(
                        pattern="dots",
                        spacing=spacing,
                        radius=radius,
                        offset_x=ox,
                        offset_y=oy,
                        min_distance=max(1.2, radius * 0.75),
                    )
                )

    for spacing in sorted({6.0, 8.0, 10.0, 12.0, round(base_spacing * 0.85, 1)}):
        for width in sorted({1.0, 1.5, 2.0, max(1.0, round(target * 5.0, 1))}):
            if width * 2.0 >= spacing:
                continue
            for angle in [0.0, 45.0, 90.0, 135.0]:
                params.append(
                    RuleParams(
                        pattern="stripes",
                        spacing=spacing,
                        width=width,
                        angle=angle,
                        offset_x=0.0,
                        min_distance=max(1.2, width),
                    )
                )

    for cutoff in sorted({2.0, 2.5, 3.0, 4.0, 5.0, 1.5 + target * 8.0}):
        params.append(
            RuleParams(
                pattern="center",
                spacing=1.0,
                cutoff=cutoff,
                min_distance=max(1.2, cutoff),
            )
        )

    if include_outline:
        for cutoff in sorted({0.7, 1.0, 1.3, 1.6, 2.0, max(0.8, 0.6 + target * 4.0)}):
            params.append(
                RuleParams(
                    pattern="thin",
                    spacing=1.0,
                    cutoff=cutoff,
                    min_distance=0.0,
                )
            )

        for spacing in sorted({7.0, 9.0, 11.0, 13.0, round(base_spacing, 1)}):
            for radius in sorted({1.0, 1.5, 2.0, max(1.0, round(base_radius * 0.6, 1))}):
                if radius * 2.0 >= spacing:
                    continue
                for region_width in [1.5, 2.5, 3.5, 4.5]:
                    params.append(
                        RuleParams(
                            pattern="edge_dots",
                            spacing=spacing,
                            radius=radius,
                            offset_x=0.0,
                            offset_y=0.0,
                            min_distance=0.0,
                            region_width=region_width,
                        )
                    )

        for spacing in sorted({6.0, 8.0, 10.0, 12.0}):
            for width in [0.8, 1.2, 1.6, 2.0]:
                if width * 2.0 >= spacing:
                    continue
                for angle in [0.0, 45.0, 90.0, 135.0]:
                    for region_width in [2.0, 3.0, 4.0]:
                        params.append(
                            RuleParams(
                                pattern="edge_stripes",
                                spacing=spacing,
                                width=width,
                                angle=angle,
                                offset_x=0.0,
                                min_distance=0.0,
                                region_width=region_width,
                            )
                        )

        for spacing in sorted({7.0, 9.0, 11.0, 13.0}):
            for width in [0.8, 1.2, 1.6]:
                if width * 2.0 >= spacing:
                    continue
                for angle in [45.0, 135.0]:
                    params.append(
                        RuleParams(
                            pattern="slash",
                            spacing=spacing,
                            width=width,
                            angle=angle,
                            offset_x=0.0,
                            min_distance=0.0,
                        )
                    )

    return params


def loss_for_metrics(metrics: dict[str, float], weights: RuleWeights) -> float:
    """Weighted tradeoff loss from guide-style metrics."""
    return float(
        weights.readability_weight * (1.0 - metrics["ssim"])
        + weights.ink_weight * metrics["ink_ratio"]
        + weights.target_weight * metrics.get("target_error", 0.0)
        + weights.topology_weight * metrics["topology_penalty"]
    )


def optimize_rule(
    foreground: np.ndarray,
    target_saving: float,
    weights: RuleWeights | None = None,
    candidate_limit: int | None = None,
    include_outline: bool = False,
) -> RuleResult:
    """Search rule candidates and return the best pseudo-label."""
    weights = weights or RuleWeights()
    candidates = candidate_params(target_saving, include_outline=include_outline)
    iterator = candidates if candidate_limit is None else list(islice(candidates, candidate_limit))

    best: RuleResult | None = None
    for params in iterator:
        remove, eco = apply_rule(foreground, params)
        metrics = evaluate_tradeoff(foreground, eco, target_saving=target_saving)
        loss = loss_for_metrics(metrics, weights)
        if best is None or loss < best.loss:
            best = RuleResult(params=params, remove_mask=remove, eco=eco, metrics=metrics, loss=loss)

    if best is None:
        raise RuntimeError("No rule candidates were generated")
    return best
