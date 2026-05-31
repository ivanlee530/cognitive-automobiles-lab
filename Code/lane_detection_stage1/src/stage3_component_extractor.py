"""Weighted connected-component lane candidate extraction for Stage 3."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ComponentExtractionConfig:
    min_area: int = 6
    min_height: int = 2
    min_y_coverage: float = 0.025
    min_elongation: float = 1.8
    max_horizontal_aspect_ratio: float = 3.0
    min_horizontal_blob_width: int = 24
    long_lane_y_coverage: float = 0.25
    compact_lane_y_coverage: float = 0.50
    min_edge_support: float = 0.04
    bottom_clutter_y_ratio: float = 0.82
    max_bottom_clutter_y_coverage: float = 0.07
    min_bottom_clutter_elongation: float = 3.0
    large_blob_min_area: int = 900
    max_lane_fill_ratio: float = 0.78
    max_lane_stroke_width: float = 18.0
    max_component_area_ratio: float = 0.03
    weak_min_score: float = 0.30
    strong_min_score: float = 0.68
    accepted_close_kernel: int = 3
    accepted_dilate_kernel: int = 3
    accepted_dilate_iterations: int = 0


@dataclass(frozen=True)
class ComponentCandidate:
    component_id: int
    status: str
    usable_for_fitting: bool
    candidate_weight: float
    score: float
    hard_reject_reason: str
    weak_reason: str
    area: int
    area_ratio: float
    x: int
    y: int
    width: int
    height: int
    centroid_x: float
    centroid_y: float
    y_coverage: float
    horizontal_aspect_ratio: float
    elongation: float
    fill_ratio: float
    edge_support: float
    stroke_width: float
    minor_axis: float
    major_axis: float
    contour: np.ndarray


def extract_component_candidates(
    component_source_mask: np.ndarray,
    edge_mask: np.ndarray,
    config: ComponentExtractionConfig | None = None,
) -> list[ComponentCandidate]:
    """Extract weighted lane support from the edge-constrained source mask."""
    if config is None:
        config = ComponentExtractionConfig()
    if component_source_mask.ndim != 2 or edge_mask.ndim != 2:
        raise ValueError("component extraction expects single-channel masks")
    if component_source_mask.shape != edge_mask.shape:
        raise ValueError("component source and edge masks must have the same shape")

    binary = (component_source_mask > 0).astype(np.uint8)
    label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    image_height, image_width = binary.shape
    image_area = max(1.0, float(image_height * image_width))
    candidates = []

    for component_id in range(1, label_count):
        x, y, width, height, area = (int(value) for value in stats[component_id])
        component_mask = np.uint8(labels == component_id) * 255
        contours, _hierarchy = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        elongation = _contour_elongation(contour)
        minor_axis, major_axis = _min_area_rect_axes(contour)
        y_coverage = height / max(1.0, float(image_height))
        horizontal_aspect_ratio = width / max(1.0, float(height))
        fill_ratio = area / max(1.0, float(width * height))
        area_ratio = area / image_area
        edge_pixels = cv2.countNonZero(cv2.bitwise_and(edge_mask, component_mask))
        edge_support = edge_pixels / max(1.0, float(area))
        stroke_width = _estimate_stroke_width(component_mask)
        score = _candidate_score(
            area=area,
            height=height,
            y_coverage=y_coverage,
            elongation=elongation,
            fill_ratio=fill_ratio,
            edge_support=edge_support,
            stroke_width=stroke_width,
            config=config,
        )
        hard_reject_reason = _hard_reject_reason(
            area=area,
            area_ratio=area_ratio,
            y=y,
            height=height,
            image_height=image_height,
            y_coverage=y_coverage,
            elongation=elongation,
            fill_ratio=fill_ratio,
            edge_support=edge_support,
            stroke_width=stroke_width,
            config=config,
        )
        weak_reasons = _weak_reasons(
            area=area,
            width=width,
            height=height,
            y_coverage=y_coverage,
            horizontal_aspect_ratio=horizontal_aspect_ratio,
            elongation=elongation,
            edge_support=edge_support,
            score=score,
            config=config,
        )
        status = _candidate_status(hard_reject_reason, weak_reasons, score, config)
        candidates.append(
            ComponentCandidate(
                component_id=component_id,
                status=status,
                usable_for_fitting=status != "reject",
                candidate_weight=_candidate_weight(status, score, config),
                score=float(score),
                hard_reject_reason=hard_reject_reason,
                weak_reason="|".join(weak_reasons),
                area=area,
                area_ratio=float(area_ratio),
                x=x,
                y=y,
                width=width,
                height=height,
                centroid_x=float(centroids[component_id][0]),
                centroid_y=float(centroids[component_id][1]),
                y_coverage=float(y_coverage),
                horizontal_aspect_ratio=float(horizontal_aspect_ratio),
                elongation=float(elongation),
                fill_ratio=float(fill_ratio),
                edge_support=float(edge_support),
                stroke_width=float(stroke_width),
                minor_axis=float(minor_axis),
                major_axis=float(major_axis),
                contour=contour,
            )
        )

    return sorted(candidates, key=lambda item: item.area, reverse=True)


def draw_candidate_status_overlay(
    roi_bgr: np.ndarray,
    candidates: list[ComponentCandidate],
) -> np.ndarray:
    """Draw strong, weak, and rejected candidates for visual QA."""
    debug = roi_bgr.copy()
    status_counts = Counter(candidate.status for candidate in candidates)
    for candidate in candidates:
        color = _status_color(candidate.status)
        thickness = 3 if candidate.status == "strong" else 2
        cv2.drawContours(debug, [candidate.contour], -1, color, thickness)
        reason = candidate.hard_reject_reason or candidate.weak_reason
        label = f"#{candidate.component_id} {candidate.status}"
        if reason:
            label = f"{label} {reason.replace('_', ' ')}"
        cv2.putText(
            debug,
            label,
            (candidate.x, max(16, candidate.y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            color,
            1,
            cv2.LINE_AA,
        )

    summary = " ".join(
        f"{status}={status_counts.get(status, 0)}"
        for status in ("strong", "weak", "reject")
    )
    cv2.rectangle(debug, (0, 0), (debug.shape[1] - 1, 35), (0, 0, 0), -1)
    cv2.putText(
        debug,
        summary,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return debug


def draw_hard_reject_overlay(
    roi_bgr: np.ndarray,
    candidates: list[ComponentCandidate],
    alpha: float = 0.42,
) -> np.ndarray:
    """Overlay hard rejects and annotate the reason for each ignored blob."""
    rejected = [candidate for candidate in candidates if candidate.status == "reject"]
    mask = make_hard_reject_mask(roi_bgr.shape[:2], candidates)
    debug = draw_mask_overlay(roi_bgr, mask, color=(30, 60, 230), alpha=alpha)
    reasons = Counter(candidate.hard_reject_reason for candidate in rejected)

    for candidate in rejected:
        cv2.drawContours(debug, [candidate.contour], -1, (30, 60, 230), 1)
        cv2.putText(
            debug,
            f"#{candidate.component_id} {candidate.hard_reject_reason.replace('_', ' ')}",
            (candidate.x, max(16, candidate.y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (30, 60, 230),
            1,
            cv2.LINE_AA,
        )

    summary = " ".join(
        f"{reason}={count}" for reason, count in sorted(reasons.items())
    )
    cv2.rectangle(debug, (0, 0), (debug.shape[1] - 1, 58), (0, 0, 0), -1)
    cv2.putText(
        debug,
        f"hard_reject={len(rejected)}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        debug,
        summary,
        (10, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (190, 190, 255),
        1,
        cv2.LINE_AA,
    )
    return debug


def make_strong_component_mask(
    image_shape: tuple[int, int],
    candidates: list[ComponentCandidate],
) -> np.ndarray:
    return _make_status_mask(image_shape, candidates, {"strong"})


def make_weak_component_mask(
    image_shape: tuple[int, int],
    candidates: list[ComponentCandidate],
) -> np.ndarray:
    return _make_status_mask(image_shape, candidates, {"weak"})


def make_fitting_support_mask(
    image_shape: tuple[int, int],
    candidates: list[ComponentCandidate],
    config: ComponentExtractionConfig | None = None,
) -> np.ndarray:
    """Create the fitting input mask from both strong and weak candidates."""
    if config is None:
        config = ComponentExtractionConfig()
    mask = _make_status_mask(image_shape, candidates, {"strong", "weak"})

    close_kernel_size = _ensure_odd_kernel(config.accepted_close_kernel)
    if close_kernel_size > 1:
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_kernel_size, close_kernel_size),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    dilate_iterations = max(0, int(config.accepted_dilate_iterations))
    if dilate_iterations > 0:
        dilate_kernel_size = _ensure_odd_kernel(config.accepted_dilate_kernel)
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilate_kernel_size, dilate_kernel_size),
        )
        mask = cv2.dilate(mask, dilate_kernel, iterations=dilate_iterations)
    return mask


def make_hard_reject_mask(
    image_shape: tuple[int, int],
    candidates: list[ComponentCandidate],
) -> np.ndarray:
    return _make_status_mask(image_shape, candidates, {"reject"})


def draw_mask_overlay(
    roi_bgr: np.ndarray,
    mask_image: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.58,
) -> np.ndarray:
    """Overlay one binary mask on the ROI with a single review color."""
    overlay = roi_bgr.copy()
    active = mask_image > 0
    if not np.any(active):
        return overlay

    highlight = np.zeros_like(overlay)
    highlight[:, :] = color
    blended = cv2.addWeighted(overlay, 1.0 - alpha, highlight, alpha, 0.0)
    overlay[active] = blended[active]
    return overlay


def _make_status_mask(
    image_shape: tuple[int, int],
    candidates: list[ComponentCandidate],
    statuses: set[str],
) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    contours = [
        candidate.contour for candidate in candidates if candidate.status in statuses
    ]
    if contours:
        cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)
    return mask


def _hard_reject_reason(
    *,
    area: int,
    area_ratio: float,
    y: int,
    height: int,
    image_height: int,
    y_coverage: float,
    elongation: float,
    fill_ratio: float,
    edge_support: float,
    stroke_width: float,
    config: ComponentExtractionConfig,
) -> str:
    if area < config.min_area:
        return "area_too_small"
    if height < config.min_height:
        return "height_too_small"
    if (
        area > config.large_blob_min_area
        and fill_ratio > config.max_lane_fill_ratio
    ):
        return "large_filled_bright_region"
    if (
        area_ratio > config.max_component_area_ratio
        and fill_ratio > config.max_lane_fill_ratio * 0.80
    ):
        return "very_large_area_ratio"
    if stroke_width > config.max_lane_stroke_width:
        return "stroke_width_too_large"
    if edge_support <= 0.0:
        return "edge_support_zero"
    if (
        y >= image_height * config.bottom_clutter_y_ratio
        and y_coverage < config.max_bottom_clutter_y_coverage
        and elongation < config.min_bottom_clutter_elongation
    ):
        return "bottom_clutter"
    return ""


def _weak_reasons(
    *,
    area: int,
    width: int,
    height: int,
    y_coverage: float,
    horizontal_aspect_ratio: float,
    elongation: float,
    edge_support: float,
    score: float,
    config: ComponentExtractionConfig,
) -> list[str]:
    reasons = []
    if (
        width >= config.min_horizontal_blob_width
        and horizontal_aspect_ratio > config.max_horizontal_aspect_ratio
        and y_coverage < config.long_lane_y_coverage
    ):
        reasons.append("horizontal_aspect_ratio_high")
    if y_coverage < config.min_y_coverage:
        reasons.append("y_coverage_low")
    if (
        elongation < config.min_elongation
        and y_coverage < config.compact_lane_y_coverage
    ):
        reasons.append("elongation_low")
    if 0.0 < edge_support < config.min_edge_support:
        reasons.append("edge_support_low")
    if height < max(8, config.min_height * 3) and area < max(48, config.min_area * 6):
        reasons.append("short_dashed_line_segment")
    if score < config.weak_min_score:
        reasons.append("score_below_weak_min")
    return reasons


def _candidate_status(
    hard_reject_reason: str,
    weak_reasons: list[str],
    score: float,
    config: ComponentExtractionConfig,
) -> str:
    if hard_reject_reason:
        return "reject"
    if not weak_reasons and score >= config.strong_min_score:
        return "strong"
    return "weak"


def _candidate_weight(
    status: str,
    score: float,
    config: ComponentExtractionConfig,
) -> float:
    if status == "reject":
        return 0.0
    if status == "strong":
        return 1.0
    span = max(1e-6, config.strong_min_score - config.weak_min_score)
    scaled = (score - config.weak_min_score) / span
    return float(0.30 + 0.30 * np.clip(scaled, 0.0, 1.0))


def _candidate_score(
    *,
    area: int,
    height: int,
    y_coverage: float,
    elongation: float,
    fill_ratio: float,
    edge_support: float,
    stroke_width: float,
    config: ComponentExtractionConfig,
) -> float:
    area_score = _clip01(area / max(1.0, float(config.min_area * 5)))
    height_score = _clip01(height / max(1.0, float(config.min_height * 5)))
    coverage_score = _clip01(y_coverage / max(1e-6, config.min_y_coverage * 3.0))
    elongation_score = _clip01(elongation / max(1e-6, config.min_elongation * 1.5))
    edge_score = _clip01(edge_support / max(1e-6, config.min_edge_support * 3.0))
    fill_score = _clip01(1.0 - max(0.0, fill_ratio - 0.50))
    stroke_score = _clip01(1.0 - stroke_width / max(1e-6, config.max_lane_stroke_width))
    return float(
        0.10 * area_score
        + 0.10 * height_score
        + 0.20 * coverage_score
        + 0.20 * elongation_score
        + 0.25 * edge_score
        + 0.05 * fill_score
        + 0.10 * stroke_score
    )


def _estimate_stroke_width(component_mask: np.ndarray) -> float:
    distance = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
    active_distances = distance[component_mask > 0]
    if active_distances.size == 0:
        return 0.0
    return float(2.0 * np.percentile(active_distances, 90))


def _min_area_rect_axes(contour: np.ndarray) -> tuple[float, float]:
    _center, (rect_width, rect_height), _angle = cv2.minAreaRect(contour)
    return (
        float(min(rect_width, rect_height)),
        float(max(rect_width, rect_height)),
    )


def _contour_elongation(contour: np.ndarray) -> float:
    points = contour.reshape(-1, 2).astype(np.float32)
    if len(points) < 3:
        return 1.0
    covariance = np.cov(points, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    minor = max(float(eigenvalues[0]), 1e-6)
    major = max(float(eigenvalues[-1]), minor)
    return float(np.sqrt(major / minor))


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _status_color(status: str) -> tuple[int, int, int]:
    if status == "strong":
        return (40, 220, 40)
    if status == "weak":
        return (0, 215, 255)
    return (30, 60, 230)


def _ensure_odd_kernel(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1
