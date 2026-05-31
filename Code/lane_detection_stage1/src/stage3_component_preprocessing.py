"""Stage 3 preprocessing for connected-component lane extraction."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ComponentPreprocessingConfig:
    blur_kernel: int = 5
    canny_low: int = 50
    canny_high: int = 150
    sobel_kernel: int = 3
    sobel_threshold: int = 40
    white_luminance_threshold: int = 185
    white_saturation_max: int = 80
    lane_region_close_kernel: int = 5
    lane_region_dilate_kernel: int = 3
    lane_region_dilate_iterations: int = 1
    lane_region_open_kernel: int = 3
    component_source_close_kernel: int = 5
    component_source_close_iterations: int = 1
    component_source_dilate_kernel: int = 3
    component_source_dilate_iterations: int = 1
    enable_color_suppression: bool = True
    orange_red_saturation_min: int = 110
    orange_red_value_min: int = 100
    color_suppression_dilate_kernel: int = 9


@dataclass(frozen=True)
class ComponentPreprocessingResult:
    edge_mask: np.ndarray
    white_lane_mask: np.ndarray
    color_reject_mask: np.ndarray
    lane_region_mask: np.ndarray
    combined_binary: np.ndarray
    component_source_mask: np.ndarray


def preprocess_component_roi(
    roi_bgr: np.ndarray,
    config: ComponentPreprocessingConfig | None = None,
) -> ComponentPreprocessingResult:
    """Create Stage 3 masks with HSV saturation to preserve tinted white paint."""
    if config is None:
        config = ComponentPreprocessingConfig()

    blur_kernel = _ensure_odd_kernel(config.blur_kernel)
    sobel_kernel = _ensure_odd_kernel(config.sobel_kernel)
    close_kernel_size = _ensure_odd_kernel(config.lane_region_close_kernel)
    dilate_kernel_size = _ensure_odd_kernel(config.lane_region_dilate_kernel)
    open_kernel_size = _ensure_odd_kernel(config.lane_region_open_kernel)
    source_close_kernel_size = _ensure_odd_kernel(
        config.component_source_close_kernel
    )
    source_dilate_kernel_size = _ensure_odd_kernel(
        config.component_source_dilate_kernel
    )

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    canny = cv2.Canny(blur, config.canny_low, config.canny_high)

    sobel_raw = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_abs = np.absolute(sobel_raw)
    max_sobel = float(np.max(sobel_abs))
    if max_sobel > 0.0:
        sobel_x = np.uint8(255.0 * sobel_abs / max_sobel)
    else:
        sobel_x = np.zeros_like(gray, dtype=np.uint8)
    _, sobel_binary = cv2.threshold(
        sobel_x,
        config.sobel_threshold,
        255,
        cv2.THRESH_BINARY,
    )

    hls = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    white_luminance_mask = cv2.inRange(
        hls[:, :, 1],
        config.white_luminance_threshold,
        255,
    )
    low_saturation_mask = cv2.inRange(
        hsv[:, :, 1],
        0,
        config.white_saturation_max,
    )
    white_lane_mask = cv2.bitwise_and(white_luminance_mask, low_saturation_mask)
    color_reject_mask = _make_color_reject_mask(hsv, config)
    if config.enable_color_suppression:
        white_lane_mask = cv2.bitwise_and(
            white_lane_mask,
            cv2.bitwise_not(color_reject_mask),
        )

    close_kernel = np.ones((close_kernel_size, close_kernel_size), dtype=np.uint8)
    dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), dtype=np.uint8)
    open_kernel = np.ones((open_kernel_size, open_kernel_size), dtype=np.uint8)
    lane_region_mask = cv2.morphologyEx(
        white_lane_mask,
        cv2.MORPH_CLOSE,
        close_kernel,
    )
    lane_region_mask = cv2.dilate(
        lane_region_mask,
        dilate_kernel,
        iterations=max(0, config.lane_region_dilate_iterations),
    )
    lane_region_mask = cv2.morphologyEx(
        lane_region_mask,
        cv2.MORPH_OPEN,
        open_kernel,
    )

    edge_mask = cv2.bitwise_or(canny, sobel_binary)
    combined_binary = cv2.bitwise_and(edge_mask, lane_region_mask)
    component_source_mask = combined_binary
    if config.component_source_close_iterations > 0:
        source_close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (source_close_kernel_size, source_close_kernel_size),
        )
        component_source_mask = cv2.morphologyEx(
            component_source_mask,
            cv2.MORPH_CLOSE,
            source_close_kernel,
            iterations=config.component_source_close_iterations,
        )
    if config.component_source_dilate_iterations > 0:
        source_dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (source_dilate_kernel_size, source_dilate_kernel_size),
        )
        component_source_mask = cv2.dilate(
            component_source_mask,
            source_dilate_kernel,
            iterations=config.component_source_dilate_iterations,
        )
    component_source_mask = cv2.bitwise_and(
        component_source_mask,
        lane_region_mask,
    )
    return ComponentPreprocessingResult(
        edge_mask=edge_mask,
        white_lane_mask=white_lane_mask,
        color_reject_mask=color_reject_mask,
        lane_region_mask=lane_region_mask,
        combined_binary=combined_binary,
        component_source_mask=component_source_mask,
    )


def _make_color_reject_mask(
    hsv: np.ndarray,
    config: ComponentPreprocessingConfig,
) -> np.ndarray:
    if not config.enable_color_suppression:
        return np.zeros(hsv.shape[:2], dtype=np.uint8)

    saturation_min = max(0, min(255, int(config.orange_red_saturation_min)))
    value_min = max(0, min(255, int(config.orange_red_value_min)))
    orange_red = cv2.inRange(
        hsv,
        (0, saturation_min, value_min),
        (25, 255, 255),
    )
    wrapped_red = cv2.inRange(
        hsv,
        (170, saturation_min, value_min),
        (179, 255, 255),
    )
    color_reject_mask = cv2.bitwise_or(orange_red, wrapped_red)
    dilate_kernel_size = _ensure_odd_kernel(config.color_suppression_dilate_kernel)
    if dilate_kernel_size > 1:
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilate_kernel_size, dilate_kernel_size),
        )
        color_reject_mask = cv2.dilate(color_reject_mask, dilate_kernel)
    return color_reject_mask


def _ensure_odd_kernel(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1
