#!/usr/bin/env python3
"""Stage 3: extract lane-like connected components from the road ROI."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from stage3_bag_images import DEFAULT_TOPIC, crop_road_roi, image_msg_to_bgr, read_first_image
from stage3_component_extractor import (
    ComponentCandidate,
    ComponentExtractionConfig,
    draw_candidate_status_overlay,
    draw_hard_reject_overlay,
    draw_mask_overlay,
    extract_component_candidates,
    make_fitting_support_mask,
    make_hard_reject_mask,
    make_strong_component_mask,
    make_weak_component_mask,
)
from stage3_component_preprocessing import (
    ComponentPreprocessingConfig,
    preprocess_component_roi,
)


def save_candidates_csv(path: Path, candidates: list[ComponentCandidate]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            name
            for name in ComponentCandidate.__dataclass_fields__
            if name != "contour"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow({name: getattr(candidate, name) for name in fieldnames})


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag-dir", type=Path, default=project_dir.parent)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--output-dir", type=Path, default=project_dir / "outputs")
    parser.add_argument("--roi-top-ratio", type=float, default=0.30)
    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=150)
    parser.add_argument("--sobel-threshold", type=int, default=40)
    parser.add_argument("--white-luminance-threshold", type=int, default=185)
    parser.add_argument("--white-saturation-max", type=int, default=80)
    parser.add_argument("--component-source-close-kernel", type=int, default=5)
    parser.add_argument("--component-source-close-iterations", type=int, default=1)
    parser.add_argument("--component-source-dilate-kernel", type=int, default=3)
    parser.add_argument("--component-source-dilate-iterations", type=int, default=1)
    parser.add_argument(
        "--enable-color-suppression",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--orange-red-saturation-min", type=int, default=110)
    parser.add_argument("--orange-red-value-min", type=int, default=100)
    parser.add_argument("--color-suppression-dilate-kernel", type=int, default=9)
    parser.add_argument("--min-component-area", type=int, default=6)
    parser.add_argument("--min-component-height", type=int, default=2)
    parser.add_argument("--min-y-coverage", type=float, default=0.025)
    parser.add_argument("--min-elongation", type=float, default=1.8)
    parser.add_argument("--max-horizontal-aspect-ratio", type=float, default=3.0)
    parser.add_argument("--long-lane-y-coverage", type=float, default=0.25)
    parser.add_argument("--compact-lane-y-coverage", type=float, default=0.50)
    parser.add_argument("--min-edge-support", type=float, default=0.04)
    parser.add_argument("--large-blob-min-area", type=int, default=900)
    parser.add_argument("--max-lane-fill-ratio", type=float, default=0.78)
    parser.add_argument("--max-lane-stroke-width", type=float, default=18.0)
    parser.add_argument("--max-component-area-ratio", type=float, default=0.03)
    parser.add_argument("--weak-min-score", type=float, default=0.30)
    parser.add_argument("--strong-min-score", type=float, default=0.68)
    parser.add_argument("--accepted-close-kernel", type=int, default=3)
    parser.add_argument("--accepted-dilate-kernel", type=int, default=3)
    parser.add_argument("--accepted-dilate-iterations", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    msg, frame_index = read_first_image(args.bag_dir, args.topic)
    image = image_msg_to_bgr(msg)
    roi, roi_y0 = crop_road_roi(image, args.roi_top_ratio)
    preprocessing = preprocess_component_roi(
        roi,
        ComponentPreprocessingConfig(
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            sobel_threshold=args.sobel_threshold,
            white_luminance_threshold=args.white_luminance_threshold,
            white_saturation_max=args.white_saturation_max,
            component_source_close_kernel=args.component_source_close_kernel,
            component_source_close_iterations=args.component_source_close_iterations,
            component_source_dilate_kernel=args.component_source_dilate_kernel,
            component_source_dilate_iterations=args.component_source_dilate_iterations,
            enable_color_suppression=args.enable_color_suppression,
            orange_red_saturation_min=args.orange_red_saturation_min,
            orange_red_value_min=args.orange_red_value_min,
            color_suppression_dilate_kernel=args.color_suppression_dilate_kernel,
        ),
    )
    extraction_config = ComponentExtractionConfig(
        min_area=args.min_component_area,
        min_height=args.min_component_height,
        min_y_coverage=args.min_y_coverage,
        min_elongation=args.min_elongation,
        max_horizontal_aspect_ratio=args.max_horizontal_aspect_ratio,
        long_lane_y_coverage=args.long_lane_y_coverage,
        compact_lane_y_coverage=args.compact_lane_y_coverage,
        min_edge_support=args.min_edge_support,
        large_blob_min_area=args.large_blob_min_area,
        max_lane_fill_ratio=args.max_lane_fill_ratio,
        max_lane_stroke_width=args.max_lane_stroke_width,
        max_component_area_ratio=args.max_component_area_ratio,
        weak_min_score=args.weak_min_score,
        strong_min_score=args.strong_min_score,
        accepted_close_kernel=args.accepted_close_kernel,
        accepted_dilate_kernel=args.accepted_dilate_kernel,
        accepted_dilate_iterations=args.accepted_dilate_iterations,
    )
    candidates = extract_component_candidates(
        preprocessing.component_source_mask,
        preprocessing.edge_mask,
        extraction_config,
    )

    output_dir = args.output_dir / "stage3_component_extract"
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"stage3_frame_{frame_index:03d}"
    strong_component_mask = make_strong_component_mask(
        preprocessing.component_source_mask.shape,
        candidates,
    )
    weak_component_mask = make_weak_component_mask(
        preprocessing.component_source_mask.shape,
        candidates,
    )
    fitting_support_mask = make_fitting_support_mask(
        preprocessing.component_source_mask.shape,
        candidates,
        extraction_config,
    )
    hard_reject_mask = make_hard_reject_mask(
        preprocessing.component_source_mask.shape,
        candidates,
    )
    outputs = {
        "lane_region_mask": preprocessing.lane_region_mask,
        "edge_mask": preprocessing.edge_mask,
        "combined_binary_debug": preprocessing.combined_binary,
        "color_reject_mask": preprocessing.color_reject_mask,
        "component_source_mask": preprocessing.component_source_mask,
        "strong_component_mask": strong_component_mask,
        "weak_component_mask": weak_component_mask,
        "fitting_support_mask": fitting_support_mask,
        "hard_reject_mask": hard_reject_mask,
        "candidate_status_overlay": draw_candidate_status_overlay(roi, candidates),
        "fitting_support_overlay": draw_mask_overlay(
            roi,
            fitting_support_mask,
            color=(40, 220, 40),
        ),
        "hard_reject_overlay": draw_hard_reject_overlay(
            roi,
            candidates,
        ),
    }
    for name, output in outputs.items():
        path = Path(f"{prefix}_{name}.png")
        cv2.imwrite(str(path), output)
        print(f"Saved {name:28s}: {path}")

    csv_path = Path(f"{prefix}_component_candidates.csv")
    save_candidates_csv(csv_path, candidates)
    print(f"Saved {'component diagnostics':28s}: {csv_path}")
    print(f"ROI y0:                       {roi_y0}")
    print(f"Components inspected:         {len(candidates)}")
    print(f"Strong candidates:            {sum(item.status == 'strong' for item in candidates)}")
    print(f"Weak candidates:              {sum(item.status == 'weak' for item in candidates)}")
    print(f"Hard rejects:                 {sum(item.status == 'reject' for item in candidates)}")
    print("Later fitting should use fitting_support_mask or candidate weights.")


if __name__ == "__main__":
    main()
