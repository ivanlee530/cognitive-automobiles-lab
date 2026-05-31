#!/usr/bin/env python3
"""Stage 3 batch visual evaluation for lane-like connected components."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from stage3_bag_images import (
    DEFAULT_TOPIC,
    count_topic_frames,
    crop_road_roi,
    image_msg_to_bgr,
    iter_selected_images,
)
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
from stage3_contact_sheets import (
    build_sample_chunks,
    make_tile,
    save_contact_sheet_index,
    save_contact_sheets,
)


@dataclass(frozen=True)
class FrameComponentSummary:
    chunk_index: int
    sample_index: int
    frame_index: int
    timestamp_ns: int
    roi_y0: int
    image_width: int
    image_height: int
    roi_width: int
    roi_height: int
    total_components: int
    strong_components: int
    weak_components: int
    hard_reject_components: int
    usable_for_fitting_components: int
    strong_area_total: int
    weak_area_total: int
    lane_region_pixel_count: int
    component_source_pixel_count: int
    color_reject_pixel_count: int


@dataclass(frozen=True)
class CandidateDiagnostic:
    chunk_index: int
    sample_index: int
    frame_index: int
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


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag-dir", type=Path, default=project_dir.parent)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "outputs" / "stage3_batch_component",
    )
    parser.add_argument("--roi-top-ratio", type=float, default=0.30)
    parser.add_argument("--num-chunks", type=int, default=6)
    parser.add_argument("--samples-per-chunk", type=int, default=20)
    parser.add_argument("--contact-sheet-cols", type=int, default=5)
    parser.add_argument("--tile-width", type=int, default=540)

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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_frames = count_topic_frames(args.bag_dir, args.topic)
    chunks = build_sample_chunks(
        total_frames,
        num_chunks=args.num_chunks,
        samples_per_chunk=args.samples_per_chunk,
    )
    selected_indices = {
        frame_index
        for chunk in chunks
        for frame_index in chunk.selected_frame_indices
    }
    if not selected_indices:
        raise RuntimeError(f"No frames found on topic {args.topic!r} in {args.bag_dir}")

    preprocessing_config = ComponentPreprocessingConfig(
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
    chunk_by_frame = {
        frame_index: chunk.chunk_index
        for chunk in chunks
        for frame_index in chunk.selected_frame_indices
    }
    sample_position_by_frame = {
        frame_index: sample_index
        for chunk in chunks
        for sample_index, frame_index in enumerate(chunk.selected_frame_indices, start=1)
    }
    image_types = (
        "roi",
        "lane_region_mask",
        "edge_mask",
        "combined_binary_debug",
        "color_reject_mask",
        "component_source_mask",
        "strong_component_mask",
        "weak_component_mask",
        "fitting_support_mask",
        "hard_reject_mask",
        "candidate_status_overlay",
        "fitting_support_overlay",
        "hard_reject_overlay",
    )
    tiles = {
        image_type: {chunk.chunk_index: [] for chunk in chunks}
        for image_type in image_types
    }
    summaries: list[FrameComponentSummary] = []
    candidate_diagnostics: list[CandidateDiagnostic] = []

    for frame_index, timestamp_ns, msg in iter_selected_images(
        args.bag_dir,
        args.topic,
        selected_indices,
    ):
        image = image_msg_to_bgr(msg)
        roi, roi_y0 = crop_road_roi(image, args.roi_top_ratio)
        preprocessing = preprocess_component_roi(roi, preprocessing_config)
        candidates = extract_component_candidates(
            preprocessing.component_source_mask,
            preprocessing.edge_mask,
            extraction_config,
        )
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
        status_overlay = draw_candidate_status_overlay(roi, candidates)
        fitting_overlay = draw_mask_overlay(
            roi,
            fitting_support_mask,
            color=(40, 220, 40),
        )
        hard_reject_overlay = draw_hard_reject_overlay(
            roi,
            candidates,
        )

        chunk_index = chunk_by_frame[frame_index]
        sample_index = sample_position_by_frame[frame_index]
        strong = [candidate for candidate in candidates if candidate.status == "strong"]
        weak = [candidate for candidate in candidates if candidate.status == "weak"]
        hard_reject = [
            candidate for candidate in candidates if candidate.status == "reject"
        ]
        summaries.append(
            FrameComponentSummary(
                chunk_index=chunk_index,
                sample_index=sample_index,
                frame_index=frame_index,
                timestamp_ns=timestamp_ns,
                roi_y0=roi_y0,
                image_width=image.shape[1],
                image_height=image.shape[0],
                roi_width=roi.shape[1],
                roi_height=roi.shape[0],
                total_components=len(candidates),
                strong_components=len(strong),
                weak_components=len(weak),
                hard_reject_components=len(hard_reject),
                usable_for_fitting_components=len(strong) + len(weak),
                strong_area_total=sum(candidate.area for candidate in strong),
                weak_area_total=sum(candidate.area for candidate in weak),
                lane_region_pixel_count=cv2.countNonZero(
                    preprocessing.lane_region_mask
                ),
                component_source_pixel_count=cv2.countNonZero(
                    preprocessing.component_source_mask
                ),
                color_reject_pixel_count=cv2.countNonZero(
                    preprocessing.color_reject_mask
                ),
            )
        )
        candidate_diagnostics.extend(
            candidate_diagnostic_from_candidate(
                chunk_index,
                sample_index,
                frame_index,
                candidate,
            )
            for candidate in candidates
        )

        detail = (
            f"components={len(candidates)} strong={len(strong)} "
            f"weak={len(weak)} reject={len(hard_reject)}"
        )
        frame_images = {
            "roi": roi,
            "lane_region_mask": preprocessing.lane_region_mask,
            "edge_mask": preprocessing.edge_mask,
            "combined_binary_debug": preprocessing.combined_binary,
            "color_reject_mask": preprocessing.color_reject_mask,
            "component_source_mask": preprocessing.component_source_mask,
            "strong_component_mask": strong_component_mask,
            "weak_component_mask": weak_component_mask,
            "fitting_support_mask": fitting_support_mask,
            "hard_reject_mask": hard_reject_mask,
            "candidate_status_overlay": status_overlay,
            "fitting_support_overlay": fitting_overlay,
            "hard_reject_overlay": hard_reject_overlay,
        }
        save_frame_comparison_sheet(
            args.output_dir,
            frame_index,
            detail,
            frame_images,
        )
        for image_type, frame_image in frame_images.items():
            tiles[image_type][chunk_index].append(
                make_tile(frame_image, frame_index, args.tile_width, detail)
            )

        print(
            f"chunk={chunk_index} sample={sample_index:02d} frame={frame_index:04d} "
            f"components={len(candidates)} strong={len(strong)} "
            f"weak={len(weak)} reject={len(hard_reject)} "
            f"source_pixels={cv2.countNonZero(preprocessing.component_source_mask)}"
        )

    if len(summaries) != len(selected_indices):
        raise RuntimeError(
            f"Read {len(summaries)} sampled frames but expected {len(selected_indices)}"
        )

    sheet_records = save_contact_sheets(
        tiles,
        chunks,
        args.output_dir,
        cols=args.contact_sheet_cols,
    )
    summary_path = args.output_dir / "stage3_component_summary.csv"
    candidates_path = args.output_dir / "stage3_component_candidates.csv"
    rejection_summary_path = args.output_dir / "stage3_rejection_reason_summary.csv"
    weak_summary_path = args.output_dir / "stage3_weak_reason_summary.csv"
    index_path = args.output_dir / "contact_sheet_index.csv"
    save_dataclass_csv(summary_path, FrameComponentSummary, summaries)
    save_dataclass_csv(candidates_path, CandidateDiagnostic, candidate_diagnostics)
    save_hard_reject_reason_summary(rejection_summary_path, candidate_diagnostics)
    save_weak_reason_summary(weak_summary_path, candidate_diagnostics)
    save_contact_sheet_index(index_path, sheet_records)

    print(f"Available bag frames:       {total_frames}")
    print(f"Sampled frames:             {len(summaries)}")
    print(f"Saved contact sheets:       {len(sheet_records)}")
    print(f"Saved component summary:    {summary_path}")
    print(f"Saved component candidates: {candidates_path}")
    print(f"Saved rejection summary:    {rejection_summary_path}")
    print(f"Saved weak summary:         {weak_summary_path}")
    print(f"Saved sheet index:          {index_path}")
    print("Inspect candidate_status_overlay and fitting_support_overlay first.")
    print("Later fitting should consume fitting_support_mask or candidate weights.")
    print("Strong and weak candidates are usable; hard rejects are ignored.")


def candidate_diagnostic_from_candidate(
    chunk_index: int,
    sample_index: int,
    frame_index: int,
    candidate: ComponentCandidate,
) -> CandidateDiagnostic:
    return CandidateDiagnostic(
        chunk_index=chunk_index,
        sample_index=sample_index,
        frame_index=frame_index,
        component_id=candidate.component_id,
        status=candidate.status,
        usable_for_fitting=candidate.usable_for_fitting,
        candidate_weight=candidate.candidate_weight,
        score=candidate.score,
        hard_reject_reason=candidate.hard_reject_reason,
        weak_reason=candidate.weak_reason,
        area=candidate.area,
        area_ratio=candidate.area_ratio,
        x=candidate.x,
        y=candidate.y,
        width=candidate.width,
        height=candidate.height,
        centroid_x=candidate.centroid_x,
        centroid_y=candidate.centroid_y,
        y_coverage=candidate.y_coverage,
        horizontal_aspect_ratio=candidate.horizontal_aspect_ratio,
        elongation=candidate.elongation,
        fill_ratio=candidate.fill_ratio,
        edge_support=candidate.edge_support,
        stroke_width=candidate.stroke_width,
        minor_axis=candidate.minor_axis,
        major_axis=candidate.major_axis,
    )


def save_dataclass_csv(path: Path, row_type, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row_type.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def save_hard_reject_reason_summary(
    path: Path,
    rows: list[CandidateDiagnostic],
) -> None:
    counts = Counter(
        row.hard_reject_reason for row in rows if row.status == "reject"
    )
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["rejection_reason", "component_count"])
        for reason, count in sorted(counts.items()):
            writer.writerow([reason, count])


def save_weak_reason_summary(
    path: Path,
    rows: list[CandidateDiagnostic],
) -> None:
    counts = Counter(
        reason
        for row in rows
        if row.status == "weak"
        for reason in row.weak_reason.split("|")
        if reason
    )
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["weak_reason", "component_count"])
        for reason, count in sorted(counts.items()):
            writer.writerow([reason, count])


def save_frame_comparison_sheet(
    output_dir: Path,
    frame_index: int,
    detail: str,
    frame_images: dict[str, np.ndarray],
) -> None:
    panel_names = (
        "roi",
        "lane_region_mask",
        "edge_mask",
        "combined_binary_debug",
        "color_reject_mask",
        "component_source_mask",
        "strong_component_mask",
        "weak_component_mask",
        "fitting_support_mask",
        "hard_reject_mask",
        "candidate_status_overlay",
    )
    panels = [
        make_diagnostic_panel(name, frame_images[name], frame_index, detail)
        for name in panel_names
    ]
    sheet = make_panel_sheet(panels, cols=2)
    comparison_dir = output_dir / "frame_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    path = comparison_dir / f"frame_{frame_index:04d}_component_diagnostic.jpg"
    cv2.imwrite(str(path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def make_diagnostic_panel(
    title: str,
    image: np.ndarray,
    frame_index: int,
    detail: str,
    width: int = 640,
) -> np.ndarray:
    if image.ndim == 2:
        panel = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        panel = image.copy()

    scale = width / panel.shape[1]
    height = max(1, int(round(panel.shape[0] * scale)))
    panel = cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, 46), (0, 0, 0), -1)
    cv2.putText(
        panel,
        f"frame {frame_index:04d} | {title}",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        detail,
        (8, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (210, 255, 210),
        1,
        cv2.LINE_AA,
    )
    return panel


def make_panel_sheet(panels: list[np.ndarray], cols: int) -> np.ndarray:
    cols = max(1, cols)
    rows = math.ceil(len(panels) / cols)
    height, width = panels[0].shape[:2]
    sheet = np.full((rows * height, cols * width, 3), 22, dtype=np.uint8)
    for index, panel in enumerate(panels):
        row = index // cols
        col = index % cols
        y0 = row * height
        x0 = col * width
        sheet[y0 : y0 + height, x0 : x0 + width] = panel
    return sheet


if __name__ == "__main__":
    main()
