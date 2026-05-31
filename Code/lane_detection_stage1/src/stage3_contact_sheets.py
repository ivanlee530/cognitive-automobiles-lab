"""Contact sheet helpers used by the Stage 3 component batch evaluator."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class SampleChunk:
    chunk_index: int
    frame_start: int
    frame_end: int
    selected_frame_indices: tuple[int, ...]


@dataclass(frozen=True)
class ContactSheetRecord:
    image_type: str
    chunk_index: int
    frame_start: int
    frame_end: int
    selected_frame_count: int
    path: Path


def build_sample_chunks(
    total_frames: int,
    num_chunks: int = 6,
    samples_per_chunk: int = 20,
) -> list[SampleChunk]:
    if total_frames <= 0:
        return []

    chunks = []
    for chunk_index, chunk_indices in enumerate(
        np.array_split(np.arange(total_frames), max(1, num_chunks)),
        start=1,
    ):
        if chunk_indices.size == 0:
            continue

        sample_count = min(max(1, samples_per_chunk), int(chunk_indices.size))
        selected_positions = np.linspace(0, chunk_indices.size - 1, sample_count)
        selected = tuple(
            int(chunk_indices[int(round(position))]) for position in selected_positions
        )
        chunks.append(
            SampleChunk(
                chunk_index=chunk_index,
                frame_start=int(chunk_indices[0]),
                frame_end=int(chunk_indices[-1]),
                selected_frame_indices=selected,
            )
        )
    return chunks


def make_tile(
    image: np.ndarray,
    frame_index: int,
    width: int,
    detail: str = "",
) -> np.ndarray:
    if image.ndim == 2:
        tile = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        tile = image.copy()

    width = max(1, width)
    scale = width / tile.shape[1]
    height = max(1, int(round(tile.shape[0] * scale)))
    tile = cv2.resize(tile, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, 44), (0, 0, 0), -1)
    cv2.putText(
        tile,
        f"frame {frame_index:04d}",
        (8, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if detail:
        cv2.putText(
            tile,
            detail,
            (8, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (210, 255, 210),
            1,
            cv2.LINE_AA,
        )
    return tile


def save_contact_sheets(
    tiles: dict[str, dict[int, list[np.ndarray]]],
    chunks: list[SampleChunk],
    output_dir: Path,
    cols: int,
) -> list[ContactSheetRecord]:
    records = []
    for image_type, chunks_tiles in tiles.items():
        type_dir = output_dir / image_type
        type_dir.mkdir(parents=True, exist_ok=True)
        for chunk in chunks:
            selected_tiles = chunks_tiles[chunk.chunk_index]
            sheet = make_contact_sheet(selected_tiles, cols=cols)
            path = type_dir / (
                f"{image_type}_chunk_{chunk.chunk_index:02d}_frames_"
                f"{chunk.frame_start:04d}_{chunk.frame_end:04d}.jpg"
            )
            cv2.imwrite(str(path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            records.append(
                ContactSheetRecord(
                    image_type=image_type,
                    chunk_index=chunk.chunk_index,
                    frame_start=chunk.frame_start,
                    frame_end=chunk.frame_end,
                    selected_frame_count=len(selected_tiles),
                    path=path,
                )
            )
    return records


def make_contact_sheet(tiles: list[np.ndarray], cols: int = 5) -> np.ndarray:
    if not tiles:
        raise ValueError("Cannot build a contact sheet without tiles.")

    cols = max(1, cols)
    rows = math.ceil(len(tiles) / cols)
    tile_height, tile_width = tiles[0].shape[:2]
    sheet = np.full((rows * tile_height, cols * tile_width, 3), 22, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row = index // cols
        col = index % cols
        y0 = row * tile_height
        x0 = col * tile_width
        sheet[y0 : y0 + tile_height, x0 : x0 + tile_width] = tile
    return sheet


def save_contact_sheet_index(path: Path, records: list[ContactSheetRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "image_type",
                "chunk_index",
                "frame_start",
                "frame_end",
                "selected_frame_count",
                "path",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.image_type,
                    record.chunk_index,
                    record.frame_start,
                    record.frame_end,
                    record.selected_frame_count,
                    record.path,
                ]
            )
