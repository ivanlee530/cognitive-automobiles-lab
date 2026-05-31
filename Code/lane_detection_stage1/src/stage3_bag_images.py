"""ROS2 bag image helpers used by the Stage 3 component pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image


DEFAULT_TOPIC = "/camera_front/color/image_raw"


def image_msg_to_bgr(msg: Image) -> np.ndarray:
    """Convert common ROS Image encodings to a BGR image for OpenCV."""
    encoding = msg.encoding.lower()

    if encoding in {"rgb8", "bgr8"}:
        channels = 3
        row_width = msg.width * channels
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        rows = raw.reshape((msg.height, msg.step))[:, :row_width]
        image = rows.reshape((msg.height, msg.width, channels))
        if encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image.copy()

    if encoding == "mono8":
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        rows = raw.reshape((msg.height, msg.step))[:, : msg.width]
        return cv2.cvtColor(rows, cv2.COLOR_GRAY2BGR)

    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def open_bag_reader(
    bag_dir: Path,
    storage_id: str = "mcap",
) -> rosbag2_py.SequentialReader:
    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_dir),
        storage_id=storage_id,
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return reader


def count_topic_frames(bag_dir: Path, topic: str) -> int:
    """Count topic frames without deserializing their payloads."""
    reader = open_bag_reader(bag_dir)
    frame_count = 0
    while reader.has_next():
        current_topic, _data, _timestamp_ns = reader.read_next()
        if current_topic == topic:
            frame_count += 1
    return frame_count


def iter_selected_images(
    bag_dir: Path,
    topic: str,
    selected_frame_indices: Iterable[int],
) -> Iterator[tuple[int, int, Image]]:
    """Yield requested topic frames while scanning the bag chronologically."""
    selected = set(selected_frame_indices)
    if not selected:
        return

    reader = open_bag_reader(bag_dir)
    frame_index = 0
    last_selected_index = max(selected)
    while reader.has_next():
        current_topic, data, timestamp_ns = reader.read_next()
        if current_topic != topic:
            continue

        if frame_index in selected:
            yield frame_index, timestamp_ns, deserialize_message(data, Image)
        if frame_index >= last_selected_index:
            break
        frame_index += 1


def read_first_image(bag_dir: Path, topic: str) -> tuple[Image, int]:
    reader = open_bag_reader(bag_dir)
    frame_index = 0
    while reader.has_next():
        current_topic, data, _timestamp = reader.read_next()
        if current_topic != topic:
            continue
        return deserialize_message(data, Image), frame_index
    raise RuntimeError(f"No messages found on topic {topic!r} in {bag_dir}")


def crop_road_roi(image: np.ndarray, roi_top_ratio: float) -> tuple[np.ndarray, int]:
    height = image.shape[0]
    roi_y0 = int(height * roi_top_ratio)
    roi_y0 = max(0, min(roi_y0, height - 1))
    return image[roi_y0:, :].copy(), roi_y0
