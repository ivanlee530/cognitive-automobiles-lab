FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-opencv \
    python3-numpy \
    ros-jazzy-cv-bridge \
    ros-jazzy-rosbag2 \
    ros-jazzy-rosbag2-py \
    ros-jazzy-rosbag2-storage-mcap \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
