"""Ingestion subsystem for images, multi-view collections, video streams, and RGB-D."""

from .metadata_extractor import MetadataExtractor
from .image_loader import ImageLoader
from .video_processor import VideoProcessor
from .rgbd_loader import RGBDLoader

__all__ = [
    "MetadataExtractor",
    "ImageLoader",
    "VideoProcessor",
    "RGBDLoader",
]
