"""ShelfMatch core pipeline — detection, segmentation, and matching."""

from shelfmatch.pipeline.shelfmatcher import ShelfMatcher, PipelineConfig, ShelfMatchResult
from shelfmatch.pipeline.detector import (
    DetectionResult,
    DetectionOutput,
    GroundingDINODetector,
    OWLv2Detector,
    YOLOWorldDetector,
)
from shelfmatch.pipeline.segmenter import SAM2Segmenter, MaskResult, SegmentationOutput
from shelfmatch.pipeline.matcher import (
    ReferenceProduct,
    MatchResult,
    ProductMatcher,
    FeatureExtractor,
)

__all__ = [
    # Core
    "ShelfMatcher",
    "PipelineConfig",
    "ShelfMatchResult",
    # Detector
    "GroundingDINODetector",
    "OWLv2Detector",
    "YOLOWorldDetector",
    "DetectionResult",
    "DetectionOutput",
    # Segmenter
    "SAM2Segmenter",
    "MaskResult",
    "SegmentationOutput",
    # Matcher
    "FeatureExtractor",
    "ReferenceProduct",
    "MatchResult",
    "ProductMatcher",
]
