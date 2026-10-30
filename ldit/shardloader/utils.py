"""
Utility functions and constants for shard dataloader.
"""

# =============================================================================
# Tumor Label Mapping
# =============================================================================

# Maps has_tumor values to class labels for conditional generation
# The highest index (num_classes) represents absence of conditioning for CFG

# For ImageNet+tumor extended training (num_classes=1002):
# Original ImageNet pretrained model has:
#   - Classes 0-999: ImageNet classes
#   - Class 1000: CFG unconditional token (LEARNED - preserve this!)
#
# Extended model (num_classes=1002):
#   - Classes 0-999: ImageNet classes (preserved from pretrained)
#   - Class 1000: CFG unconditional (PRESERVED from pretrained, reused for unknown/CFG)
#   - Class 1001: no_tumor (NEW - randomly initialized)
#   - Class 1002: has_tumor (NEW - randomly initialized)
IMAGENET_NUM_CLASSES = 1000
TUMOR_LABEL_MAPPING_EXTENDED = {
    -1: 1000,  # Unknown → reuse learned ImageNet CFG token
     0: 1001,  # No tumor confirmed (NEW class)
     1: 1002,  # Has tumor confirmed (NEW class)
}

# RadImageNet 5-region conditioning, appended ON TOP of the ImageNet table
# (same philosophy as TUMOR_LABEL_MAPPING_EXTENDED): ImageNet 0-999 preserved,
# 1000 = learned ImageNet CFG/uncond token (reused), regions = NEW classes
# 1001..1005 (randomly initialised). num_classes = 1005. region_label is the
# 0-4 produced by the extractor (CT/abd, CT/lung, MR/hip, MR/knee, MR/shoulder).
REGION_LABEL_MAPPING_EXTENDED = {
    0: 1001,  # CT/abd
    1: 1002,  # CT/lung
    2: 1003,  # MR/hip
    3: 1004,  # MR/knee
    4: 1005,  # MR/shoulder
}

# =============================================================================
# ShardWindowSampler Constants
# =============================================================================

# Seed component scaling factors (ensures no collisions across epochs/windows)
SEED_SCALE = 1_000_000      # Base seed space
EPOCH_SCALE = 10_000        # Supports up to 10,000 epochs
WINDOW_SCALE = 100          # Supports up to 100 windows per epoch

# =============================================================================
# Helper Functions
# =============================================================================

