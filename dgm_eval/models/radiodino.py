import torchvision.transforms as TF
import numpy as np
import torch
import timm
import sys

from .encoder import Encoder
from ..resizer import pil_resize


class RadioDinoEncoder(Encoder):
    """
    RadioDino encoder wrapper for dgm-eval.

    RadioDino is a DINOv2-based model fine-tuned on medical imaging data.
    Uses timm to load from Hugging Face Hub: hf_hub:Snarcy/RadioDino-b16
    """

    def setup(self, clean_resize: bool = False):
        """
        Load RadioDino model from Hugging Face Hub via timm.

        Args:
            clean_resize: Use PIL resize instead of torchvision (default: False)
        """
        try:
            self.model = timm.create_model("hf_hub:Snarcy/RadioDino-b16", pretrained=True)
            self.model.eval()
            self.clean_resize = clean_resize

            print(f"RadioDino encoder loaded from hf_hub:Snarcy/RadioDino-b16", file=sys.stderr)
        except Exception as e:
            raise RuntimeError("Failed to load RadioDino model") from e

    def transform(self, image):
        """
        Transform PIL image to RadioDino input format.

        Args:
            image: PIL Image

        Returns:
            Normalized tensor [3, 224, 224]
        """
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])

        if self.clean_resize:
            image = pil_resize(image, (224, 224))
        else:
            image = TF.Compose([
                TF.Resize((224, 224), TF.InterpolationMode.BICUBIC),
                TF.ToTensor(),
            ])(image)

        return TF.Normalize(imagenet_mean, imagenet_std)(image)
