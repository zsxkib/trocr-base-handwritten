# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import subprocess
import time
from cog import BasePredictor, Input, Path
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

TROCR_WEIGHTS_CACHE = "./weights"
TROCR_WEIGHTS_URL = "https://weights.replicate.delivery/default/TrOCR/weights.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        if not os.path.exists(TROCR_WEIGHTS_CACHE):
            download_weights(TROCR_WEIGHTS_URL, TROCR_WEIGHTS_CACHE)
            
        self.processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-handwritten",
            cache_dir=TROCR_WEIGHTS_CACHE,
            local_files_only=True,
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten",
            cache_dir=TROCR_WEIGHTS_CACHE,
            local_files_only=True,
        )

    def predict(self, image: Path = Input(description="Input image")) -> str:
        """Run a single prediction on the model"""
        image = Image.open(str(image)).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return generated_text
