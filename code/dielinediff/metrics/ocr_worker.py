"""Run PaddleOCR in a separate CPU Python process."""

import base64
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

MODELS = {
    "text_detection": "PP-OCRv5_mobile_det",
    "text_recognition": "PP-OCRv5_mobile_rec",
    "textline_orientation": "PP-LCNet_x1_0_textline_ori",
}


def model_options():
    cache = Path(os.environ["PADDLE_PDX_CACHE_HOME"]) / "official_models"
    options = {}
    required = ("inference.json", "inference.pdiparams", "inference.yml")
    for field, name in MODELS.items():
        options[f"{field}_model_name"] = name
        for directory in (cache / name, cache / f"{name}_infer"):
            if all((directory / file).is_file() for file in required):
                options[f"{field}_model_dir"] = str(directory)
                break
    return options


def main(threshold):
    with redirect_stdout(sys.stderr):
        ocr = PaddleOCR(
            device="cpu", use_doc_orientation_classify=False, use_doc_unwarping=False,
            use_textline_orientation=True, enable_mkldnn=False, cpu_threads=2,
            **model_options(),
        )
    for line in sys.stdin:
        source = Image.open(io.BytesIO(base64.b64decode(line))).convert("RGB")
        bgr = np.asarray(source)[..., ::-1].copy()
        with redirect_stdout(sys.stderr):
            result = ocr.predict(bgr, text_rec_score_thresh=0.0)[0]
        texts = result["rec_texts"]
        angles = result.get("textline_orientation_angles", [-1] * len(texts))
        payload = []
        for text, confidence, angle in zip(texts, result["rec_scores"], angles, strict=True):
            if text.strip() and confidence >= threshold:
                payload.append([text, {0: 0, 1: 180, 180: 180}.get(int(angle), -1)])
        print("E01_OCR:" + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main(float(sys.argv[1]))
