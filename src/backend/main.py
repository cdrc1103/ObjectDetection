import os
from pathlib import Path
from uuid import uuid4

import cv2
import uvicorn
from fastapi import FastAPI, File, UploadFile

from src.backend.inference import (
    detect_and_draw_box,
    process_image_byte_stream,
)

HOST = "0.0.0.0" if os.getenv("DOCKER-SETUP") else "127.0.0.1"
PORT = 8000

app = FastAPI(title="Object Detection")


@app.post("/predict")
def prediction(model: str, file: UploadFile = File(...)) -> dict[str, str]:
    """
    This endpoint handles all the logic necessary for the object detection to work. It requires the desired model
    and the image in which to perform object detection.
    """
    input_image = process_image_byte_stream(file)
    output_image = detect_and_draw_box(input_image, model=model)
    image_path = str(
        Path(__file__).parents[2] / "images_with_boxes" / f"{uuid4()}.jpg"
    )
    cv2.imwrite(image_path, output_image)
    return {"name": image_path}


if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT)
