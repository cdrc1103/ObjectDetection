import io

import cv2
import cvlib as cv
import numpy as np
from cvlib.object_detection import draw_bbox
from fastapi import UploadFile


def process_image_byte_stream(file: UploadFile):
    """Read the byte stream and decode it"""
    # Read image as a stream of bytes
    image_stream = io.BytesIO(file.file.read())
    # Start the stream from the beginning (position zero)
    image_stream.seek(0)
    # Write the stream of bytes into a numpy array
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    # Decode the numpy array as an image
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def detect_and_draw_box(
    image, model: str = "yolov3-tiny", confidence: float = 0.5
):
    """Detects common objects on an image and creates a new image with bounding boxes.

    Args:
        image (str): Filename of the image.
        model (str): Either "yolov3" or "yolov3-tiny". Defaults to "yolov3-tiny".
        confidence (float, optional): Desired confidence level. Defaults to 0.5.
    """

    # Perform the object detection
    bbox, label, conf = cv.detect_common_objects(
        image, confidence=confidence, model=model
    )

    # Create a new image that includes the bounding boxes
    output_image = draw_bbox(image, bbox, label, conf)

    return output_image
