import numpy as np
import cv2
import json
import sys

if len(sys.argv) < 2:
    print("Usage: python preprocess.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Could not load image from {image_path}", file=sys.stderr)
    sys.exit(1)

img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
img = 255 - img
img = img.astype('float32') / 255.0
flattened_data = img.flatten().tolist()
payload = {"data": flattened_data}
print(json.dumps(payload))
