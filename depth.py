import torch
import cv2
import numpy as np
import argparse

# -------------------------------
# Argument parsing
# -------------------------------
parser = argparse.ArgumentParser(description="Depth estimation using MiDaS")
parser.add_argument(
    "--image", "-i", type=str, required=True, help="Path to the input image"
)
args = parser.parse_args()
img_name = args.image

# -------------------------------
# Load MiDaS model
# -------------------------------
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

# -------------------------------
# Read and preprocess image
# -------------------------------
img = cv2.imread(img_name)
if img is None:
    raise FileNotFoundError(f"Image {img_name} not found")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_batch = transform(img_rgb).unsqueeze(0)  # Add batch dimension

# -------------------------------
# Predict depth
# -------------------------------
with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

# -------------------------------
# Normalize and save depth map
# -------------------------------
depth = prediction.cpu().numpy()
depth = (depth - depth.min()) / (depth.max() - depth.min())

output_path = img_name.replace(".png", "_depth.png")
cv2.imwrite(output_path, (depth * 255).astype(np.uint8))
print(f"Depth map saved to {output_path}")