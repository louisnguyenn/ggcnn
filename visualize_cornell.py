
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from models.ggcnn import GGCNN
from utils.data.cornell_data import CornellDataset
from utils.dataset_processing import grasp

# Configuration
DATASET_PATH = "data/cornell"
WEIGHTS_PATH = (
    "ggcnn_weights_cornell/"
    "ggcnn_epoch_23_cornell_statedict.pt"
)

# 1. Load Cornell dataset
dataset = CornellDataset(
    DATASET_PATH,
    output_size=300,
    include_depth=True,
    include_rgb=False
)

print(f"Dataset contains {len(dataset)} samples")

# 2. Load pretrained model
model = GGCNN()

model.load_state_dict(
    torch.load(
        WEIGHTS_PATH,
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

# 3. Select the first sample
x, labels, idx, rot, zoom = dataset[0]

# Add batch dimension
input_tensor = x.unsqueeze(0)

# 4. Run inference
with torch.no_grad():
    quality, cos, sin, width = model(input_tensor)

# 5. Convert outputs to NumPy
quality = quality.squeeze().cpu().numpy()

angle = 0.5 * np.arctan2(
    sin.squeeze().cpu().numpy(),
    cos.squeeze().cpu().numpy()
)

width = width.squeeze().cpu().numpy() * 150

# 6. Find highest-quality grasp
row, col = np.unravel_index(
    np.argmax(quality),
    quality.shape
)

best_quality = quality[row, col]
best_angle = angle[row, col]
best_width = width[row, col]

print("\nBest Cornell Grasp Prediction:")
print(f"Position (u, v): ({col}, {row})")
print(f"Quality: {best_quality:.4f}")
print(f"Angle: {np.degrees(best_angle):.2f} degrees")
print(f"Width: {best_width:.2f} pixels")

# 7. Display depth and quality
depth = input_tensor.squeeze().cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(depth, cmap="viridis")
axes[0].plot(col, row, "r+", markersize=14)
axes[0].set_title("Cornell Depth + Grasp Center")

im = axes[1].imshow(quality, cmap="viridis")
axes[1].plot(col, row, "r+", markersize=14)
axes[1].set_title("GG-CNN Grasp Quality")

plt.colorbar(im, ax=axes[1])
plt.tight_layout()
plt.savefig("cornell_prediction.png", dpi=150)
plt.show()
