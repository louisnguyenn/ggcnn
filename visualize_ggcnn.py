
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.ggcnn import GGCNN

model = GGCNN()
model.load_state_dict(torch.load(
    "ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt",
    map_location="cpu",
    weights_only=True
))
model.eval()

# Synthetic depth image: raised rectangular object
depth = np.zeros((300, 300), dtype=np.float32)
depth[100:200, 110:190] = -0.05

# Run inference
x = torch.from_numpy(depth)[None, None]

with torch.no_grad():
    quality, cos, sin, width = model(x)

quality = quality.squeeze().numpy()
angle = 0.5 * np.arctan2(
    sin.squeeze().numpy(),
    cos.squeeze().numpy()
)
width = width.squeeze().numpy() * 150

# Find the highest-quality grasp location
row, col = np.unravel_index(
    np.argmax(quality),
    quality.shape
)

best_quality = quality[row, col]
best_angle = angle[row, col]
best_width = width[row, col]

print("\nBest Grasp Prediction:")
print(f"Position (u, v): ({col}, {row})")
print(f"Quality: {best_quality:.4f}")
print(f"Angle: {np.degrees(best_angle):.2f} degrees")
print(f"Width: {best_width:.2f} pixels")


# Visualize
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

images = [depth, quality, angle, width]
titles = [
    "Synthetic Depth",
    "Grasp Quality",
    "Grasp Angle",
    "Grasp Width (pixels)"
]

for ax, img, title in zip(axes, images, titles):
    im = ax.imshow(img, cmap="viridis")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

# Mark the highest-quality grasp on the depth image
axes[0].plot(col, row, "r+", markersize=14,
             markeredgewidth=2)
axes[0].set_title("Depth + Best Grasp Center")

plt.tight_layout()
plt.savefig("ggcnn_predictions.png", dpi=150)
plt.show()
