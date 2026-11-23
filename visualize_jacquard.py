
import torch
import numpy as np
import matplotlib.pyplot as plt

from models.ggcnn import GGCNN
from utils.data.jacquard_data import JacquardDataset

# 1. Load Jacquard dataset
dataset = JacquardDataset(
    "data/jacquard/Samples",
    output_size=300,
    include_depth=True,
    include_rgb=False
)

x, labels, idx, rot, zoom = dataset[0]

# 2. Load pretrained GG-CNN
model = GGCNN()

weights = torch.load(
    "ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(weights)
model.eval()

# 3. Run inference
with torch.no_grad():
    pos, cos, sin, width = model(x.unsqueeze(0))

# 4. Extract prediction maps
quality = pos.squeeze().numpy()
cos_map = cos.squeeze().numpy()
sin_map = sin.squeeze().numpy()
width_map = width.squeeze().numpy()

# 5. Find highest-quality grasp
v, u = np.unravel_index(
    np.argmax(quality),
    quality.shape
)

angle = 0.5 * np.arctan2(
    sin_map[v, u],
    cos_map[v, u]
)

grasp_width = width_map[v, u] * 150.0

print("Predicted grasp:")
print(f"Position: ({u}, {v})")
print(f"Quality: {quality[v, u]:.4f}")
print(f"Angle: {np.degrees(angle):.2f} degrees")
print(f"Width: {grasp_width:.2f} pixels")


# 6. Visualize predictions
from matplotlib.patches import Polygon

depth = x.squeeze().numpy()

# Rectangle dimensions
grasp_height = 30.0
half_width = grasp_width / 2
half_height = grasp_height / 2

# Rectangle corners, centered at the origin
corners = np.array([
    [-half_width, -half_height],
    [ half_width, -half_height],
    [ half_width,  half_height],
    [-half_width,  half_height]
])

# Rotate rectangle by the predicted grasp angle
rotation_matrix = np.array([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle),  np.cos(angle)]
])

rotated_corners = corners @ rotation_matrix.T

# Translate rectangle to predicted grasp center
rotated_corners += np.array([u, v])

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Depth image
axes[0].imshow(depth, cmap="gray")
axes[0].set_title("Jacquard Depth Image")

# Grasp quality
axes[1].imshow(quality, cmap="jet")
axes[1].scatter(u, v, c="white", s=40)
axes[1].set_title("Grasp Quality")

# Predicted grasp rectangle
axes[2].imshow(depth, cmap="gray")

rectangle = Polygon(
    rotated_corners,
    closed=True,
    fill=False,
    edgecolor="lime",
    linewidth=2
)

axes[2].add_patch(rectangle)

# Highlight the two opposing gripper edges
for i, j in [(0, 3), (1, 2)]:
    axes[2].plot(
        rotated_corners[[i, j], 0],
        rotated_corners[[i, j], 1],
        color="red",
        linewidth=4
    )

# Mark grasp center
axes[2].scatter(u, v, c="yellow", s=50)

axes[2].set_title("Predicted Grasp Rectangle")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

# Retrieve ground-truth grasp rectangles
gt_rectangles = dataset.get_gtbb(idx, rot, zoom)

print(f"Ground-truth rectangles: {len(gt_rectangles.grs)}")

# Draw the first 10 ground-truth rectangles
for gt in gt_rectangles.grs[:127]:
    gt_points = gt.points

    gt_polygon = Polygon(
        gt_points[:, ::-1],
        closed=True,
        fill=False,
        edgecolor="cyan",
        linewidth=1,
        alpha=0.6
    )

    axes[2].add_patch(gt_polygon)

axes[2].set_title("Prediction vs. Ground Truth")

plt.show()
