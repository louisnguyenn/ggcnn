
from utils.data.jacquard_data import JacquardDataset

DATASET_PATH = "data/jacquard/Samples"

dataset = JacquardDataset(
    DATASET_PATH,
    output_size=300,
    include_depth=True,
    include_rgb=False
)

print(f"Dataset contains {len(dataset)} samples")

x, labels, idx, rot, zoom = dataset[0]

print("\nJacquard sample loaded successfully!")
print("Input shape:", x.shape)
print("Input dtype:", x.dtype)
print("Sample index:", idx)
print("Rotation:", rot)
print("Zoom:", zoom)
