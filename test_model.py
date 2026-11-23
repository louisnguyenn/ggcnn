
import torch
from models.ggcnn import GGCNN

# Load the GG-CNN architecture
model = GGCNN()

# Load pretrained weights
weights_path = (
    "ggcnn_weights_cornell/"
    "ggcnn_epoch_23_cornell_statedict.pt"
)

state_dict = torch.load(
    weights_path,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()

# Test with a synthetic depth image
depth = torch.zeros(1, 1, 300, 300)

with torch.no_grad():
    quality, cos, sin, width = model(depth)

print("Model loaded successfully!")
print("Quality:", quality.shape)
print("Cosine:", cos.shape)
print("Sine:", sin.shape)
print("Width:", width.shape)
