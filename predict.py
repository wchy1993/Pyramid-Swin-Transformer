import torch
from datasets import get_dataloaders
from model import get_model
from Pyramid_Swin_Transformer import PyramidSwinTransformer
from config import valdir

# Configuration for loading pretrained weights
pretrained_weights_path = "weights_epoch_1.pth"  # TODO: Replace with your pretrained weights path

# Load data (only validation loader for prediction)
_, val_loader = get_dataloaders(traindir=None, valdir=valdir)

# Model initialization
model = get_model(PyramidSwinTransformer)
if torch.cuda.is_available():
    model = model.cuda()

# Load the pretrained weights
model.load_state_dict(torch.load(pretrained_weights_path))
model.eval()

# Prediction function
def predict():
    predictions = []
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(val_loader):  # We don't need labels for prediction
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            outputs = model(inputs)
            # Convert outputs to desired format for predictions (e.g., argmax for classification)
            predicted_labels = torch.argmax(outputs, dim=1)
            predictions.extend(predicted_labels.cpu().numpy())
    return predictions

predictions = predict()
print(predictions)
