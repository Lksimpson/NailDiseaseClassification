# Model Artifacts Setup Guide

## Current Status

The application is now configured to automatically look for trained DenseNet model weights. If no trained weights are found, it will use a randomly initialized model (which won't work well for predictions).

## How to Use Trained Model Weights

### Option 1: Place Model in Default Location

After training your model in the notebook, save it and place it in one of these locations:

1. `models/densenet_nail_disease_best.pth` (recommended)
2. `models/densenet_nail_disease_weights.pth`
3. `models/densenet_model.pth`
4. `models/nail_disease_classifier.pth`

### Option 2: Use Environment Variable

Set the `DISEASE_MODEL_PATH` environment variable:

```bash
export DISEASE_MODEL_PATH=/path/to/your/model.pth
python app.py
```

### Option 3: Save Model from Notebook

Add this code to your training notebook after training:

```python
# Save model weights (state_dict only)
torch.save(model.state_dict(), 'models/densenet_nail_disease_weights.pth')

# OR save full checkpoint (includes optimizer state, epoch, etc.)
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
    # ... other info
}
torch.save(checkpoint, 'models/densenet_nail_disease_best.pth')
```

## Model Format Support

The application supports both formats:

1. **State Dict Only**: `torch.save(model.state_dict(), 'model.pth')`
2. **Checkpoint Format**: Dictionary with `'model_state_dict'` or `'state_dict'` key

## Verifying Model Loading

When you start the application, you should see one of these messages:

- ✅ `"Found model weights at: models/densenet_nail_disease_best.pth"`
- ✅ `"Loaded model weights from models/densenet_nail_disease_best.pth"`
- ⚠️ `"Warning: No model weights provided. Using randomly initialized model."`

If you see the warning, the model won't make accurate predictions until you provide trained weights.

## Training a Model

To train a model using the DenseNet architecture:

1. Open `densenet_test_disease_classification.ipynb`
2. Train the model on your dataset
3. Save the model using one of the methods above
4. Place it in the `models/` directory
5. Restart the Flask application

The application will automatically detect and load the trained weights.

