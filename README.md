# Tramit Sequence Prediction Model

## Overview
This notebook implements a deep learning model to predict the next likely "tramit" (transaction/procedure) in a sequence based on historical patterns. The model uses an LSTM architecture to learn patterns in sequences of tramits and can predict the most probable next tramits given a context window.

## Dependencies
```sh
pip install torch tqdm numpy pandas
```

## Data Structure
The model works with two main data sources:
- `accions.csv`: Contains transaction records with timestamps and session information
- `tramits.csv`: Contains tramit definitions/mappings

## Code Components

### 1. Data Preprocessing
- Loads and processes tramit data from CSV files
- Creates a consistent mapping between tramit IDs and numerical indices
- Sorts and groups transactions by session to create sequences
- Filters sequences to include only those with multiple tramits

### 2. Dataset Implementation (`TramitContextDataset`)
A custom PyTorch Dataset class that:
- Takes sequences of tramits and converts them into context-target pairs
- Handles padding for context windows
- Provides validation for the tramit ID mapping
- Converts tramit IDs to numerical indices for model processing

### 3. Model Architecture (`TramitPredictor`)
A neural network model implementing:
- Embedding layer for tramit representation
- Two-layer LSTM for sequence processing
- Fully connected layers for final prediction
- Dropout for regularization

Key parameters:
- `embedding_dim`: 32
- `hidden_dim`: 64
- `context_size`: 3 (default)
- Dropout rate: 0.2

### 4. Training Implementation
The training process includes:
- Dataset splitting (80% train, 20% validation)
- Progress tracking with tqdm
- Model checkpointing (saves best model based on validation loss)
- Performance metrics tracking (train loss, validation loss, accuracy)

Training parameters:
- Batch size: 32
- Number of epochs: 50
- Learning rate: 0.001
- Optimizer: Adam

### 5. Prediction Function
The `predict_next_tramit` function:
- Takes a context sequence
- Returns top 5 predictions with probabilities
- Handles device (CPU/GPU) appropriately
- Converts numerical predictions back to tramit IDs

## Usage Example

```python
# Initialize and train the model
model, train_losses, val_losses = train_tramit_predictor(
    sequences, 
    context_size=3, 
    batch_size=32, 
    num_epochs=50
)

# Make predictions
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
context = [1, 2, 3]  # Example context (encoded tramit IDs)
predictions = predict_next_tramit(model, context, device)

# View predictions
for tramit, probability in predictions:
    print(f"Tramit: {tramit}, Probability: {probability:.4f}")
```

## Model Performance
The model's performance can be monitored through:
- Training loss
- Validation loss
- Validation accuracy (shown in training progress)

The example output shows prediction probabilities for the top 5 most likely next tramits, with confidence scores for each prediction.

## Output Files
The notebook generates two main output files:
- `output/mapping.json`: Contains the tramit ID to index mapping
- `output/model.pth`: Contains the saved model weights (best performing model based on validation loss)

## Notes
- The model assumes that tramit sequences have temporal significance
- The context size of 3 can be adjusted based on the specific use case
- The model uses base64-encoded tramit IDs for security/privacy
