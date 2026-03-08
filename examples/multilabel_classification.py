import numpy as np
import torch

from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.dataset import TextClassificationDataset
from torchTextClassifiers.model import TextClassificationModel, TextClassificationModule
from torchTextClassifiers.model.components import (
    AttentionConfig,
    CategoricalVariableNet,
    ClassificationHead,
    TextEmbedder,
    TextEmbedderConfig,
)
from torchTextClassifiers.tokenizers import HuggingFaceTokenizer

# ==========================================
# 1. Ragged-lists approach
# ==========================================

# In multilabel classification, each instance can be assigned multiple labels simultaneously.
# Let's use fake data where labels is a list of lists (ragged array).
sample_text_data = [
    "This is a positive example",
    "This is a negative example",
    "Another positive case",
    "Another negative case",
    "Good example here",
    "Bad example here",
]

# Each inner list contains labels for the corresponding instance
labels_ragged = [[0, 1, 5], [0, 4], [1, 5], [0, 1, 4], [1, 5], [0]]

# Note: labels_ragged is a "jagged array." 
# np.array(labels_ragged) would not work directly as a standard numeric matrix.
# However, torchTextClassifiers handles this directly.

# Load a pre-trained tokenizer
tokenizer = HuggingFaceTokenizer.load_from_pretrained(
    "google-bert/bert-base-uncased", output_dim=126
)

X = np.array(sample_text_data)
Y_ragged = labels_ragged 

# Configure the model and training
# We use BCEWithLogitsLoss for multilabel tasks to treat each label 
# as a separate binary classification problem.
embedding_dim = 96
num_classes = max(max(label_list) for label_list in labels_ragged) + 1

model_config = ModelConfig(
    embedding_dim=embedding_dim,
    num_classes=num_classes,
)

training_config = TrainingConfig(
    lr=1e-3,
    batch_size=4,
    num_epochs=1,
    loss=torch.nn.BCEWithLogitsLoss(),  # Essential for multilabel
)

# Initialize the classifier with ragged_multilabel=True
ttc_ragged = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config,
    ragged_multilabel=True,  # Key for ragged list input!
)

print("Starting training with ragged labels...")
ttc_ragged.train(
    X_train=X,
    y_train=Y_ragged,
    training_config=training_config,
)

# Behind the scenes, the ragged lists are converted into a binary matrix (one-hot version).

# ==========================================
# 2. One-hot / multidimensional output approach
# ==========================================

# You can also provide a one-hot/multidimensional array (or float probabilities).
# Here, each row is a vector of size equal to the number of labels.
labels_one_hot = [
    [1., 1., 0., 0., 0., 1.],
    [1., 0., 0., 0., 1., 0.],
    [0., 1., 0., 0., 0., 1.],
    [1., 1., 0., 0., 1., 0.],
    [0., 1., 0., 0., 0., 1.],
    [1., 0., 0., 0., 1., 0.]
]
Y_one_hot = np.array(labels_one_hot)

# When using one-hot/dense arrays, set ragged_multilabel=False (default)
ttc_dense = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config,
)

print("\nStarting training with one-hot labels...")
ttc_dense.train(
    X_train=X,
    y_train=Y_one_hot,
    training_config=training_config,
)
