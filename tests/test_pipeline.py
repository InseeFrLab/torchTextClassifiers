import numpy as np
import pytest
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
from torchTextClassifiers.tokenizers import HuggingFaceTokenizer, NGramTokenizer, WordPieceTokenizer
from torchTextClassifiers.utilities.plot_explainability import (
    map_attributions_to_char,
    map_attributions_to_word,
    plot_attributions_at_char,
    plot_attributions_at_word,
)


@pytest.fixture
def sample_data():
    """Fixture providing sample data for all tests."""
    sample_text_data = [
        "This is a positive example",
        "This is a negative example",
        "Another positive case",
        "Another negative case",
        "Good example here",
        "Bad example here",
    ]
    categorical_data = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]]).astype(int)
    labels = np.array([1, 0, 1, 0, 1, 5])

    return sample_text_data, categorical_data, labels


@pytest.fixture
def model_params():
    """Fixture providing common model parameters."""
    return {
        "embedding_dim": 96,
        "n_layers": 2,
        "n_head": 4,
        "num_classes": 10,
        "categorical_vocab_sizes": [2, 2],
        "categorical_embedding_dims": [4, 7],
    }


def run_full_pipeline(tokenizer, sample_text_data, categorical_data, labels, model_params):
    """Helper function to run the complete pipeline for a given tokenizer."""
    # Create dataset
    dataset = TextClassificationDataset(
        texts=sample_text_data,
        categorical_variables=categorical_data.tolist(),
        tokenizer=tokenizer,
        labels=None,
    )

    dataloader = dataset.create_dataloader(batch_size=4)
    batch = next(iter(dataloader))

    # Get tokenizer parameters
    vocab_size = tokenizer.vocab_size
    padding_idx = tokenizer.padding_idx
    sequence_len = tokenizer.output_dim

    # Create attention config
    attention_config = AttentionConfig(
        n_layers=model_params["n_layers"],
        n_head=model_params["n_head"],
        n_kv_head=model_params["n_head"],
        sequence_len=sequence_len,
    )

    # Create text embedder
    text_embedder_config = TextEmbedderConfig(
        vocab_size=vocab_size,
        embedding_dim=model_params["embedding_dim"],
        padding_idx=padding_idx,
        attention_config=attention_config,
    )

    text_embedder = TextEmbedder(text_embedder_config=text_embedder_config)
    text_embedder.init_weights()

    # Create categorical variable net
    categorical_var_net = CategoricalVariableNet(
        categorical_vocabulary_sizes=model_params["categorical_vocab_sizes"],
        categorical_embedding_dims=model_params["categorical_embedding_dims"],
    )

    # Create classification head
    expected_input_dim = model_params["embedding_dim"] + categorical_var_net.output_dim
    classification_head = ClassificationHead(
        input_dim=expected_input_dim,
        num_classes=model_params["num_classes"],
    )

    # Create model
    model = TextClassificationModel(
        text_embedder=text_embedder,
        categorical_variable_net=categorical_var_net,
        classification_head=classification_head,
    )

    # Test forward pass
    model(**batch)

    # Create module
    module = TextClassificationModule(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        optimizer_params={"lr": 1e-3},
        scheduler=None,
        scheduler_params=None,
        scheduler_interval="epoch",
    )

    # Test prediction
    module.predict_step(batch)

    # Prepare data for training
    X = np.column_stack([sample_text_data, categorical_data])
    Y = labels

    # Create model config
    model_config = ModelConfig(
        embedding_dim=model_params["embedding_dim"],
        categorical_vocabulary_sizes=model_params["categorical_vocab_sizes"],
        categorical_embedding_dims=model_params["categorical_embedding_dims"],
        num_classes=model_params["num_classes"],
        attention_config=attention_config,
    )

    # Create training config
    training_config = TrainingConfig(
        lr=1e-3,
        batch_size=4,
        num_epochs=1,
    )

    # Create classifier
    ttc = torchTextClassifiers(
        tokenizer=tokenizer,
        model_config=model_config,
    )

    # Train
    ttc.train(
        X_train=X,
        y_train=Y,
        X_val=X,
        y_val=Y,
        training_config=training_config,
    )

    # Predict with explanations
    top_k = 5
    predictions = ttc.predict(X, top_k=top_k, explain=True)

    # Test explainability functions
    text_idx = 0
    text = sample_text_data[text_idx]
    offsets = predictions["offset_mapping"][text_idx]
    attributions = predictions["attributions"][text_idx]
    word_ids = predictions["word_ids"][text_idx]

    word_attributions = map_attributions_to_word(attributions, word_ids)
    char_attributions = map_attributions_to_char(attributions, offsets, text)

    # Note: We're not actually plotting in tests, just calling the functions
    # to ensure they don't raise errors
    plot_attributions_at_char(text, char_attributions)
    plot_attributions_at_word(text, word_attributions)


def test_wordpiece_tokenizer(sample_data, model_params):
    """Test the full pipeline with WordPieceTokenizer."""
    sample_text_data, categorical_data, labels = sample_data

    vocab_size = 100
    tokenizer = WordPieceTokenizer(vocab_size, output_dim=50)
    tokenizer.train(sample_text_data)

    # Check tokenizer works
    result = tokenizer.tokenize(sample_text_data)
    assert result.input_ids.shape[0] == len(sample_text_data)

    # Run full pipeline
    run_full_pipeline(tokenizer, sample_text_data, categorical_data, labels, model_params)


def test_huggingface_tokenizer(sample_data, model_params):
    """Test the full pipeline with HuggingFaceTokenizer."""
    sample_text_data, categorical_data, labels = sample_data

    tokenizer = HuggingFaceTokenizer.load_from_pretrained(
        "google-bert/bert-base-uncased", output_dim=50
    )

    # Check tokenizer works
    result = tokenizer.tokenize(sample_text_data)
    assert result.input_ids.shape[0] == len(sample_text_data)

    # Run full pipeline
    run_full_pipeline(tokenizer, sample_text_data, categorical_data, labels, model_params)


def test_ngram_tokenizer(sample_data, model_params):
    """Test the full pipeline with NGramTokenizer."""
    sample_text_data, categorical_data, labels = sample_data

    tokenizer = NGramTokenizer(
        min_count=3, min_n=2, max_n=5, num_tokens=100, len_word_ngrams=2, output_dim=76
    )
    tokenizer.train(sample_text_data)

    # Check tokenizer works
    result = tokenizer.tokenize(
        sample_text_data[0], return_offsets_mapping=True, return_word_ids=True
    )
    assert result.input_ids is not None

    # Check batch decode
    batch_result = tokenizer.tokenize(sample_text_data)
    decoded = tokenizer.batch_decode(batch_result.input_ids.tolist())
    assert len(decoded) == len(sample_text_data)

    # Run full pipeline
    run_full_pipeline(tokenizer, sample_text_data, categorical_data, labels, model_params)
