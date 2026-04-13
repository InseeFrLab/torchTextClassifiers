import numpy as np
import pytest
import torch
from sklearn.preprocessing import LabelEncoder

from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.categorical_value_encoder import CategoricalValueEncoder, DictEncoder
from torchTextClassifiers.dataset import TextClassificationDataset
from torchTextClassifiers.model import TextClassificationModel, TextClassificationModule
from torchTextClassifiers.model.components import (
    AttentionConfig,
    CategoricalVariableNet,
    ClassificationHead,
    LabelAttentionConfig,
    TextEmbedder,
    TextEmbedderConfig,
)
from torchTextClassifiers.tokenizers import NGramTokenizer

try:
    from torchTextClassifiers.tokenizers import HuggingFaceTokenizer, WordPieceTokenizer
except ImportError:
    pass

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
    # String categorical variables — two features, two unique values each
    categorical_data = np.array(
        [
            ["cat", "red"],
            ["dog", "blue"],
            ["cat", "red"],
            ["dog", "blue"],
            ["cat", "red"],
            ["dog", "blue"],
        ]
    )
    # String labels
    labels = np.array(["positive", "negative", "positive", "negative", "positive", "negative"])

    return sample_text_data, categorical_data, labels


@pytest.fixture
def model_params():
    """Fixture providing common model parameters (class count and vocab sizes are
    derived from data at runtime inside run_full_pipeline)."""
    return {
        "embedding_dim": 96,
        "n_layers": 2,
        "n_head": 4,
        "categorical_embedding_dims": [4, 7],
    }


def run_full_pipeline(
    tokenizer,
    sample_text_data,
    categorical_data,
    labels,
    model_params,
    label_attention_enabled: bool = False,
):
    """Helper function to run the complete pipeline for a given tokenizer."""

    # --- Encode categorical variables (string → int) ---
    n_features = categorical_data.shape[1]
    encoders = {
        str(i): DictEncoder(
            {v: j for j, v in enumerate(sorted(set(categorical_data[:, i].tolist())))}
        )
        for i in range(n_features)
    }
    cat_encoder = CategoricalValueEncoder(encoders)
    encoded_categorical = cat_encoder.transform(categorical_data)
    vocab_sizes = cat_encoder.vocabulary_sizes

    # --- Encode string labels to contiguous integers ---
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)

    # --- Direct component test: dataset with already-encoded integers ---
    dataset = TextClassificationDataset(
        texts=sample_text_data,
        categorical_variables=encoded_categorical.tolist(),
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
        label_attention_config=(
            LabelAttentionConfig(
                n_head=attention_config.n_head,
                num_classes=num_classes,
            )
            if label_attention_enabled
            else None
        ),
    )

    text_embedder = TextEmbedder(text_embedder_config=text_embedder_config)
    text_embedder.init_weights()

    # Create categorical variable net (vocab sizes from fitted encoder)
    categorical_var_net = CategoricalVariableNet(
        categorical_vocabulary_sizes=vocab_sizes,
        categorical_embedding_dims=model_params["categorical_embedding_dims"],
    )

    # Create classification head
    expected_input_dim = model_params["embedding_dim"] + categorical_var_net.output_dim
    classification_head = ClassificationHead(
        input_dim=expected_input_dim,
        num_classes=num_classes if not label_attention_enabled else 1,
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

    # --- Wrapper pipeline with string categorical data ---
    # X keeps categorical columns as raw strings; the wrapper encoder handles them.
    X = np.column_stack([sample_text_data, categorical_data])
    Y = encoded_labels  # integer-encoded labels (from LabelEncoder)

    # Create model config (vocab sizes and num_classes come from the encoders)
    model_config = ModelConfig(
        embedding_dim=model_params["embedding_dim"],
        categorical_vocabulary_sizes=vocab_sizes,
        categorical_embedding_dims=model_params["categorical_embedding_dims"],
        num_classes=num_classes,
        attention_config=attention_config,
        n_heads_label_attention=attention_config.n_head,
    )

    training_config = TrainingConfig(
        lr=1e-3,
        batch_size=4,
        num_epochs=1,
    )

    # Create classifier — pass the fitted categorical encoder
    ttc = torchTextClassifiers(
        tokenizer=tokenizer,
        model_config=model_config,
        categorical_encoder=cat_encoder,
    )

    # Train with raw string categorical data
    ttc.train(
        X_train=X,
        y_train=Y,
        X_val=X,
        y_val=Y,
        training_config=training_config,
    )
    assert ttc.save_path is not None
    ttc.load(ttc.save_path)  # test load (encoder is also saved/restored)

    # Predict with explanations
    top_k = min(5, num_classes)

    predictions = ttc.predict(
        X,
        top_k=top_k,
        explain_with_label_attention=label_attention_enabled,
        explain_with_captum=True,
    )

    # Test label attention assertions
    if label_attention_enabled:
        assert (
            predictions["label_attention_attributions"] is not None
        ), "Label attention attributions should not be None when label_attention_enabled is True"
        label_attention_attributions = predictions["label_attention_attributions"]
        expected_shape = (
            len(sample_text_data),  # batch_size
            model_params["n_head"],  # n_head
            num_classes,  # num_classes (derived from label encoder)
            tokenizer.output_dim,  # seq_len
        )
        assert label_attention_attributions.shape == expected_shape, (
            f"Label attention attributions shape mismatch. "
            f"Expected {expected_shape}, got {label_attention_attributions.shape}"
        )
    else:
        assert (
            predictions.get("label_attention_attributions") is None
        ), "Label attention attributions should be None when label_attention_enabled is False"

    # Test explainability functions
    text_idx = 0
    text = sample_text_data[text_idx]
    offsets = predictions["offset_mapping"][text_idx]
    attributions = predictions["captum_attributions"][text_idx]
    word_ids = predictions["word_ids"][text_idx]

    words, word_attributions = map_attributions_to_word(attributions, text, word_ids, offsets)
    char_attributions = map_attributions_to_char(attributions, offsets, text)

    plot_attributions_at_char(text, char_attributions)
    plot_attributions_at_word(
        text=text,
        words=words.values(),
        attributions_per_word=word_attributions,
    )


def test_wordpiece_tokenizer(sample_data, model_params):
    """Test the full pipeline with WordPieceTokenizer."""
    sample_text_data, categorical_data, labels = sample_data

    vocab_size = 100
    tokenizer = WordPieceTokenizer(vocab_size, output_dim=50)
    tokenizer.train(sample_text_data)

    result = tokenizer.tokenize(sample_text_data)
    assert result.input_ids.shape[0] == len(sample_text_data)

    run_full_pipeline(tokenizer, sample_text_data, categorical_data, labels, model_params)


def test_huggingface_tokenizer(sample_data, model_params):
    """Test the full pipeline with HuggingFaceTokenizer."""
    sample_text_data, categorical_data, labels = sample_data

    tokenizer = HuggingFaceTokenizer.load_from_pretrained(
        "google-bert/bert-base-uncased", output_dim=50
    )

    result = tokenizer.tokenize(sample_text_data)
    assert result.input_ids.shape[0] == len(sample_text_data)

    run_full_pipeline(tokenizer, sample_text_data, categorical_data, labels, model_params)


def test_ngram_tokenizer(sample_data, model_params):
    """Test the full pipeline with NGramTokenizer."""
    sample_text_data, categorical_data, labels = sample_data

    tokenizer = NGramTokenizer(
        min_count=3, min_n=2, max_n=5, num_tokens=100, len_word_ngrams=2, output_dim=76
    )
    tokenizer.train(sample_text_data)

    result = tokenizer.tokenize(
        sample_text_data[0], return_offsets_mapping=True, return_word_ids=True
    )
    assert result.input_ids is not None

    batch_result = tokenizer.tokenize(sample_text_data)
    decoded = tokenizer.batch_decode(batch_result.input_ids.tolist())
    assert len(decoded) == len(sample_text_data)

    run_full_pipeline(tokenizer, sample_text_data, categorical_data, labels, model_params)


def test_label_attention_enabled(sample_data, model_params):
    """Test the full pipeline with label attention enabled (using WordPieceTokenizer)."""
    sample_text_data, categorical_data, labels = sample_data

    vocab_size = 100
    tokenizer = WordPieceTokenizer(vocab_size, output_dim=50)
    tokenizer.train(sample_text_data)

    result = tokenizer.tokenize(sample_text_data)
    assert result.input_ids.shape[0] == len(sample_text_data)

    run_full_pipeline(
        tokenizer,
        sample_text_data,
        categorical_data,
        labels,
        model_params,
        label_attention_enabled=True,
    )
