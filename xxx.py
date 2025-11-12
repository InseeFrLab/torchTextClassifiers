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
from torchTextClassifiers.tokenizers import HuggingFaceTokenizer, WordPieceTokenizer
from torchTextClassifiers.utilities.plot_explainability import (
    map_attributions_to_char,
    map_attributions_to_word,
    plot_attributions_at_char,
    plot_attributions_at_word,
)

sample_text_data = [
    "This is a positive example",
    "This is a negative example",
    "Another positive case",
    "Another negative case",
    "Good example here",
    "Bad example here",
]
categorical_data = [[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]]
labels = [1, 0, 1, 0, 1, 5]

###
tokenizer = WordPieceTokenizer(3, output_dim=None)
tokenizer.train(sample_text_data)
tokenizer.tokenize(sample_text_data).input_ids.shape


###
tokenizer = HuggingFaceTokenizer.load_from_pretrained(
    "google-bert/bert-base-uncased", output_dim=126
)
tokenizer.tokenize(sample_text_data).input_ids.shape


dataset = TextClassificationDataset(
    texts=sample_text_data, categorical_variables=categorical_data, tokenizer=tokenizer, labels=None
)

dataloader = dataset.create_dataloader(batch_size=4)

batch = next(iter(dataloader))

vocab_size = tokenizer.vocab_size
padding_idx = tokenizer.padding_idx

embedding_dim = 96
n_layers = 2
n_head = 4
n_kv_head = n_head
sequence_len = tokenizer.output_dim

attention_config = AttentionConfig(
    n_layers=n_layers,
    n_head=n_head,
    n_kv_head=n_kv_head,
    sequence_len=sequence_len,
)

text_embedder_config = TextEmbedderConfig(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    padding_idx=padding_idx,
    attention_config=attention_config,
)


text_embedder = TextEmbedder(
    text_embedder_config=text_embedder_config,
)
text_embedder.init_weights()


categorical_vocab_sizes = [2, 2]
categorical_embedding_dims = [4, 7]

categorical_var_net = CategoricalVariableNet(
    categorical_vocabulary_sizes=categorical_vocab_sizes,
    categorical_embedding_dims=categorical_embedding_dims,
)

num_classes = 10
expected_input_dim = embedding_dim + categorical_var_net.output_dim
classification_head = ClassificationHead(
    input_dim=expected_input_dim,
    num_classes=num_classes,
)

model = TextClassificationModel(
    text_embedder=text_embedder,
    categorical_variable_net=categorical_var_net,
    classification_head=classification_head,
)

model(**batch)

import torch

module = TextClassificationModule(
    model=model,
    loss=torch.nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam,
    optimizer_params={"lr": 1e-3},
    scheduler=None,
    scheduler_params=None,
    scheduler_interval="epoch",
)

module.predict_step(batch)

# Convert categorical data to numpy array
import numpy as np

categorical_data = np.array(categorical_data).astype(int)

# Combine text (as a column vector) with categorical data
X = np.column_stack([sample_text_data, categorical_data])
Y = np.array(labels)

model_config = ModelConfig(
    embedding_dim=embedding_dim,
    categorical_vocabulary_sizes=categorical_vocab_sizes,
    categorical_embedding_dims=categorical_embedding_dims,
    num_classes=num_classes,
    attention_config=attention_config,
)

training_config = TrainingConfig(
    lr=1e-3,
    batch_size=4,
    num_epochs=1,
)

ttc = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config,
)

ttc.train(
    X_train=X,
    y_train=Y,
    X_val=X,
    y_val=Y,
    training_config=training_config,
)

top_k = 5
yyy = ttc.predict(X, top_k=top_k, explain=True)

text_idx = 0
text = sample_text_data[text_idx]
offsets = yyy["offset_mapping"][text_idx]  # seq_len, 2
attributions = yyy["attributions"][text_idx]  # top_k, seq_len
word_ids = yyy["word_ids"][text_idx]  # seq_len

word_attributions = map_attributions_to_word(attributions, word_ids)
char_attributions = map_attributions_to_char(attributions, offsets, text)

plot_attributions_at_char(text, char_attributions)
plot_attributions_at_word(text, word_attributions)
