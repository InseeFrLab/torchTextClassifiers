from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn

import torchTextClassifiers
from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.dataset import TextClassificationDataset
from torchTextClassifiers.model import TextClassificationModule
from torchTextClassifiers.model.components import (
    AttentionConfig,
    CategoricalForwardType,
    CategoricalVariableNet,
    ClassificationHead,
    LabelAttentionConfig,
    SentenceEmbedder,
    SentenceEmbedderConfig,
    TokenEmbedder,
    TokenEmbedderConfig,
)
from torchTextClassifiers.tokenizers import WordPieceTokenizer
from torchTextClassifiers.value_encoder import DictEncoder, ValueEncoder

sample_text_data = [
    "This is a positive example",
    "This is a negative example",
    "Another positive case",
    "Another negative case",
    "Good example here",
    "Bad example here",
]

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

labels_level_1 = np.array(["positive", "negative", "positive", "negative", "positive", "neutral"])
labels_level_2 = np.array(["good", "bad", "good", "bad", "good", "bad"])
labels_level3 = np.array(["A", "B", "D", "B", "C", "B"])


df = pd.DataFrame(
    {
        "text": sample_text_data,
        "category": categorical_data[:, 0],
        "color": categorical_data[:, 1],
        "label_level_1": labels_level_1,  # You can switch to labels_level_2 or labels_level_3 for testing
        "label_level_2": labels_level_2,
        "label_level_3": labels_level3,
    }
)
vocab_size = 10
tokenizer = WordPieceTokenizer(vocab_size, output_dim=50)
tokenizer.train(sample_text_data)

encoders = {}
# category : DictEncoder (ours)
feature = "category"
mapping = {val: idx for idx, val in enumerate(df[feature].unique())}
encoders[feature] = DictEncoder(mapping)

# color: LabelEncoder (sklearn)
le = LabelEncoder()
le.fit(df["color"])
encoders["color"] = le

feature = "label_level_1"
le_label = LabelEncoder()
le_label.fit(df[feature])

feature = "label_level_2"
le_label_2 = LabelEncoder()
le_label_2.fit(df[feature])

feature = "label_level_3"
le_label_3 = DictEncoder({val: idx for idx, val in enumerate(df[feature].unique())})

label_encoder = [le_label, le_label_2, le_label_3]
# OR you can also use DictEncoder
# dict_mapping = {val: idx for idx, val in enumerate(df[feature].unique())}
# label_encoder = DictEncoder(dict_mapping)

value_encoder = ValueEncoder(label_encoder, encoders)


model_config = ModelConfig(
    embedding_dim=10,
    categorical_embedding_dims=5,
    n_heads_label_attention=2,
    num_classes=value_encoder.num_classes,
    attention_config=AttentionConfig(n_layers=2, n_head=5, n_kv_head=5, positional_encoding=False),
    aggregation_method=None,
)
training_config = TrainingConfig(
    num_epochs=1,
    batch_size=6,
    lr=1e-3,
    raw_categorical_inputs=True,
)

train_dataset = TextClassificationDataset(
    texts=df["text"].values,
    categorical_variables=value_encoder.transform(
        df[["category", "color"]].values
    ),  # None if no cat vars
    tokenizer=tokenizer,
    labels=value_encoder.transform_labels(
        df[["label_level_1", "label_level_2", "label_level_3"]].values
    ),  # None if no labels
)
train_dataloader = train_dataset.create_dataloader(
    batch_size=training_config.batch_size,
    num_workers=training_config.num_workers,
    shuffle=False,
    **training_config.dataloader_params if training_config.dataloader_params else {},
)
batch = next(iter(train_dataloader))


token_embedder_config = TokenEmbedderConfig(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=model_config.embedding_dim,
    padding_idx=tokenizer.padding_idx,
    attention_config=model_config.attention_config,
)
token_embedder = TokenEmbedder(
    token_embedder_config=token_embedder_config,
)
categorical_var_net = CategoricalVariableNet(
    categorical_vocabulary_sizes=value_encoder.vocabulary_sizes,
    categorical_embedding_dims=model_config.categorical_embedding_dims,
    text_embedding_dim=model_config.embedding_dim,
)

all_sentence_embedders = []
all_classification_heads = []

for num_classes in value_encoder.num_classes:  # ty:ignore[not-iterable]
    sentence_embedder_config = SentenceEmbedderConfig(
        label_attention_config=LabelAttentionConfig(
            n_head=model_config.n_heads_label_attention,
            num_classes=num_classes,
            embedding_dim=model_config.embedding_dim,
        ),
        aggregation_method=model_config.aggregation_method,
    )

    sentence_embedder = SentenceEmbedder(sentence_embedder_config=sentence_embedder_config)
    all_sentence_embedders.append(sentence_embedder)

    classif_head_input_dim = model_config.embedding_dim
    if categorical_var_net.forward_type != CategoricalForwardType.SUM_TO_TEXT:
        classif_head_input_dim += categorical_var_net.output_dim

    # because we use LabelAttention, the sentence embedder outputs a (num_classes, embedding_dim) tensor, and the classification head should output a single logit per class (i.e. num_classes=1)
    classification_head = ClassificationHead(input_dim=classif_head_input_dim, num_classes=1)
    all_classification_heads.append(classification_head)


class MultiLevelTextClassificationModel(nn.Module):
    def __init__(
        self,
        token_embedder: TokenEmbedder,
        sentence_embedders: list[SentenceEmbedder],
        classification_heads: list[ClassificationHead],
        categorical_variable_net: CategoricalVariableNet,
    ):
        super().__init__()
        self.token_embedder = token_embedder
        self.sentence_embedders = sentence_embedders
        self.classification_heads = classification_heads
        self.categorical_variable_net = categorical_variable_net
        self.num_classes: list[int] = [
            self.sentence_embedders[i].label_attention_config.num_classes
            if self.sentence_embedders[i].label_attention_config is not None
            else self.classification_heads[i].num_classes
            for i in range(len(self.sentence_embedders))
        ]

    def forward(self, input_ids, attention_mask, categorical_vars=None, **kwargs):
        token_embed_output = self.token_embedder(input_ids, attention_mask)
        x_token = token_embed_output["token_embeddings"]
        x_cat = self.categorical_variable_net(categorical_vars)  # (bs, cat_emb_dim)

        print(f"Token embeddings shape: {x_token.shape}")
        print(
            f"x_cat shape: {x_cat.shape}"
        )  # Debugging line to check shape of categorical variable embeddings
        outputs = []
        for sentence_embedder, classification_head in zip(
            self.sentence_embedders, self.classification_heads
        ):
            if sentence_embedder.label_attention_config is not None:
                num_classes = sentence_embedder.label_attention_config.num_classes
                x_cat_level = x_cat.unsqueeze(1).expand(-1, num_classes, -1)
            else:
                x_cat_level = x_cat

            print(
                f"x_cat_level shape: {x_cat_level.shape}"
            )  # Debugging line to check shape of categorical variable embeddings after expansion
            sentence_embedding = sentence_embedder(
                token_embeddings=x_token, attention_mask=attention_mask
            )[
                "sentence_embedding"
            ]  # (bs, embedding_dim) or (bs, num_classes, embedding_dim) if label attention

            print(
                f"Sentence embedding shape: {sentence_embedding.shape}"
            )  # Debugging line to check shape of sentence embeddings
            if (
                self.categorical_variable_net.forward_type
                == CategoricalForwardType.AVERAGE_AND_CONCAT
                or self.categorical_variable_net.forward_type
                == CategoricalForwardType.CONCATENATE_ALL
            ):
                x_combined = torch.cat((sentence_embedding, x_cat_level), dim=-1)
            else:
                assert (
                    self.categorical_variable_net.forward_type == CategoricalForwardType.SUM_TO_TEXT
                )

                x_combined = sentence_embedding + x_cat_level

            print(
                f"x_combined shape: {x_combined.shape}"
            )  # Debugging line to check shape of combined features before classification head
            output = classification_head(x_combined).squeeze(-1)
            outputs.append(output)
            print(
                f"Output shape for current level: {output.shape}"
            )  # Debugging line to check shape of output logits for current level

        return outputs


class MultiLevelCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes: Optional[list[int]] = None):
        super().__init__()
        self.num_classes = num_classes
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        total_loss = 0
        for idx, output in enumerate(outputs):
            label = labels[:, idx]  # (batch_size,)
            if self.num_classes is not None:
                total_loss += self.loss_fn(output.squeeze(), label) * self.num_classes[idx]
            else:
                total_loss += self.loss_fn(output.squeeze(), label)

        if self.num_classes is not None:
            total_weight = sum(self.num_classes)
        else:
            total_weight = len(outputs)

        return total_loss / total_weight  # average loss across levels


model = MultiLevelTextClassificationModel(
    token_embedder=token_embedder,
    sentence_embedders=all_sentence_embedders,
    classification_heads=all_classification_heads,
    categorical_variable_net=categorical_var_net,
)

module = TextClassificationModule(
    model=model,
    loss=MultiLevelCrossEntropyLoss(),
    optimizer=torch.optim.Adam,
    optimizer_params={"lr": 1e-3},
    scheduler=None,
    scheduler_params=None,
)

print(model.num_classes)

batch = next(iter(train_dataloader))
print(batch["labels"].shape)
outputs = model(**batch)
print(f"Outputs shapes: {[output.shape for output in outputs]}")


ttc = torchTextClassifiers.from_model(
    tokenizer=tokenizer, pytorch_model=model, value_encoder=value_encoder
)

training_config = TrainingConfig(
    num_epochs=1,
    batch_size=6,
    lr=1e-3,
    raw_categorical_inputs=True,
    loss=MultiLevelCrossEntropyLoss(num_classes=value_encoder.num_classes),
)

ttc.train(
    X_train=df[["text", "category", "color"]].values,
    y_train=df[["label_level_1", "label_level_2", "label_level_3"]].values,
    training_config=training_config,
)


print(
    ttc.predict(
        X_test=df[["text", "category", "color"]].values,
    )
)
