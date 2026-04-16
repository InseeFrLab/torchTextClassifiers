import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
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

labels = np.array(["positive", "negative", "positive", "negative", "positive", "negative"])


df = pd.DataFrame(
    {
        "text": sample_text_data,
        "category": categorical_data[:, 0],
        "color": categorical_data[:, 1],
        "label": labels,
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

feature = "label"
le_label = LabelEncoder()
le_label.fit(df[feature])
label_encoder = le_label

# OR you can also use DictEncoder
# dict_mapping = {val: idx for idx, val in enumerate(df[feature].unique())}
# label_encoder = DictEncoder(dict_mapping)

value_encoder = ValueEncoder(label_encoder, encoders)


model_config = ModelConfig(
    embedding_dim=10,
    categorical_embedding_dims=[5, 5],
)
training_config = TrainingConfig(
    num_epochs=1,
    batch_size=2,
    lr=1e-3,
)

ttc = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config,
    value_encoder=value_encoder,
)

ttc.train(
    X_train=df[["text", "category", "color"]].values,
    y_train=df["label"].values,
    training_config=training_config,
)

torchTextClassifiers.load("my_ttc/")

ttc.predict(
    X_test=df[["text", "category", "color"]].values,
    raw_categorical_inputs=True,  # Set to True since we're providing raw categorical values
    top_k=2,
)
