import pytest
import torch

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
from torchTextClassifiers.model.model import TextClassificationModel

BATCH = 4
SEQ_LEN = 20
EMB_DIM = 16  # divisible by 4 (n_head) and head_dim=4 is even (rotary)
VOCAB_SIZE = 100
PADDING_IDX = 0
NUM_CLASSES = 3


@pytest.fixture
def input_ids():
    ids = torch.randint(1, VOCAB_SIZE, (BATCH, SEQ_LEN))
    ids[:, -2:] = PADDING_IDX
    return ids


@pytest.fixture
def attention_mask(input_ids):
    return (input_ids != PADDING_IDX).long()


@pytest.fixture
def token_embeddings():
    return torch.randn(BATCH, SEQ_LEN, EMB_DIM)


class TestTokenEmbedder:
    def test_no_attention(self, input_ids, attention_mask):
        embedder = TokenEmbedder(
            TokenEmbedderConfig(
                vocab_size=VOCAB_SIZE, embedding_dim=EMB_DIM, padding_idx=PADDING_IDX
            )
        )
        out = embedder(input_ids, attention_mask)
        assert out["token_embeddings"].shape == (BATCH, SEQ_LEN, EMB_DIM)
        assert out["attention_mask"].shape == (BATCH, SEQ_LEN)

    def test_with_attention(self, input_ids, attention_mask):
        embedder = TokenEmbedder(
            TokenEmbedderConfig(
                vocab_size=VOCAB_SIZE,
                embedding_dim=EMB_DIM,
                padding_idx=PADDING_IDX,
                attention_config=AttentionConfig(
                    n_layers=2, n_head=4, n_kv_head=4, positional_encoding=False
                ),
            )
        )
        out = embedder(input_ids, attention_mask)
        assert out["token_embeddings"].shape == (BATCH, SEQ_LEN, EMB_DIM)

    def test_with_rotary_positional_encoding(self, input_ids, attention_mask):
        embedder = TokenEmbedder(
            TokenEmbedderConfig(
                vocab_size=VOCAB_SIZE,
                embedding_dim=EMB_DIM,
                padding_idx=PADDING_IDX,
                attention_config=AttentionConfig(
                    n_layers=1,
                    n_head=4,
                    n_kv_head=4,
                    positional_encoding=True,
                    sequence_len=SEQ_LEN,
                ),
            )
        )
        out = embedder(input_ids, attention_mask)
        assert out["token_embeddings"].shape == (BATCH, SEQ_LEN, EMB_DIM)

    def test_shape_mismatch_raises(self):
        embedder = TokenEmbedder(
            TokenEmbedderConfig(
                vocab_size=VOCAB_SIZE, embedding_dim=EMB_DIM, padding_idx=PADDING_IDX
            )
        )
        with pytest.raises(ValueError):
            embedder(
                torch.randint(1, VOCAB_SIZE, (BATCH, SEQ_LEN)),
                torch.ones(BATCH, SEQ_LEN + 1, dtype=torch.long),
            )


class TestSentenceEmbedder:
    @pytest.mark.parametrize("method", ["mean", "first", "last"])
    def test_aggregation_methods(self, token_embeddings, attention_mask, method):
        embedder = SentenceEmbedder(SentenceEmbedderConfig(aggregation_method=method))
        out = embedder(token_embeddings, attention_mask)
        assert out["sentence_embedding"].shape == (BATCH, EMB_DIM)
        assert out["label_attention_matrix"] is None

    def test_label_attention_output_shape(self, token_embeddings, attention_mask):
        embedder = SentenceEmbedder(
            SentenceEmbedderConfig(
                aggregation_method=None,
                label_attention_config=LabelAttentionConfig(
                    n_head=4, num_classes=NUM_CLASSES, embedding_dim=EMB_DIM
                ),
            )
        )
        out = embedder(token_embeddings, attention_mask)
        assert out["sentence_embedding"].shape == (BATCH, NUM_CLASSES, EMB_DIM)
        assert out["label_attention_matrix"] is None

    def test_label_attention_matrix_returned(self, token_embeddings, attention_mask):
        embedder = SentenceEmbedder(
            SentenceEmbedderConfig(
                aggregation_method=None,
                label_attention_config=LabelAttentionConfig(
                    n_head=4, num_classes=NUM_CLASSES, embedding_dim=EMB_DIM
                ),
            )
        )
        out = embedder(token_embeddings, attention_mask, return_label_attention_matrix=True)
        assert out["label_attention_matrix"].shape == (BATCH, 4, NUM_CLASSES, SEQ_LEN)

    def test_none_aggregation_without_label_attention_raises(self):
        with pytest.raises(ValueError):
            SentenceEmbedder(SentenceEmbedderConfig(aggregation_method=None))


class TestCategoricalVariableNet:
    def test_concatenate_all(self):
        net = CategoricalVariableNet(
            categorical_vocabulary_sizes=[4, 5],
            categorical_embedding_dims=[3, 6],
        )
        assert net.forward_type == CategoricalForwardType.CONCATENATE_ALL
        assert net.output_dim == 9
        out = net(torch.randint(0, 3, (BATCH, 2)))
        assert out.shape == (BATCH, 9)

    def test_average_and_concat(self):
        net = CategoricalVariableNet(
            categorical_vocabulary_sizes=[4, 5],
            categorical_embedding_dims=8,
        )
        assert net.forward_type == CategoricalForwardType.AVERAGE_AND_CONCAT
        assert net.output_dim == 8
        out = net(torch.randint(0, 3, (BATCH, 2)))
        assert out.shape == (BATCH, 8)

    def test_sum_to_text(self):
        net = CategoricalVariableNet(
            categorical_vocabulary_sizes=[4, 5],
            categorical_embedding_dims=None,
            text_embedding_dim=EMB_DIM,
        )
        assert net.forward_type == CategoricalForwardType.SUM_TO_TEXT
        assert net.output_dim == EMB_DIM
        out = net(torch.randint(0, 3, (BATCH, 2)))
        assert out.shape == (BATCH, EMB_DIM)

    def test_out_of_range_value_raises(self):
        net = CategoricalVariableNet(
            categorical_vocabulary_sizes=[4, 5],
            categorical_embedding_dims=[3, 6],
        )
        with pytest.raises(ValueError):
            net(torch.tensor([[10, 1]] * BATCH))  # first feature value 10 >= vocab 4


class TestTextClassificationModel:
    def _token_embedder(self):
        return TokenEmbedder(
            TokenEmbedderConfig(
                vocab_size=VOCAB_SIZE, embedding_dim=EMB_DIM, padding_idx=PADDING_IDX
            )
        )

    def _sentence_embedder(self, label_attention=False):
        if label_attention:
            return SentenceEmbedder(
                SentenceEmbedderConfig(
                    aggregation_method=None,
                    label_attention_config=LabelAttentionConfig(
                        n_head=4, num_classes=NUM_CLASSES, embedding_dim=EMB_DIM
                    ),
                )
            )
        return SentenceEmbedder(SentenceEmbedderConfig(aggregation_method="mean"))

    def test_text_only(self, input_ids, attention_mask):
        model = TextClassificationModel(
            token_embedder=self._token_embedder(),
            sentence_embedder=self._sentence_embedder(),
            classification_head=ClassificationHead(input_dim=EMB_DIM, num_classes=NUM_CLASSES),
        )
        logits = model(input_ids, attention_mask, torch.empty(BATCH, 0))
        assert logits.shape == (BATCH, NUM_CLASSES)

    def test_text_and_categorical(self, input_ids, attention_mask):
        cat_net = CategoricalVariableNet(
            categorical_vocabulary_sizes=[4, 5],
            categorical_embedding_dims=[3, 6],
        )
        model = TextClassificationModel(
            token_embedder=self._token_embedder(),
            sentence_embedder=self._sentence_embedder(),
            categorical_variable_net=cat_net,
            classification_head=ClassificationHead(
                input_dim=EMB_DIM + cat_net.output_dim, num_classes=NUM_CLASSES
            ),
        )
        logits = model(input_ids, attention_mask, torch.randint(0, 3, (BATCH, 2)))
        assert logits.shape == (BATCH, NUM_CLASSES)

    def test_label_attention_logits_and_matrix(self, input_ids, attention_mask):
        model = TextClassificationModel(
            token_embedder=self._token_embedder(),
            sentence_embedder=self._sentence_embedder(label_attention=True),
            classification_head=ClassificationHead(input_dim=EMB_DIM, num_classes=1),
        )
        result = model(
            input_ids,
            attention_mask,
            torch.empty(BATCH, 0),
            return_label_attention_matrix=True,
        )
        assert result["logits"].shape == (BATCH, NUM_CLASSES)
        assert result["label_attention_matrix"].shape == (BATCH, 4, NUM_CLASSES, SEQ_LEN)

    def test_missing_sentence_embedder_raises(self):
        with pytest.raises(ValueError):
            TextClassificationModel(
                token_embedder=self._token_embedder(),
                sentence_embedder=None,
                classification_head=ClassificationHead(input_dim=EMB_DIM, num_classes=NUM_CLASSES),
            )
