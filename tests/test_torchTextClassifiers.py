from torchTextClassifiers import torchTextClassifiers
from torchTextClassifiers.model import TextClassificationModel
from torchTextClassifiers.model.components import ClassificationHead


class TestTorchTextClassifiers:
    """Test the main torchTextClassifiers class."""

    def test_initialization(self, model_config, mock_tokenizer):
        """Test basic initialization."""
        ttc = torchTextClassifiers(
            tokenizer=mock_tokenizer,
            model_config=model_config,
        )

        assert isinstance(ttc.pytorch_model, TextClassificationModel)
        assert isinstance(ttc.classification_head, ClassificationHead)
