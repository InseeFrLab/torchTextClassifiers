import logging
import os
from typing import List, Optional

from torchTextClassifiers.tokenizers import HAS_HF, HuggingFaceTokenizer

if not HAS_HF:
    raise ImportError(
        "The HuggingFace dependencies are needed to use this tokenizer. Please run 'uv add torchTextClassifiers --extra huggingface."
    )
else:
    from tokenizers import (
        Tokenizer,
        decoders,
        models,
        normalizers,
        pre_tokenizers,
        processors,
        trainers,
    )
    from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class WordPieceTokenizer(HuggingFaceTokenizer):
    def __init__(self, vocab_size: int, trained: bool = False, output_dim: Optional[int] = None):
        """Largely inspired by https://huggingface.co/learn/llm-course/chapter6/8"""

        super().__init__(vocab_size=vocab_size, output_dim=output_dim)

        self.unk_token = "[UNK]"
        self.pad_token = "[PAD]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.special_tokens = [
            self.unk_token,
            self.pad_token,
            self.cls_token,
            self.sep_token,
        ]
        self.vocab_size = vocab_size
        self.context_size = output_dim

        self.tokenizer = Tokenizer(models.WordPiece(unk_token=self.unk_token))

        self.tokenizer.normalizer = normalizers.BertNormalizer(
            lowercase=True
        )  # NFD, lowercase, strip accents - BERT style

        self.tokenizer.pre_tokenizer = (
            pre_tokenizers.BertPreTokenizer()
        )  # split on whitespace and punctuation - BERT style
        self.trained = trained

    def _post_training(self):
        if not self.trained:
            raise RuntimeError(
                "Tokenizer must be trained before applying post-training configurations."
            )

        self.tokenizer.post_processor = processors.BertProcessing(
            (self.cls_token, self.tokenizer.token_to_id(self.cls_token)),
            (self.sep_token, self.tokenizer.token_to_id(self.sep_token)),
        )
        self.tokenizer.decoder = decoders.WordPiece(prefix="##")
        self.padding_idx = self.tokenizer.token_to_id("[PAD]")
        self.tokenizer.enable_padding(pad_id=self.padding_idx, pad_token="[PAD]")

        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=self.tokenizer)
        self.vocab_size = len(self.tokenizer)

    def train(
        self, training_corpus: List[str], save_path: str = None, filesystem=None, s3_save_path=None
    ):
        trainer = trainers.WordPieceTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
        )
        self.tokenizer.train_from_iterator(training_corpus, trainer=trainer)
        self.trained = True
        self._post_training()

        if save_path:
            self.tokenizer.save(save_path)
            logger.info(f"💾 Tokenizer saved at {save_path}")
            if filesystem and s3_save_path:
                parent_dir = os.path.dirname(save_path)
                if not filesystem.exists(parent_dir):
                    filesystem.mkdirs(parent_dir)
                filesystem.put(save_path, s3_save_path)
                logger.info(f"💾 Tokenizer uploaded to S3 at {s3_save_path}")
