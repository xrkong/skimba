from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer

from dataclasses import dataclass, field
from typing import Optional

from stable_diffusion.dataclass import BaseDataclass
from diffusers import StableDiffusionPipeline


@dataclass
class ClipConfig(BaseDataclass):
    tokenizer: str = field(
        default="runwayml/stable-diffusion-v1-5",
        metadata={"help": "Tokenizer to use for text encoding."},
    )
    text_encoder: str = field(
        default="runwayml/stable-diffusion-v1-5",
        metadata={"help": "Text encoder model to use."},
    )
    max_seq_len: int = field(
        default=77, metadata={"help": "Maximum sequence length for tokenized text."}
    )
    model_dir: Optional[str] = field(
        default="data/pretrained",
        metadata={"help": "Path to a directory to store the pretrained CLIP model."},
    )


class CLIPModel(nn.Module):
    # @staticmethod
    # def add_clip_args(model_parser):
    #     clip_group = model_parser.add_argument_group("clip")
    #     clip_group.add_argument(
    #         "--tokenizer",
    #         type=str,
    #         default="runwayml/stable-diffusion-v1-5",
    #     )
    #     clip_group.add_argument(
    #         "--text_encoder",
    #         type=str,
    #         default="runwayml/stable-diffusion-v1-5",
    #     )
    #     clip_group.add_argument(
    #         "--max_seq_len",
    #         type=int,
    #         default=77,
    #     )
    #     clip_group.add_argument(
    #         "--cache_dir",
    #         type=str,
    #         default=None,
    #         help="Path to a directory to store the pretrained clip model",
    #     )
    #     return clip_group
    def __init__(
        self,
        tokenizer = None,
        text_encoder =None,
        max_seq_len = 77,
        cache_dir = None,

    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.max_seq_len = max_seq_len
        self.cache_dir = cache_dir
        self.max_seq_len = self.max_seq_len
        self.text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
            self.text_encoder, cache_dir=self.cache_dir, subfolder="text_encoder"
        )
        # self.text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(self.text_encoder)  # transformer
        self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
            self.tokenizer,
            use_fast=False,
            cache_dir=self.cache_dir,
            subfolder="tokenizer",
        )

    def tokenize(
        self,
        prompt: str = "",
        max_length: int = None,
        padding: str = "max_length",
        truncation: bool = True,
    ):
        return self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
            max_length=(self.max_seq_len if max_length is None else max_length),
        )

    def encode_text(self, text):
        """Encode text to text embedding
        Args:
            - text (str):
                  text to encode, shape = [batch, seq_len]
        Returns:
            - text_embedding (torch.Tensor):
                  text embedding, shape = [batch, seq_len, d_model]
        """
        return self.text_encoder(text)
