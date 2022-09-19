from transformers import (
    MMBTForClassification,
    MMBTConfig,
    AutoConfig,
    AutoModel,
)
from torch import nn
import torch
from torchvision.models import resnet152, ResNet152_Weights


PRETRAINED_MODEL_CKP = 'cl-tohoku/bert-base-japanese-whole-word-masking'


class ImageEncoder(nn.Module):
    POOLING_BREAKDOWN = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2), 5: (5, 1), 6: (3, 2), 7: (7, 1), 8: (4, 2), 9: (3, 3)}

    def __init__(self, pretrained_weight):
        super().__init__()
        model = resnet152(weights=pretrained_weight)
        # 最後のaveragePooling2d, 全結合層を除く
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d(self.POOLING_BREAKDOWN[3])

    def forward(self, x):
        # size: (N, 2048, 3, 1)
        out = self.pool(self.model(x))
        # size: (N, 2048, 3)
        out = torch.flatten(out, start_dim=2)
        # size: (N, 3, 2048)
        out = out.transpose(1, 2).contiguous()
        return out


def load_model(
            ckp=PRETRAINED_MODEL_CKP,
            output_hidden_states=False
        ):
    transformer_config = AutoConfig.from_pretrained(ckp)
    transformer = AutoModel.from_pretrained(ckp)
    mmbt_config = MMBTConfig(transformer_config, num_labels=2)
    mmbt_config.output_hidden_states = output_hidden_states
    model = MMBTForClassification(
        mmbt_config,
        transformer,
        ImageEncoder(ResNet152_Weights.IMAGENET1K_V2)
    )
    mmbt_config.use_return_dict = True
    model.config = model.mmbt.config
    return model
