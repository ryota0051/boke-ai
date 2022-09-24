import torch
from torch import nn


class ClipEncoderMulti(nn.Module):
    def __init__(self, clip_model, num_embeds, num_features):
        super().__init__()
        self.clip_model = clip_model
        self.num_embeds = num_embeds
        self.num_features = num_features

    def forward(self, x: torch.Tensor):
        out = self.clip_model.get_image_features(x.view(-1, *x.size()[2:]))
        out = out.view(-1, self.num_embeds, self.num_features).float()
        return out
