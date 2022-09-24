from transformers import (
    MMBTForClassification,
    MMBTConfig,
    AutoConfig,
    AutoModel,
)

from src.models.MMBT.mmbt import PRETRAINED_MODEL_CKP
from src.models.MMBT_with_CLIP_encoder.img_encoder import (
    ClipEncoderMulti
)


def load_model(
            clip_model,
            ckp=PRETRAINED_MODEL_CKP,
            output_hidden_states=False,
            model_hidden_size=512,
            num_embeds=4
        ):
    # transformerのロード
    transformer_config = AutoConfig.from_pretrained(ckp)
    transformer = AutoModel.from_pretrained(ckp)

    # 画像encoder準備
    img_encoder = ClipEncoderMulti(
        clip_model,
        num_embeds=num_embeds,
        num_features=model_hidden_size
    )

    # MMBTモデル構築
    mmbt_config = MMBTConfig(
        transformer_config,
        num_labels=2,
        modal_hidden_size=model_hidden_size
    )
    mmbt_config.output_hidden_states = output_hidden_states
    model = MMBTForClassification(
        mmbt_config,
        transformer,
        img_encoder
    )
    mmbt_config.use_return_dict = True
    model.config = model.mmbt.config
    return model
