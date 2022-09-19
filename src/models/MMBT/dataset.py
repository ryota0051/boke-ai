import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision.models import ResNet152_Weights
from torchvision.io import read_image
from transformers.models.bert_japanese.tokenization_bert_japanese import BertJapaneseTokenizer


class BokeTextImageDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            tokenizer: BertJapaneseTokenizer,
            max_seq_len: int,
            image_transform: ResNet152_Weights.IMAGENET1K_V2):
        self.df = df
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.image_trans = image_transform.transforms()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        sentence = torch.tensor(
            self.tokenizer.encode(
                row['text'],
                max_length=self.max_seq_len,
                padding='max_length',
                truncation=True
            )
        )
        start_token, sentence, end_token = sentence[0], sentence[1:-1], sentence[-1]
        img = self.image_trans(self._read_img(row['img_path']))
        return {
            'image_start_token': start_token,
            'image_end_token': end_token,
            'sentence': sentence,
            'image': img,
            'label': torch.tensor(row['is_laugh'])
        }

    @staticmethod
    def _read_img(path: str):
        image_tensor = read_image(path)
        # 白黒画像の場合にチャンネル数1 => 3にする。
        if image_tensor.shape[0] == 1:
            image_tensor = image_tensor.expand(3, *image_tensor.shape[1:])
        return image_tensor


def collate_fn(batch):
    """バッチ入力をtensor化して返す
    Args:
        batch: BoketeTextImageDatasetが返すデータ辞書を要素として持つリスト

    Returns:
        以下の辞書
        {
            "input_ids": バッチごとにベクトル化したtext shape=(N, 単語数),
            "attention_mask": paddingしたか否かを示すマスク shape=(N, 単語数),
            "input_modal": 前処理済み入力画像 shape=(N, 3, 224, 224),
            "modal_start_tokens": ベクトル化した単語の最初の要素 shape=(N, ),
            "modal_end_tokens": ベクトル化した単語の最後の要素 shape=(N, ),
            "labels": 正解ラベル shape=(N, )
        }
    """
    seq_len_list = [len(row['sentence']) for row in batch]
    bsz, max_seq_len = len(batch), max(seq_len_list)

    # 0埋めされたtxtの重みを無視するためのマスク(0: 無視, 1: 無視しない)
    mask_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)
    # 文書ベクトル用配列
    text_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)
    for i, (input_row, length) in enumerate(zip(batch, seq_len_list)):
        text_tensor[i, :length] = input_row['sentence']
        mask_tensor[i, :length] = 1
    img_tensor = torch.stack([row['image'] for row in batch])
    target_tensor = torch.stack([row['label'] for row in batch])
    img_start_token = torch.stack([row['image_start_token'] for row in batch])
    img_end_token = torch.stack([row['image_end_token'] for row in batch])
    return {
        'input_ids': text_tensor,
        'attention_mask': mask_tensor,
        'input_modal': img_tensor,
        'modal_start_tokens': img_start_token,
        'modal_end_tokens': img_end_token,
        'labels': target_tensor,
    }
