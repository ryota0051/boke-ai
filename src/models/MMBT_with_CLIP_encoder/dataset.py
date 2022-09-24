import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple
import torch
from torch.utils.data import Dataset
from transformers.models.bert_japanese.tokenization_bert_japanese import BertJapaneseTokenizer


class MMBTClipDsataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            tokenizer: BertJapaneseTokenizer,
            image_transform,
            desired_img_size=224,
            max_seq_len=48):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.desired_img_size = desired_img_size
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        # tokenの取得
        sentence = torch.tensor(
            self.tokenizer.encode(
                row['text'],
                max_length=self.max_seq_len,
                padding='max_length',
                truncation=True
            )
        )
        start_token, sentence, end_token = sentence[0], sentence[1:-1], sentence[-1]

        # 画像準備
        img = Image.open(row['img_path']).convert("RGB")
        sliced_imgs = slice_img(img, self.desired_img_size)
        sliced_imgs = [
            np.array(self.image_transform(sliced_img))
            for sliced_img in sliced_imgs
        ]
        img = resize_pad_img(img, self.desired_img_size)
        img = np.array(self.image_transform(img))
        sliced_imgs = [img] + sliced_imgs
        sliced_imgs = torch.from_numpy(np.array(sliced_imgs))

        # 正解ラベル取得
        label = torch.tensor(row['is_laugh'])
        return {
            'image_start_token': start_token,
            'image_end_token': end_token,
            'sentence': sentence,
            'image': sliced_imgs,
            'label': label
        }


def slice_img(img: Image, desired_size: int):
    old_img_size: Tuple[int, int] = img.size
    ratio = float(desired_size) / min(old_img_size)
    new_size = tuple([int(x * ratio) for x in old_img_size])
    new_img = img.resize(new_size, Image.ANTIALIAS)
    img_arr = np.array(new_img)
    imgs = []
    height, width = img_arr.shape[0], img_arr.shape[1]
    # 画像の長い方を基準に以下の3つに分ける
    # - 左(上)部分 shape = (desizred_size, desizred_size)
    # - 左右(上下) shape = (desizred_size, desizred_size)
    # - 右(下)部分 shape = (desizred_size, desizred_size)
    if height < width:
        middle = width // 2
        half = desired_size // 2
        # 画像左部分(w=0~desired_size)
        imgs.append(Image.fromarray(img_arr[:, :desired_size]))
        # 画像の左右均等に切り抜いた部分
        imgs.append(Image.fromarray(img_arr[:, middle-half:middle+half]))
        # 画像右部分(w=width - desired_size~width)
        imgs.append(Image.fromarray(img_arr[:, width-desired_size:width]))
    else:
        middle = height // 2
        half = desired_size // 2
        # 画像の上部分(h=0~desired_size)
        imgs.append(Image.fromarray(img_arr[:desired_size, :]))
        # 画像の上下均等に切り抜いた部分
        imgs.append(Image.fromarray(img_arr[middle-half:middle+half, :]))
        # 画像の下部分(h=height - desired_size~height)
        imgs.append(Image.fromarray(img_arr[height-desired_size:height, :]))
    return imgs


def resize_pad_img(img: Image, desired_size: int):
    old_img_size: Tuple[int, int] = img.size
    ratio = float(desired_size) / max(old_img_size)
    new_size = tuple([int(x * ratio) for x in old_img_size])
    img = img.resize(new_size, Image.ANTIALIAS)

    new_img = Image.new('RGB', (desired_size, desired_size))
    new_img.paste(img, (
        (desired_size - new_size[0]) // 2,
        (desired_size - new_size[1]) // 2
    ))
    return new_img
