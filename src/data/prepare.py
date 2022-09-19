from pathlib import Path

import pandas as pd


def load_base_df(
            csv_path_root,
            img_path_root,
            img_path_col='img_path',
            img_file_name_col='odai_photo_file_name',
            target_col='is_laugh'
        ):
    """学習に使用する提供済みcsvをロードする

    Args:
        csv_path_root: csvファイルのルートパスで、以下のファイルがあることを想定
            - train.csv
            - test.csv
            - sample_submission.csv
        img_path_root: 画像パスのルートパスで、以下の構成であることを想定
            ./
            |- train # 学習用の画像ディレクトリ
            |_ test # テスト用の画像ディレクトリ
        img_path_col: データフレームの画像パスを保持するカラム名
        img_file_name_col: 画像ファイル名カラム
        target_col: 目的変数カラム

    Returns:
        (train_df, test_df, submission_df)
    """
    train_img_path = Path(img_path_root) / 'train'
    test_img_path = Path(img_path_root) / 'test'

    train_df = pd.read_csv(str(Path(csv_path_root) / 'train.csv'))
    test_df = pd.read_csv(str(Path(csv_path_root) / 'test.csv'))
    submission_df = pd.read_csv(str(Path(csv_path_root) / 'sample_submission.csv'))

    # 学習用, test用データフレームに画像パスのカラム追加
    train_df[img_path_col] = (
        train_df[img_file_name_col]
        .apply(lambda x: str(train_img_path / x))
    )
    test_df[img_path_col] = (
        test_df[img_file_name_col]
        .apply(lambda x: str(test_img_path / x))
    )
    # test用データフレームのis_laughカラムを準備
    test_df[target_col] = 0
    return train_df, test_df, submission_df
