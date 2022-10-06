from typing import List, Dict, Union
from pathlib import Path
import os

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


class Create5FoldDataFrame():
    def __init__(
                self,
                fold_root_dir: str,
                base_train_csv_path: str,
                train_feature_csv_path_list: List[str],
                test_feature_csv_path_list: List[str],
            ):
        self.fold_root_dir = fold_root_dir
        self.base_train_csv_path = base_train_csv_path
        self.train_feature_csv_path_list = train_feature_csv_path_list
        self.test_feature_csv_path_list = test_feature_csv_path_list

    def __call__(self) -> Dict[str, Dict[str, Dict[str, Union[pd.DataFrame, pd.Series]]]]:
        # 1. 5foldのデータからデータフレーム辞書作成
        fold_vector_dict = self._create_fold_vector_dict(self.fold_root_dir)
        # 2. 5foldのデータに特徴量をつける
        dataset_dict = self._add_features2fold_vector_dict(
            fold_vector_dict,
            self.train_feature_csv_path_list,
            self.test_feature_csv_path_list
        )
        # 3. ラベルをつける
        dataset_dict = self._add_label2train_and_valid_df(
            dataset_dict, self.base_train_csv_path)
        # 4. データをラベルと特徴量ベクトルに分ける
        dataset_dict = self._split_dataset2X_and_y(dataset_dict)
        return dataset_dict

    @staticmethod
    def _create_fold_vector_dict(fold_root_dir: str) -> Dict[str, Dict[str, pd.DataFrame]]:
        """k-foldの結果(MMBTの中間層ベクトルを想定)が保存されているルートディレクトリ配下のtrain.csv, valid.csv, test.csvを読み込み、辞書に格納する。
        Args:
            fold_root_dir: k-foldの結果が保存されているディレクトリで、構成は下記を想定
                ./
                |-fold_1
                |   |-train.csv
                |   |-valid.csv
                |   |_test.csv
                :
                |_fold_N
        Returns:
            下記の辞書
            {
                'fold_1': {
                    'train': 学習用データフレーム,
                    'valid': 検証用データフレーム,
                    'test': テストデータフレーム,
                },
                ...
                'fold_N': fold_1と同じ構成
            }
        """
        fold_dir_list = os.listdir(fold_root_dir)
        result = {fold_dir_name: {'train': None, 'valid': None, 'test': None} for fold_dir_name in fold_dir_list}
        for fold_dir_name in fold_dir_list:
            dir_path = Path(fold_root_dir) / fold_dir_name 
            train_csv_path, valid_csv_path, test_csv_path = str(dir_path / 'train.csv'), str(dir_path / 'valid.csv'), str(dir_path / 'test.csv')
            result[fold_dir_name]['train'] = pd.read_csv(train_csv_path)
            result[fold_dir_name]['valid'] = pd.read_csv(valid_csv_path)
            result[fold_dir_name]['test'] = pd.read_csv(test_csv_path)
        return result

    @staticmethod
    def _add_features2fold_vector_dict(
                fold_vector_dict: Dict[str, Dict[str, pd.DataFrame]],
                train_feature_csv_path_list: List[str],
                test_feature_csv_path_list: List[str]
            ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """MMBTの中間層の値にその他の特徴量を
        """
        train_feature_df_list = [pd.read_csv(train_feature_csv_path) for train_feature_csv_path in train_feature_csv_path_list]
        test_feature_df_list = [pd.read_csv(test_feature_csv_path) for test_feature_csv_path in test_feature_csv_path_list]
        for fold_name in fold_vector_dict:
            for train_feature_df in train_feature_df_list:
                fold_vector_dict[fold_name]['train'] = fold_vector_dict[fold_name]['train'].merge(train_feature_df, on='id')
                fold_vector_dict[fold_name]['valid'] = fold_vector_dict[fold_name]['valid'].merge(train_feature_df, on='id')
            for test_feature_df in test_feature_df_list:
                fold_vector_dict[fold_name]['test'] = fold_vector_dict[fold_name]['test'].merge(test_feature_df, on='id')
        return fold_vector_dict

    @staticmethod
    def _add_label2train_and_valid_df(
                fold_vector_dict: Dict[str, Dict[str, pd.DataFrame]],
                base_train_csv_path: str
            ) -> Dict[str, Dict[str, pd.DataFrame]]:
        base_train_df = pd.read_csv(base_train_csv_path, usecols=['id', 'is_laugh'])
        for fold_name in fold_vector_dict:
            fold_vector_dict[fold_name]['train'] = fold_vector_dict[fold_name]['train'].merge(base_train_df, on='id')
            fold_vector_dict[fold_name]['valid'] = fold_vector_dict[fold_name]['valid'].merge(base_train_df, on='id')
        return fold_vector_dict

    @staticmethod
    def _split_dataset2X_and_y(
                dataset_dict: Dict[str, Dict[str, pd.DataFrame]]
            ) -> Dict[str, Dict[str, Dict[str, Union[pd.DataFrame, pd.Series]]]]:
        for fold_name in dataset_dict:
            # まずidを落としておく
            for phase in dataset_dict[fold_name]:
                dataset_dict[fold_name][phase] = dataset_dict[fold_name][phase].drop(columns=['id'])
            for phase in dataset_dict[fold_name]:
                if phase == 'test':
                    dataset_dict[fold_name][phase] = {'X': dataset_dict[fold_name][phase]}
                else:
                    dataset_dict[fold_name][phase] = {
                        'X': dataset_dict[fold_name][phase].drop(columns=['is_laugh']),
                        'y': dataset_dict[fold_name][phase]['is_laugh']
                    }
        return dataset_dict
