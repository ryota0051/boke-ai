import pandas as pd
import numpy as np

from src.data.columns import TARGET_COLUMN


def to_submission(
            submission_csv_path: str,
            y_pred: np.ndarray,
            dst: str
        ):
    """提出用データを作成して、保存する
    Args:
        submission_csv_path: 提出用csv雛形パス
        y_pred: 予測値
        dst: 保存先パス
    """
    submission_df = pd.read_csv(submission_csv_path)
    submission_df[TARGET_COLUMN] = y_pred
    submission_df.to_csv(dst, index=False)
