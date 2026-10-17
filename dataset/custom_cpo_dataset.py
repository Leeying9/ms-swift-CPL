from typing import Any, Dict, Optional

from swift.llm.dataset.register import DatasetMeta, register_dataset
from swift.llm.dataset.preprocessor import RowPreprocessor


class MyCPOQuadruplePreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """处理 XPO 成对偏好数据（win/lose）。

        期望输入行（jsonl 的一行）：
        {
            "x_win": str,
            "y_win": str,
            "x_lose": str,
            "y_lose": str,
            "mission": Optional[str]
        }

        输出：
        {
            'messages': [
                {'role': 'user', 'content': x_win},
                {'role': 'assistant', 'content': y_win}
            ],
            'rejected_messages': [
                {'role': 'user', 'content': x_lose},
                {'role': 'assistant', 'content': y_lose}
            ],
            'mission': mission
        }
        """

        x_win = row.get('x_win')
        y_win = row.get('y_win')
        x_lose = row.get('x_lose')
        y_lose = row.get('y_lose')
        mission = row.get('mission')

        # 基本校验
        required_fields = [x_win, y_win, x_lose, y_lose]
        if not all(isinstance(v, str) for v in required_fields):
            return None
        if not all(len(v) > 0 for v in required_fields):
            return None

        messages = [
            {"role": "user", "content": x_win},
            {"role": "assistant", "content": y_win},
        ]
        rejected_messages = [
            {"role": "user", "content": x_lose},
            {"role": "assistant", "content": y_lose},
        ]

        return {
            'messages': messages,
            'rejected_messages': rejected_messages,
            'mission': mission,
        }


# 注册数据集，训练时它可通过 --dataset custom_cpo_dataset 指定数据集名称
# CPL 
register_dataset(
    DatasetMeta(
        dataset_name='en-de-2w-weighted-intra-and-cross',
    dataset_path='dataset/cpl/en-de/intra-and-cross.jsonl',
        preprocess_func=MyCPOQuadruplePreprocessor(),
        tags=['rlhf', 'cpo', 'custom'],
    ))

#CPL w/o cross
register_dataset(
    DatasetMeta(
        dataset_name='en-de-2w-weighted-intra',
    dataset_path='dataset/cpl/en-de/intra.jsonl',
        preprocess_func=MyCPOQuadruplePreprocessor(),
        tags=['rlhf', 'cpo', 'custom'],
    ))

# CPL w/o intra
register_dataset(
    DatasetMeta(
        dataset_name='en-de-2w-weighted-cross',
    dataset_path='dataset/cpl/en-de/cross.jsonl',
        preprocess_func=MyCPOQuadruplePreprocessor(),
        tags=['rlhf', 'cpo', 'custom'],
    ))