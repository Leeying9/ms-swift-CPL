# Copyright (c) Alibaba, Inc. and its affiliates.
import warnings
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from .my_trl_cpo_trainer import MyTrlCPOTrainer
from ..mixin import SwiftMixin
from .rlhf_mixin import RLHFTrainerMixin

# 关键：删除底层（封装后）原始 __init__，与原 cpo_trainer.py 中 `del HFCPOTrainer.__init__` 作用一致
# del MyTrlCPOTrainer.__init__


class MyCPOTrainer(RLHFTrainerMixin, SwiftMixin, MyTrlCPOTrainer):
    """
    基于 CPO 的自定义 Trainer，支持MYCPO训练方法。
    
    MYCPO与CPO的唯一区别：
    - 在损失计算中使用权重参数w进行加权
    - 数据流程与原始CPO完全一致
    """

    def __init__(self, model: Optional[Union[PreTrainedModel, nn.Module, str]] = None, *_args, **kwargs):
        ref_model = kwargs.get('ref_model')
        assert ref_model is None, 'CPO/SimPO does not require a ref_model.'

        args = kwargs['args']
        self.label_smoothing = args.label_smoothing
        self.loss_type = args.loss_type
        self.cpo_alpha = args.cpo_alpha
        # 补充 beta（默认 0.1 由下游配置提供）
        self.beta = args.beta
        # XPO 三系数：从命令行透传
        self.lambda_intra = getattr(args, 'lambda_intra', 1.0)
        self.lambda_cross = getattr(args, 'lambda_cross', 1.0)
        self.lambda_bc = getattr(args, 'lambda_bc', 1.0)
        if args.loss_type == 'simpo':
            self.simpo_gamma = args.simpo_gamma
            if self.cpo_alpha > 0:
                warnings.warn(
                    'You are using CPO-SimPO method because you set a non-zero cpo_alpha. '
                    'This will result in the CPO-SimPO method '
                    '(https://github.com/fe1ixxu/CPO_SIMPO/tree/main). '
                    'If you want to use a pure SimPO method, please set cpo_alpha to 0.'
                )
        # 这里的 super().__init__ 不会调用原 trl.CPOTrainer 的 __init__（已被删除），
        # 只会走 RLHFTrainerMixin / SwiftMixin 的初始化链，与原实现保持一致。
        super().__init__(model, *_args, **kwargs)

   
