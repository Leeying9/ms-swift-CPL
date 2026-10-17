# 作为对 trl.CPOTrainer 的轻量封装，避免上层直接依赖 trl 包。
from trl import CPOTrainer as _HFCPOTrainer
from typing import Any, Callable, Literal, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    is_comet_available,
    is_wandb_available,
)
from torch import autocast
from contextlib import nullcontext

# 备份原始 __init__（如果以后需要恢复或调试，可以再用）
_original_trl_cpo_init = getattr(_HFCPOTrainer, '__init__', None)

# 删除原始 __init__，以便上层混入链 (RLHFTrainerMixin / SwiftMixin) 接管初始化流程
if hasattr(_HFCPOTrainer, '__init__'):
    del _HFCPOTrainer.__init__


class MyTrlCPOTrainer(_HFCPOTrainer):
    """
    轻量中转。不新增行为。
    原 trl.CPOTrainer.__init__ 已被移除，上层自定义 Trainer 通过 super()
    时不会进入原始 trl 初始化。
    如需暂时恢复，可在运行时执行:
        setattr(_HFCPOTrainer, '__init__', _original_trl_cpo_init)

    训练目标：纯 CPO
        L = - log σ(β (log π(chosen) - log π(rejected))) - log π(chosen)
    其中 chosen 对应更优样本（由上游根据 w 选择），rejected 为次优样本。
    """
    
    
    
    def cpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        计算一个batch的loss
        计算逐样本的 l_pair：
            l_pair = -log σ[ β (log P(y_win|x_win) - log P(y_lose|x_lose)) ]

        返回 (l_pair_per_example, chosen_rewards, rejected_rewards)。NLL 在外部计算。
        """
        device = self.accelerator.device
        logits = (policy_chosen_logps - policy_rejected_logps).to(device)
        beta = getattr(self.args, 'beta', 0.1)
        self.beta = beta

        l_pair = -F.logsigmoid(self.beta * logits)

        chosen_rewards = self.beta * (policy_chosen_logps.to(device)).detach()
        rejected_rewards = self.beta * (policy_rejected_logps.to(device)).detach()

        return l_pair, chosen_rewards, rejected_rewards

    def concatenated_forward(
        self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]]
    ) -> tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
    
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "decoder_input_ids": self._shift_right(concatenated_batch["concatenated_labels"]),
            }
            if self.is_encoder_decoder
            else {}
        )

        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        )
        # all_logits[i, t, :] 就是模型在第 i 个样本上、位置 t 处预测“下一个 token 是词表中每个 token 的概率分布”的未归一化对数值（logits）
        all_logits = outputs.logits

        def cross_entropy_loss_per_sample(logits, labels):
            if not self.is_encoder_decoder:
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            # Per-token loss without reduction
            loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=self.label_pad_token_id)
            batch_size, seq_len, vocab = logits.shape
            logits = logits.view(-1, vocab)
            labels = labels.view(-1)
            labels = labels.to(logits.device)
            per_token = loss_fct(logits, labels).view(batch_size, seq_len)
            valid_mask = (labels.view(batch_size, seq_len) != self.label_pad_token_id)
            denom = valid_mask.sum(dim=1).clamp_min(1)
            per_sample = (per_token * valid_mask).sum(dim=1) / denom
            return per_sample

        labels = concatenated_batch["concatenated_labels"].clone()

        if self.cpo_alpha == 0:
            nll_loss_1 = torch.zeros((len_chosen,), device=self.accelerator.device)
            nll_loss_2 = torch.zeros((labels.shape[0] - len_chosen,), device=self.accelerator.device)
        else:
            # Per-sample NLL for chosen (y1|x1)
            nll_loss_1 = cross_entropy_loss_per_sample(all_logits[:len_chosen], labels[:len_chosen])
            # Per-sample NLL for rejected (y2|x2)
            nll_loss_2 = cross_entropy_loss_per_sample(all_logits[len_chosen:], labels[len_chosen:])

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=self.loss_type in ["ipo", "simpo"],
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]


        if self.aux_loss_enabled:
            return (
                chosen_logps,
                rejected_logps,
                chosen_logits,
                rejected_logits,
                nll_loss_1,
                nll_loss_2,
                outputs.aux_loss,
            )

        return (
            chosen_logps,
            rejected_logps,
            chosen_logits,
            rejected_logits,
            nll_loss_1,
            nll_loss_2,
        )
    

    def get_batch_loss_metrics(
            self,
            model,
            batch: dict[str, Union[list, torch.LongTensor]],
            train_eval: Literal["train", "eval"] = "train",
        ):
            """Compute the CPO/MyCPO loss and other metrics for the given batch of inputs for train or test."""


            metrics = {}

            forward_output = self.concatenated_forward(model, batch)


            (
                policy_chosen_logps,
                policy_rejected_logps,
                policy_chosen_logits,
                policy_rejected_logits,
                policy_nll_loss_1,
                policy_nll_loss_2,
                *maybe_aux,
            ) = forward_output
            if self.aux_loss_enabled and len(maybe_aux) > 0:
                aux_loss = maybe_aux[0]

            # 计算逐样本的 l_pair（不再使用 w）
            l_pair, chosen_rewards, rejected_rewards = self.cpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
            )

            # lambda_nll = getattr(self.args, 'lambda_nll', 1.0)
            # mission → mission_intra/mission_cross（严格要求存在且取值为'intra'或'cross'）
            missions = batch.get('mission')
            if missions is None:
                raise ValueError("Missing required field 'mission' in batch. Expected per-sample 'intra' or 'cross'.")
            if not isinstance(missions, (list, tuple)):
                missions = [missions]
            if len(missions) != l_pair.shape[0]:
                raise ValueError(f"Length mismatch: got mission len={len(missions)} but batch size={l_pair.shape[0]}.")

            intra_flags = []
            cross_flags = []
            for i, m in enumerate(missions):
                if m == 'intra':
                    intra_flags.append(1.0)
                    cross_flags.append(0.0)
                elif m == 'cross':
                    intra_flags.append(0.0)
                    cross_flags.append(1.0)
                else:
                    raise ValueError(f"Invalid mission value at index {i}: {m}. Expected 'intra' or 'cross'.")
            mission_intra = torch.tensor(intra_flags, device=policy_chosen_logps.device, dtype=l_pair.dtype)
            mission_cross = torch.tensor(cross_flags, device=policy_chosen_logps.device, dtype=l_pair.dtype)

            # 系数（命令行透传）
            lambda_intra = getattr(self.args, 'lambda_intra', 1.0)
            lambda_cross = getattr(self.args, 'lambda_cross', 1.0)
            lambda_bc = getattr(self.args, 'lambda_bc', 1.0)

            # Loss = mission_intra * λ_intra * l_pair + mission_cross * λ_cross * l_pair + λ_bc * NLL
            # all λ default to 1.0 ,and should not be specified
            pair_weight = mission_intra * lambda_intra + mission_cross * lambda_cross
            pair_term = (pair_weight * l_pair).mean()
            nll_term = lambda_bc * policy_nll_loss_1.mean()

            loss = pair_term + nll_term
            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            prefix = "eval_" if train_eval == "eval" else ""
            metrics[f"{prefix}rewards/chosen"] = self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
            metrics[f"{prefix}rewards/rejected"] = self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
            metrics[f"{prefix}rewards/accuracies"] = self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
            metrics[f"{prefix}rewards/margins"] = (
                self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards).mean().item()
            )
            metrics[f"{prefix}logps/rejected"] = (
                self.accelerator.gather_for_metrics(policy_rejected_logps).detach().mean().item()
            )
            metrics[f"{prefix}logps/chosen"] = (
                self.accelerator.gather_for_metrics(policy_chosen_logps).detach().mean().item()
            )
            metrics[f"{prefix}logits/rejected"] = (
                self.accelerator.gather_for_metrics(policy_rejected_logits.detach().mean()).mean().item()
            )
            metrics[f"{prefix}logits/chosen"] = (
                self.accelerator.gather_for_metrics(policy_chosen_logits.detach().mean()).mean().item()
            )
            metrics[f"{prefix}nll_loss/chosen"] = (
                self.accelerator.gather_for_metrics(policy_nll_loss_1).detach().mean().item()
            )
            metrics[f"{prefix}nll_loss/rejected"] = (
                self.accelerator.gather_for_metrics(policy_nll_loss_2).detach().mean().item()
            )
            metrics[f"{prefix}l_pair"] = self.accelerator.gather_for_metrics(l_pair).detach().mean().item()
            metrics[f"{prefix}pair_term"] = pair_term.item()
            metrics[f"{prefix}nll_term"] = nll_term.item()

            if self.aux_loss_enabled:
                loss += self.aux_loss_coef * aux_loss

            return loss, metrics

    
    def compute_loss(
            self,
            model: Union[PreTrainedModel, nn.Module],
            inputs: dict[str, Union[torch.Tensor, Any]],
            return_outputs=False,
            num_items_in_batch=None,
        ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:

        compute_loss_context_manager = (
            autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )

        with compute_loss_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # force log the metrics
        self.store_metrics(metrics, train_eval="train")
        
        if return_outputs:
            return (loss, metrics)
        return loss

