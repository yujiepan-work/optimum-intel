import bisect
from contextlib import contextmanager
import logging
import math
from typing import Dict, List
from unittest.mock import patch

from torch.optim.lr_scheduler import LambdaLR
import nncf
from nncf.common.logging.logger import set_log_level
from nncf.common.scopes import matches_any
from nncf.experimental.torch.search_building_blocks.search_blocks import BuildingBlockType
from nncf.experimental.torch.sparsity.movement.structured_mask_handler import (
    StructuredMaskContext,
    StructuredMaskContextGroup,
    StructuredMaskHandler,
)
from nncf.experimental.torch.sparsity.movement.structured_mask_strategy import (
    STRUCTURED_MASK_STRATEGY,
    BaseTransformerStructuredMaskStrategy,
    StructuredMaskRule,
    detect_supported_model_family,
)
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.sparsity.base_algo import SparseModuleInfo


def get_cosine_with_decayed_hard_restarts_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    cycle_ratios=[1, 1, 1, 1],
    cycle_decays=[1, 1, 1, 1],
    last_epoch: int = -1,
):
    assert len(cycle_ratios) == len(cycle_decays)
    total_cosine_steps = max(1, num_training_steps - num_warmup_steps)
    cycle_starting_steps = [0]
    _current_ratio = 0
    for ratio in cycle_ratios:
        _current_ratio += ratio
        starting_step = int(total_cosine_steps * _current_ratio / sum(cycle_ratios))
        cycle_starting_steps.append(starting_step)

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step >= num_training_steps:
            return 0.0
        cycle_idx = bisect.bisect_right(cycle_starting_steps, current_step - num_warmup_steps)
        steps_in_cycle = current_step - num_warmup_steps - cycle_starting_steps[cycle_idx - 1]
        total_steps_in_cycle = cycle_starting_steps[cycle_idx] - cycle_starting_steps[cycle_idx - 1]
        progress = float(steps_in_cycle) / float(total_steps_in_cycle)
        decay = float(cycle_decays[cycle_idx - 1])
        return decay * max(0.0, 0.5 * (1.0 + math.cos(math.pi * (progress % 1.0))))

    return LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)


_original_create_structured_mask_context_groups = StructuredMaskHandler._create_structured_mask_context_groups


class HuggingFacMobileBertStructuredMaskStrategy(BaseTransformerStructuredMaskStrategy):
    MHSA_Q: str = "{re}query"
    MHSA_K: str = "{re}key"
    MHSA_V: str = "{re}value"
    MHSA_O: str = "{re}MobileBertSelfOutput"
    FFN_I: str = "{re}MobileBertIntermediate"
    FFN_O: list = ["{re}FFNOutput", "{re}MobileBertOutput"]

    @classmethod
    def from_compressed_model(cls, compressed_model: NNCFNetwork):
        hidden_dim = compressed_model.nncf_module.mobilebert.config.true_hidden_size
        num_heads = compressed_model.nncf_module.mobilebert.config.num_attention_heads
        return cls(dim_per_head=hidden_dim // num_heads)

    @property
    def rules_by_group_type(self) -> Dict[str, List[StructuredMaskRule]]:
        config = {
            BuildingBlockType.MHSA: [
                StructuredMaskRule(
                    keywords=[self.MHSA_Q, self.MHSA_K, self.MHSA_V],
                    prune_by_row=True,
                    prune_grid=(self.dim_per_head, -1),
                ),
                StructuredMaskRule(
                    keywords=[self.MHSA_O],
                    prune_by_row=False,
                    prune_grid=(-1, self.dim_per_head),
                ),
            ],
            BuildingBlockType.FF: [
                StructuredMaskRule(
                    keywords=[self.FFN_I],
                    prune_by_row=True,
                    prune_grid=(1, -1),
                ),
                StructuredMaskRule(
                    keywords=self.FFN_O,
                    prune_by_row=False,
                    prune_grid=(-1, 1),
                ),
            ],
        }
        return config


@staticmethod
def _mobilebert_create_structured_mask_context_groups(
    compressed_model: NNCFNetwork,
    sparsified_module_info_list: List[SparseModuleInfo],
    rules_by_group_type: Dict[BuildingBlockType, List[StructuredMaskRule]],
) -> List[StructuredMaskContextGroup]:
    groups = _original_create_structured_mask_context_groups(
        compressed_model, sparsified_module_info_list, rules_by_group_type
    )
    if detect_supported_model_family(compressed_model) == "huggingface_mobilebert":
        # current get_building_blocks cannot detect MHSA in mobilebert
        # manual creation of MHSA group
        group_id = len(groups)
        sm_info_per_mhsa_group = dict()
        for sparse_info in sparsified_module_info_list:
            if "MobileBertAttention" in sparse_info.module_node_name:
                txblk = sparse_info.module_node_name.split("/")[4]
                if txblk not in sm_info_per_mhsa_group:
                    sm_info_per_mhsa_group[txblk] = []
                sm_info_per_mhsa_group[txblk].append(sparse_info)

        for group_id_offset, (group_key, group_sminfos) in enumerate(sm_info_per_mhsa_group.items()):
            assert len(group_sminfos) == 4, f"bug's alive, check mobilebert definition at {group_key}"
            group_type = BuildingBlockType.MHSA
            ctxes = []
            for minfo in group_sminfos:
                for rule in rules_by_group_type[group_type]:
                    if matches_any(minfo.module_node_name, rule.keywords):
                        ctx = StructuredMaskContext(
                            minfo.operand, minfo.module_node_name, rule.prune_grid, rule.prune_by_row
                        )
                        ctxes.append(ctx)
                        break
                else:
                    raise ValueError(f"No structured mask rule for {minfo.module_node_name}.")
            groups.append(StructuredMaskContextGroup(group_id + group_id_offset, group_type, ctxes))
    # for group in groups:
    #     print(group)
    return groups


@contextmanager
def nncf_patch_for_mobilebert(nncf_log_level=logging.INFO):
    print("Patching mobilebert support...")
    original_func = _original_create_structured_mask_context_groups
    func_import_name = ".".join([original_func.__module__, original_func.__qualname__])
    strategy_name = "huggingface_mobilebert"
    STRUCTURED_MASK_STRATEGY.register(strategy_name)(HuggingFacMobileBertStructuredMaskStrategy)
    set_log_level(nncf_log_level)
    try:
        with patch(func_import_name, _mobilebert_create_structured_mask_context_groups) as patcher:
            yield patcher
    finally:
        del STRUCTURED_MASK_STRATEGY.registry_dict[strategy_name]
        print("Exit patching mobilebert support.")
