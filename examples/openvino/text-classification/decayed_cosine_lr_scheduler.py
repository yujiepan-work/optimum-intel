import bisect
from contextlib import contextmanager
import math
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch
import argparse

from optimum.intel.openvino import OVTrainer
import torch
from torch.optim.lr_scheduler import LambdaLR

import transformers
from transformers import TrainingArguments
from transformers.trainer_callback import TrainerControl, TrainerState


def get_cosine_with_decayed_hard_restarts_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    cycle_ratios=[1, 1, 1, 1],
    cycle_decays=[1, 1, 1, 1],
    last_epoch: int = -1,
):
    """TODO"""
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


def create_scheduler(self, cosine_cycle_ratios, cosine_cycle_decays,
                     num_training_steps: int, optimizer: torch.optim.Optimizer = None):
    if self.lr_scheduler is None:
        cosine_cycle_ratios = [float(r) for r in cosine_cycle_ratios.strip().split(",")]
        cosine_cycle_decays = [float(d) for d in cosine_cycle_decays.strip().split(",")]
        print(
            f"Using decayed cosine restarts: ratio={cosine_cycle_ratios}, decay={cosine_cycle_decays}."
        )
        self.lr_scheduler = get_cosine_with_decayed_hard_restarts_schedule_with_warmup(
            optimizer=self.optimizer if optimizer is None else optimizer,
            num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
            cycle_ratios=cosine_cycle_ratios,
            cycle_decays=cosine_cycle_decays,
        )
    return self.lr_scheduler


@contextmanager
def patch_decayed_cosine_lr_scheduler():
    original_argv = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--cosine_cycle_ratios', type=str, default=1)
    parser.add_argument('--cosine_cycle_decays', type=str, default=1)
    cosine_args, others = parser.parse_known_args()
    sys.argv = [original_argv[0]] + others

    original_create_scheduler = OVTrainer.create_scheduler

    def new_create_scheduler(self: OVTrainer, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if self.args.lr_scheduler_type == 'cosine_with_restarts':
            return create_scheduler(self, cosine_args.cosine_cycle_ratios, cosine_args.cosine_cycle_decays,
                                    num_training_steps, optimizer)
        return original_create_scheduler(self, num_training_steps, optimizer)

    print('Patching create_scheduler...')
    with patch(".".join([original_create_scheduler.__module__, original_create_scheduler.__qualname__]),
               new_create_scheduler):
        yield
    print('Exit patching create_scheduler...')
