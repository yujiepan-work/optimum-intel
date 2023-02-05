import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from nncf.experimental.torch.sparsity.movement.scheduler import MovementPolynomialThresholdScheduler
from optimum.intel.openvino import OVTrainer


MovementPolynomialThresholdScheduler._calc_init_threshold_from_controller


@contextmanager
def patch_movement_init_threshold():
    original_argv = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument("--movement_init_sparsity", type=float, default=0.001)
    args, others = parser.parse_known_args()
    movement_init_sparsity = args.movement_init_sparsity
    sys.argv = [original_argv[0]] + others

    original_func = MovementPolynomialThresholdScheduler._calc_init_threshold_from_controller

    @torch.no_grad()
    def new_func(self, target_sparsity: float = movement_init_sparsity):
        print(f'Calculate threshold so that init sparisty is {movement_init_sparsity}')
        return original_func(self, movement_init_sparsity)

    print(f'Patching to set init sparisty as {movement_init_sparsity}.')
    patch_name = 'nncf.experimental.torch.sparsity.movement.scheduler.MovementPolynomialThresholdScheduler._calc_init_threshold_from_controller'
    with patch(patch_name, new_func):
        yield
    print(f'Exit patching to set init sparisty as {movement_init_sparsity}.')
