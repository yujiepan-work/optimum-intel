import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
from unittest.mock import patch

from optimum.intel.openvino import OVTrainer
import torch


@contextmanager
def patch_load_pretrained_pytorch_model():
    original_argv = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_pytorch_model", type=str, default='')
    args, others = parser.parse_known_args()
    sys.argv = [original_argv[0]] + others

    model_path = Path(str(args.pretrained_pytorch_model))
    if not model_path.is_file():
        print(f"Does not load pretrained model.")
        yield
    else:
        print(f"Patching to load pretrained model from {model_path} and freeze the controller...")
        original_init = OVTrainer.__init__
        patch_name = 'optimum.intel.openvino.OVTrainer.__init__'

        def new_init(self: OVTrainer, *args, **kwargs):
            original_init(self, *args, **kwargs)
            state_dict = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=True)
            print(f'Loaded {model_path}.')
            self.compression_controller.freeze()

        with patch(patch_name, new_init):
            yield

    print("Exit patching create_scheduler...")
