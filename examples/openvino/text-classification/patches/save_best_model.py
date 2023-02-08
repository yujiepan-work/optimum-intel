import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

from optimum.intel.openvino import OVTrainer

import transformers
from transformers import TrainingArguments
from transformers.trainer_callback import TrainerControl, TrainerState


class SaveBestModelCallback(transformers.trainer_callback.TrainerCallback):
    def __init__(self, save_best_model_after_epoch: int = -1, metric_for_best_model="accuracy", save_best_model_after_sparsity=0.8) -> None:
        super().__init__()
        self.best_metric = -1.0
        self.save_best_model_after_epoch = save_best_model_after_epoch
        self.metric_for_best_model = metric_for_best_model
        self.save_best_model_after_sparsity = save_best_model_after_sparsity

    def register_trainer(self, trainer: transformers.Trainer):
        self.trainer = trainer

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_save(args, state, control, **kwargs)
        last_eval_metric = None
        last_epoch = None
        last_sparisty = None
        for log in state.log_history:
            if f"eval_{self.metric_for_best_model}" in log:
                last_eval_metric = log[f"eval_{self.metric_for_best_model}"]
            if "epoch" in log:
                last_epoch = log["epoch"]
            if "compression/magnitude_sparsity/sparsity_level_for_sparsified_layers" in log:
                last_sparisty = log["compression/magnitude_sparsity/sparsity_level_for_sparsified_layers"]
        if last_eval_metric is None or self.best_metric >= last_eval_metric:
            print("Skip saving due to not best metric")
            return control
        if last_sparisty is None or last_sparisty < self.save_best_model_after_sparsity:
            print(f"Skip saving due to less than {self.save_best_model_after_sparsity} sparsity")
            return control
        if last_epoch is None or last_epoch <= float(self.save_best_model_after_epoch):
            print("Skip saving due to too early epoch")
            return control

        self.best_metric = last_eval_metric
        folder = Path(args.output_dir, "best_model").absolute()
        if folder.exists():
            shutil.rmtree(folder.resolve().as_posix())
        folder.mkdir(parents=True, exist_ok=True)
        if self.trainer.is_world_process_zero():
            self.trainer.save_model(output_dir=folder.as_posix(), _internal_call=True)
            state.save_to_json(Path(folder, "trainer_state.json").as_posix())
            with open(Path(folder, "best_metric.json"), "w", encoding="utf-8") as f:
                json.dump({"best_metric": self.best_metric}, f)
        return control


@contextmanager
def patch_save_best_model(metric_for_best_model="accuracy"):
    original_argv = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_best_model_after_epoch", type=int, default=7)
    parser.add_argument("--save_best_model_after_sparsity", type=float, default=0.8)
    args, others = parser.parse_known_args()
    sys.argv = [original_argv[0]] + others
    save_best_model_after_epoch = args.save_best_model_after_epoch
    save_best_model_after_sparsity = args.save_best_model_after_sparsity

    original_init = OVTrainer.__init__

    def new_init(self, *args, **kwargs):
        save_best_callback = SaveBestModelCallback(
            save_best_model_after_epoch,
            metric_for_best_model,
            save_best_model_after_sparsity
        )
        callbacks = kwargs.pop("callbacks", None) or []
        callbacks.append(save_best_callback)
        original_init(self, *args, callbacks=callbacks, **kwargs)
        save_best_callback.register_trainer(self)

    print(f"Patching save best model callback after epoch {save_best_model_after_epoch} "
          f"after sparsity {save_best_model_after_sparsity}...")
    patch_name = 'optimum.intel.openvino.OVTrainer.__init__'
    with patch(patch_name, new_init):
        yield
    print("Exit patching save best model callback...")
