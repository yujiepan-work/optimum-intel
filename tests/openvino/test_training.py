#  Copyright 2023 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import copy
import tempfile
import unittest
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    default_data_collator,
)
from transformers.utils import WEIGHTS_NAME

import evaluate
from nncf.torch.dynamic_graph.graph_tracer import create_mock_tensor
from optimum.intel.openvino import OVTrainingArguments
from optimum.intel.openvino.configuration import DEFAULT_QUANTIZATION_CONFIG, OVConfig
from optimum.intel.openvino.modeling import OVModelForQuestionAnswering, OVModelForSequenceClassification
from optimum.intel.openvino.quantization import OVQuantizer
from optimum.intel.openvino.trainer import OVTrainer
from parameterized import param, parameterized


CUSTOMIZED_QUANTIZATION_CONFIG = {
    "algorithm": "quantization",
    "initializer": {
        "range": {
            "num_init_samples": 16,
            "type": "percentile",
            "params": {"min_percentile": 0.01, "max_percentile": 99.99},
        },
        "batchnorm_adaptation": {"num_bn_adaptation_samples": 4},
    },
    "scope_overrides": {"activations": {"{re}.*matmul_0": {"mode": "symmetric"}}},
    "ignored_scopes": [],
}

MOVEMENT_SPARSITY_CONFIG_FOR_BERT = {
    "algorithm": "movement_sparsity",
    "params": {
        "warmup_start_epoch": 1,
        "warmup_end_epoch": 2,
        "importance_regularization_factor": 1.0,
        "enable_structured_masking": True,
    },
    "sparse_structure_by_scopes": [
        {"mode": "block", "sparse_factors": [32, 32], "target_scopes": "{re}.*BertAttention.*"},
        {"mode": "per_dim", "axis": 0, "target_scopes": "{re}.*BertIntermediate.*"},
        {"mode": "per_dim", "axis": 1, "target_scopes": "{re}.*BertOutput.*"},
    ],
    "ignored_scopes": ["{re}.*NNCFEmbedding", "{re}.*LayerNorm.*", "{re}.*pooler.*", "{re}.*classifier.*"],
}


def generate_mock_tokens(input_infos):
    mock_tokens = dict()
    for info in input_infos:
        single_batch_info = copy.copy(info)
        input_shape = tuple([1] + list(info.shape)[1:])
        single_batch_info.shape = input_shape
        mock_tokens[info.keyword] = create_mock_tensor(single_batch_info, "cpu")
    return mock_tokens


@dataclass
class OVTrainerTestDescription:
    model_id: str
    teacher_model_id: Optional[str] = None
    nncf_compression_config: Union[List[Dict], Dict] = field(default_factory=list)
    num_fake_quantize: int = 0
    num_int8: int = 0
    num_binary_masks: int = 0
    compression_metrics: List[str] = field(default_factory=list)


class OVTrainerTest(unittest.TestCase):
    TRAIN_DESCRIPTIONS = [
        param(
            model_id="hf-internal-testing/tiny-bert",
            teacher_model_id="hf-internal-testing/tiny-bert",
            nncf_compression_config=[],
            compression_metrics=["distillation_loss"],
        ),
        param(
            model_id="hf-internal-testing/tiny-bert",
            nncf_compression_config=DEFAULT_QUANTIZATION_CONFIG,
            expected_fake_quantize=19,
            expected_int8=14,
        ),
        param(
            model_id="hf-internal-testing/tiny-bert",
            teacher_model_id="hf-internal-testing/tiny-bert",
            nncf_compression_config=DEFAULT_QUANTIZATION_CONFIG,
            expected_fake_quantize=19,
            expected_int8=14,
            compression_metrics=["distillation_loss"],
        ),
        param(
            model_id="hf-internal-testing/tiny-bert",
            nncf_compression_config=CUSTOMIZED_QUANTIZATION_CONFIG,
            expected_fake_quantize=31,
            expected_int8=17,
        ),
        param(
            model_id="hf-internal-testing/tiny-bert",
            teacher_model_id="hf-internal-testing/tiny-bert",
            nncf_compression_config=CUSTOMIZED_QUANTIZATION_CONFIG,
            expected_fake_quantize=31,
            expected_int8=17,
            compression_metrics=["distillation_loss"],
        ),
        param(
            model_id="hf-internal-testing/tiny-bert",
            nncf_compression_config=MOVEMENT_SPARSITY_CONFIG_FOR_BERT,
            expected_binary_masks=24,
            compression_metrics=["compression_loss"],
        ),
        param(
            model_id="hf-internal-testing/tiny-bert",
            teacher_model_id="hf-internal-testing/tiny-bert",
            nncf_compression_config=MOVEMENT_SPARSITY_CONFIG_FOR_BERT,
            expected_binary_masks=24,
            compression_metrics=["compression_loss", "distillation_loss"],
        ),
        param(
            model_id="hf-internal-testing/tiny-bert",
            nncf_compression_config=[DEFAULT_QUANTIZATION_CONFIG, MOVEMENT_SPARSITY_CONFIG_FOR_BERT],
            expected_fake_quantize=19,
            expected_int8=14,
            expected_binary_masks=24,
            compression_metrics=["compression_loss"],
        ),
        param(
            model_id="hf-internal-testing/tiny-bert",
            nncf_compression_config=[CUSTOMIZED_QUANTIZATION_CONFIG, MOVEMENT_SPARSITY_CONFIG_FOR_BERT],
            expected_fake_quantize=31,
            expected_int8=17,
            expected_binary_masks=24,
            compression_metrics=["compression_loss"],
        ),
        param(
            model_id="hf-internal-testing/tiny-bert",
            teacher_model_id="hf-internal-testing/tiny-bert",
            nncf_compression_config=[DEFAULT_QUANTIZATION_CONFIG, MOVEMENT_SPARSITY_CONFIG_FOR_BERT],
            expected_fake_quantize=19,
            expected_int8=14,
            expected_binary_masks=24,
            compression_metrics=["compression_loss", "distillation_loss"],
        ),
        param(
            model_id="hf-internal-testing/tiny-bert",
            teacher_model_id="hf-internal-testing/tiny-bert",
            nncf_compression_config=[CUSTOMIZED_QUANTIZATION_CONFIG, MOVEMENT_SPARSITY_CONFIG_FOR_BERT],
            expected_fake_quantize=31,
            expected_int8=17,
            expected_binary_masks=24,
            compression_metrics=["compression_loss", "distillation_loss"],
        ),
    ]

    @parameterized.expand(TRAIN_DESCRIPTIONS[:1])
    def test_training(
        self,
        model_id,
        teacher_model_id=None,
        nncf_compression_config=[],
        expected_fake_quantize=0,
        expected_int8=0,
        expected_binary_masks=0,
        compression_metrics=[],
    ):
        ov_config = OVConfig()
        ov_config.compression = nncf_compression_config
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        teacher_model = None
        if teacher_model_id:
            teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_model_id)

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = self.build_sst2_trainer(
                output_dir=tmp_dir,
                ov_config=ov_config,
                tokenizer=tokenizer,
                model=model,
                teacher_model=teacher_model,
            )
            trainer.train()
            trainer.save_model()

            for metric in compression_metrics:
                self.assertIn(metric, trainer.compression_metrics)

            ovmodel = OVModelForSequenceClassification.from_pretrained(tmp_dir)
            num_fake_quantize, num_int8 = self.count_quantization_op_number(ovmodel)
            self.assertEqual(expected_fake_quantize, num_fake_quantize)
            self.assertEqual(expected_int8, num_int8)

            state_dict = torch.load(Path(tmp_dir, WEIGHTS_NAME), map_location="cpu")
            num_binary_masks = sum(key.endswith("_binary_mask") for key in state_dict)
            self.assertEqual(expected_binary_masks, num_binary_masks)

            tokens = generate_mock_tokens(trainer.model.input_infos)
            outputs = ovmodel(**tokens)
            self.assertTrue("logits" in outputs)

            model_resume = AutoModelForSequenceClassification.from_pretrained(tmp_dir)
            trainer_resume = self.build_sst2_trainer(
                tmp_dir,
                ov_config=ov_config,
                tokenizer=tokenizer,
                model=model_resume,
                teacher_model=teacher_model,
            )
            trainer_resume.evaluate()

    def build_sst2_trainer(
        self,
        output_dir,
        ov_config,
        tokenizer,
        model,
        teacher_model=None,
        num_train_epochs=3,
        do_train=True,
        do_eval=True,
        **training_args
    ):
        def tokenizer_fn(examples):
            return tokenizer(examples["sentence"], padding="max_length", max_length=128)

        dataset = load_dataset("glue", "sst2")
        dataset = dataset.map(tokenizer_fn, batched=True)
        train_dataset = dataset["train"].select(range(16))
        eval_dataset = dataset["validation"].select(range(16))
        metric = evaluate.load("glue", "sst2")

        def compute_metrics(p):
            return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

        args = dict(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            do_train=do_train,
            do_eval=do_eval,
            logging_steps=1,
        )
        args.update(training_args)
        ov_config.log_dir = output_dir

        trainer = OVTrainer(
            model=model,
            teacher_model=teacher_model,
            args=OVTrainingArguments(**args),
            ov_config=ov_config,
            task="sequence-classification",
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
        )
        return trainer

    def count_quantization_op_number(self, ovmodel):
        num_fake_quantize = 0
        num_int8 = 0
        for elem in ovmodel.model.get_ops():
            if "FakeQuantize" in elem.name:
                num_fake_quantize += 1
            if "8" in elem.get_element_type().get_type_name():
                num_int8 += 1
        return num_fake_quantize, num_int8

    def check_mask(self, ovmodel):
        return True
