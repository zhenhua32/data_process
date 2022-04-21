#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for sequence classification."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os

import numpy as np
from transformers import set_seed, create_optimizer

import tensorflow as tf  # noqa: E402

# import tensorflow.python.keras as keras
import tensorflow.keras as keras

# 不使用 from transformers import TFAutoModelForSequenceClassification
# 而是用自己重新定义的分类模型, 修改了模型输出
from model_helper import TFBertForSequenceClassification
from param_helper import load_params, SavePretrainedCallback
from dataset_helper import load_data_layer, covert_to_tf_dataset

logger = logging.getLogger(__name__)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Reduce the amount of console output from TF


def main():
    # 没支持回归, 所以 is_regression 就没用了
    model_args, data_args, training_args, checkpoint = load_params()
    datasets, config, is_regression, data_collator, non_label_column_names = load_data_layer(
        model_args, data_args, training_args, checkpoint
    )
    tf_data = covert_to_tf_dataset(datasets, data_args, training_args, data_collator, non_label_column_names, True)

    with training_args.strategy.scope():
        # region Load pretrained model
        # Set seed before initializing model
        set_seed(training_args.seed)
        #
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        if checkpoint is None:
            model_path = model_args.model_name_or_path
        else:
            model_path = checkpoint
        model = TFBertForSequenceClassification.from_pretrained(
            model_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model.summary()
        # endregion

        # region Optimizer, loss and compilation
        optimizer = keras.optimizers.Adam(
            learning_rate=training_args.learning_rate,
            beta_1=training_args.adam_beta1,
            beta_2=training_args.adam_beta2,
            epsilon=training_args.adam_epsilon,
            clipnorm=training_args.max_grad_norm,
        )
        # 定义两个损失函数
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        loss_fn2 = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ["accuracy"]
        # model.compile(optimizer=optimizer, loss={"output_1": loss_fn, "output_2": loss_fn2}, metrics=metrics)
        model.compile(optimizer=optimizer, loss=[loss_fn, loss_fn2], metrics=metrics)
        # endregion

        # region Training and validation
        if tf_data["train"] is not None:
            callbacks = [SavePretrainedCallback(output_dir=training_args.output_dir)]
            model.fit(
                tf_data["train"],
                validation_data=tf_data["validation"],
                epochs=int(training_args.num_train_epochs),
                callbacks=callbacks,
            )
        elif tf_data["validation"] is not None:
            # If there's a validation dataset but no training set, just evaluate the metrics
            logger.info("Computing metrics on validation data...")
            if is_regression:
                loss = model.evaluate(tf_data["validation"])
                logger.info(f"Loss: {loss:.5f}")
            else:
                loss, accuracy = model.evaluate(tf_data["validation"])
                logger.info(f"Loss: {loss:.5f}, Accuracy: {accuracy * 100:.4f}%")
        # endregion

        # 这个没验证过, 因为有两个输出, 可能需要改动
        # region Prediction
        if tf_data["test"] is not None:
            logger.info("Doing predictions on test dataset...")
            predictions = model.predict(tf_data["test"])["logits"]
            predicted_class = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)
            output_test_file = os.path.join(training_args.output_dir, "test_results.txt")
            with open(output_test_file, "w") as writer:
                writer.write("index\tprediction\n")
                for index, item in enumerate(predicted_class):
                    if is_regression:
                        writer.write(f"{index}\t{item:3.3f}\n")
                    else:
                        item = config.id2label[item]
                        writer.write(f"{index}\t{item}\n")
            logger.info(f"Wrote predictions to {output_test_file}!")
        # endregion

    # region Prediction losses
    # This section is outside the scope() because it's very quick to compute, but behaves badly inside it
    if "test" in datasets and "label" in datasets["test"].features:
        print("Computing prediction loss on test labels...")
        labels = datasets["test"]["label"]
        loss = float(loss_fn(labels, predictions).numpy())
        print(f"Test loss: {loss:.4f}")
    # endregion


if __name__ == "__main__":
    main()
