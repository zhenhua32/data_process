import logging

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    DefaultDataCollator,
    PretrainedConfig,
    TFTrainingArguments,
)

from param_helper import DataTrainingArguments, ModelArguments

logger = logging.getLogger(__name__)


"""
加载数据集的辅助函数
"""


def load_data(
    model_args: ModelArguments, data_args: DataTrainingArguments, training_args: TFTrainingArguments, checkpoint=None
):
    # region Loading data
    # For CSV/JSON files, this script will use the 'label' field as the label and the 'sentence1' and optionally
    # 'sentence2' fields as inputs if they exist. If not, the first two fields not named label are used if at least two
    # columns are provided. Note that the term 'sentence' can be slightly misleading, as they often contain more than
    # a single grammatical sentence, when the task requires it.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {"train": data_args.train_file, "validation": data_args.validation_file, "test": data_args.test_file}
    data_files = {key: file for key, file in data_files.items() if file is not None}

    for key in data_files.keys():
        logger.info(f"Loading a local file for {key}: {data_files[key]}")

    if data_args.input_file_extension == "csv":
        # Loading a dataset from local csv files
        # 实际上就是用 pandas 读取 csv 的, 见
        # https://github.com/huggingface/datasets/blob/369dadbd775a9db07a2db9a571d1a69cdf20b13c/src/datasets/packaged_modules/csv/csv.py#L124
        datasets = load_dataset(
            "csv",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            header=None,
            sep="\t",
            column_names=["label", "sentence1"],
        )
    else:
        # Loading a dataset from local json files
        datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    # endregion

    # region Label preprocessing
    # If you've passed us a training set, we try to infer your labels from it
    if "train" in datasets:
        # By default we assume that if your label column looks like a float then you're doing regression,
        # and if not then you're doing classification. This is something you may want to change!
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    # If you haven't passed a training set, we read label info from the saved model (this happens later)
    else:
        num_labels = None
        label_list = None
        is_regression = None
    # endregion

    # region Load model config and tokenizer
    if checkpoint is not None:
        config_path = training_args.output_dir
    elif model_args.config_name:
        config_path = model_args.config_name
    else:
        config_path = model_args.model_name_or_path
    if num_labels is not None:
        config = AutoConfig.from_pretrained(
            config_path,
            num_labels=num_labels,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        config = AutoConfig.from_pretrained(
            config_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # 保存一下 tokenizer
    tokenizer.save_pretrained(training_args.output_dir)
    # endregion

    # region Dataset preprocessing
    # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
    column_names = {col for cols in datasets.column_names.values() for col in cols}
    non_label_column_names = [name for name in column_names if name != "label"]
    if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    elif "sentence1" in non_label_column_names:
        sentence1_key, sentence2_key = "sentence1", None
    else:
        if len(non_label_column_names) >= 2:
            sentence1_key, sentence2_key = non_label_column_names[:2]
        else:
            sentence1_key, sentence2_key = non_label_column_names[0], None

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Ensure that our labels match the model's, if it has some pre-specified
    if "train" in datasets:
        if not is_regression and config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
            label_name_to_id = config.label2id
            if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
                label_to_id = label_name_to_id  # Use the model's labels
            else:
                logger.warning(
                    "Your model seems to have been trained with labels, but they don't match the dataset: ",
                    f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                    "\nIgnoring the model labels as a result.",
                )
                label_to_id = {v: i for i, v in enumerate(label_list)}
        elif not is_regression:
            label_to_id = {v: i for i, v in enumerate(label_list)}
        else:
            label_to_id = None
        # Now we've established our label2id, let's overwrite the model config with it.
        config.label2id = label_to_id
        if config.label2id is not None:
            config.id2label = {id: label for label, id in label_to_id.items()}
        else:
            config.id2label = None
    else:
        label_to_id = config.label2id  # Just load the data from the model

    if "validation" in datasets and config.label2id is not None:
        validation_label_list = datasets["validation"].unique("label")
        for val_label in validation_label_list:
            assert val_label in label_to_id, f"Label {val_label} is in the validation set but not the training set!"

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, max_length=max_seq_length, truncation=True)

        # Map labels to IDs
        if config.label2id is not None and "label" in examples:
            result["label"] = [(config.label2id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

    if data_args.pad_to_max_length:
        data_collator = DefaultDataCollator(return_tensors="tf")
    else:
        data_collator = DataCollatorWithPadding(tokenizer, return_tensors="tf")
    # endregion

    return datasets, config, is_regression, data_collator, non_label_column_names


def covert_to_tf_dataset(
    datasets,
    data_args: DataTrainingArguments,
    training_args: TFTrainingArguments,
    data_collator,
    non_label_column_names,
    is_layer_label=False,
):
    """
    转换成 tf 的 dataset 格式
    """
    # region Convert data to a tf.data.Dataset
    tf_data = dict()
    max_samples = {
        "train": data_args.max_train_samples,
        "validation": data_args.max_val_samples,
        "test": data_args.max_test_samples,
    }
    for key in ("train", "validation", "test"):
        if key not in datasets:
            tf_data[key] = None
            continue
        # if key in ("train", "validation"):
        #     assert "label" in datasets[key].features, f"Missing labels from {key} data!"
        if key == "train":
            shuffle = True
            batch_size = training_args.per_device_train_batch_size
            drop_remainder = True  # Saves us worrying about scaling gradients for the last batch
        else:
            shuffle = False
            batch_size = training_args.per_device_eval_batch_size
            drop_remainder = False
        samples_limit = max_samples[key]
        dataset = datasets[key]
        if samples_limit is not None:
            dataset = dataset.select(range(samples_limit))

        if is_layer_label:
            label_cols = [x for x in dataset.column_names if x.startswith("output_")]
            label_cols = label_cols if label_cols else None
        else:
            label_cols = "label" if "label" in dataset.column_names else None

        data = dataset.to_tf_dataset(
            columns=[col for col in dataset.column_names if col not in set(non_label_column_names + ["label"])],
            shuffle=shuffle,
            batch_size=batch_size,
            collate_fn=data_collator,
            drop_remainder=drop_remainder,
            # `label_cols` is needed for user-defined losses, such as in this example
            label_cols=label_cols,
        )
        tf_data[key] = data
    # endregion

    return tf_data


"""
来一个层次分类的版本
"""


def load_data_layer(
    model_args: ModelArguments, data_args: DataTrainingArguments, training_args: TFTrainingArguments, checkpoint=None
):
    """
    用于层次分类, 删除对回归的兼容性
    """
    data_files = {"train": data_args.train_file, "validation": data_args.validation_file, "test": data_args.test_file}
    data_files = {key: file for key, file in data_files.items() if file is not None}

    for key in data_files.keys():
        logger.info(f"Loading a local file for {key}: {data_files[key]}")

    if data_args.input_file_extension == "csv":
        # Loading a dataset from local csv files
        # 实际上就是用 pandas 读取 csv 的, 见
        # https://github.com/huggingface/datasets/blob/369dadbd775a9db07a2db9a571d1a69cdf20b13c/src/datasets/packaged_modules/csv/csv.py#L124
        datasets = load_dataset(
            "csv",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            header=None,
            sep="\t",
            column_names=["label", "sentence1"],
        )
    else:
        # Loading a dataset from local json files
        datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)

    # A useful fast method:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
    label_list = datasets["train"].unique("label")
    label_list.sort()  # Let's sort it for determinism

    # 从原始的分类中, 划分出一级类目和二级类目
    labels1 = sorted(set([x.split("_")[0] for x in label_list]))
    labels2 = sorted(set([x.split("_")[1] for x in label_list]))
    num_labels1 = len(labels1)
    num_labels2 = len(labels2)
    print(labels1, num_labels1)
    print(labels2, num_labels2)

    # region Load model config and tokenizer
    if checkpoint is not None:
        config_path = training_args.output_dir
    elif model_args.config_name:
        config_path = model_args.config_name
    else:
        config_path = model_args.model_name_or_path

    config = AutoConfig.from_pretrained(
        config_path,
        num_labels1=num_labels1,
        num_labels2=num_labels2,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # 保存一下 tokenizer
    tokenizer.save_pretrained(training_args.output_dir)

    column_names = {col for cols in datasets.column_names.values() for col in cols}
    non_label_column_names = [name for name in column_names if name != "label"]
    if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    elif "sentence1" in non_label_column_names:
        sentence1_key, sentence2_key = "sentence1", None
    else:
        if len(non_label_column_names) >= 2:
            sentence1_key, sentence2_key = non_label_column_names[:2]
        else:
            sentence1_key, sentence2_key = non_label_column_names[0], None

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Ensure that our labels match the model's, if it has some pre-specified
    label1_to_id = {v: i for i, v in enumerate(labels1)}
    label2_to_id = {v: i for i, v in enumerate(labels2)}
    id_to_label1 = {id: label for label, id in label1_to_id.items()}
    id_to_label2 = {id: label for label, id in label2_to_id.items()}
    config.label1_to_id = label1_to_id
    config.label2_to_id = label2_to_id
    config.id_to_label1 = id_to_label1
    config.id_to_label2 = id_to_label2

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, max_length=max_seq_length, truncation=True)

        # Map labels to IDs
        if "label" in examples:
            result["output_1"] = [(config.label1_to_id[x.split("_")[0]] if x != -1 else -1) for x in examples["label"]]
            result["output_2"] = [(config.label2_to_id[x.split("_")[1]] if x != -1 else -1) for x in examples["label"]]
            del examples["label"]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

    if data_args.pad_to_max_length:
        data_collator = DefaultDataCollator(return_tensors="tf")
    else:
        data_collator = DataCollatorWithPadding(tokenizer, return_tensors="tf")
    # endregion

    return datasets, config, False, data_collator, non_label_column_names
