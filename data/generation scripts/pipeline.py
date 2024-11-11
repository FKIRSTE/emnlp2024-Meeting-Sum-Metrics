#!/usr/bin/env python
# coding=utf-8
# adaptation of run_summarization.py
# https://github.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization.py

"""
Fine-tuning the library models for sequence to sequence.
"""

import logging
import os

import wandb

import sled   # *** required so that SledModels will be registered for the AutoClasses ***
from sled import SledTokenizer

import evaluate
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
import pandas as pd
from datasets import load_dataset
from filelock import FileLock

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Trainer,
    Seq2SeqTrainer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version

from helpers.args import get_args, ModelArguments, DataTrainingArguments
from helpers.fn import setup_logger, from_checkpoint

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.26.1")
#
# require_version("datasets>=1.8.0",
#                "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [
    MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]

summarization_name_mapping = {
    "ami": ("transcript", "summary"),
    "icsi": ("transcript", "summary"),
    "qmsum": ("transcript", "summary"),
}
output_decider = 0


def main(model_args, data_args, training_args):

    # Setup logging
    setup_logger(training_args=training_args, logger=logger)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = from_checkpoint(
        training_args=training_args, logger=logger)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer, model = get_model(model_args)
    model.config.num_beams = 8 # default 4, min 1, max 8
    model.config.max_length = 1024  # 512
    model.config.min_length = 100
    model.config.length_penalty = 2.0
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.gradient_checkpointing_enable()  # eneable if there is a memory issue

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.

    # resize_embeddings(model_args, data_args, tokenizer, model)
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    raw_datasets = get_datasets(model_args, data_args)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        column_names = raw_datasets["test"].column_names
    else:
        logger.info(
            "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert (
            data_args.lang is not None
        ), f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --lang argument"

        tokenizer.src_lang = data_args.lang
        tokenizer.tgt_lang = data_args.lang

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id

    # Get the column names for input/target.
    # dataset_columns = summarization_name_mapping.get(
    #    data_args.dataset_name, None)
    # if data_args.text_column is None:
    #    text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    # else:
    #    text_column = data_args.text_column
    #    if text_column not in column_names:
    #        raise ValueError(
    #            f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
    #        )
    # if data_args.summary_column is None:
    #    summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    # else:
    #    summary_column = data_args.summary_column
    #    if summary_column not in column_names:
    #        raise ValueError(
    #            f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
    #        )

    text_column = "transcript"
    summary_column = "summary"

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length"  # if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        # remove pairs where at least one record is None

        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i]:
                inputs.append(examples[text_column][i])
                targets.append(examples[summary_column][i])

        # inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(
            inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        # Tokenize targets with the `text_target` keyword argument
        # labels = tokenizer(
        #    text_target=targets, max_length=max_target_length, padding=padding, truncation=True)
        labels = tokenizer(
            targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        # if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        #    labels["input_ids"] = [
        #        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        #    ]
        #
        # model_inputs["labels"] = labels["input_ids"]
        # return model_inputs

        examples["input_ids"] = model_inputs.input_ids
        examples["attention_mask"] = model_inputs.attention_mask

        # create 0 global_attention_mask lists
        examples["global_attention_mask"] = len(examples["input_ids"]) * [
            [0 for _ in range(len(examples["input_ids"][0]))]
        ]

        # since above lists are references, the following line changes the 0 index for all samples
        examples["global_attention_mask"][0][0] = 1
        examples["labels"] = labels.input_ids

        # We have to make sure that the PAD token is ignored
        examples["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels]
            for labels in examples["labels"]
        ]

        return examples

    train_dataset = None
    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(
                len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                batch_size=training_args.per_device_train_batch_size,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
            train_dataset.set_format(type="torch", columns=[
                "input_ids", "attention_mask", "global_attention_mask", "labels"])

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(
                len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                batch_size=training_args.per_device_eval_batch_size,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
            eval_dataset.set_format(type="torch", columns=[
                "input_ids", "attention_mask", "global_attention_mask", "labels"])

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(
                range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                batch_size=training_args.per_device_eval_batch_size,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
            predict_dataset.set_format(type="torch", columns=[
                "input_ids", "attention_mask", "global_attention_mask", "labels"])

    # Metric
    metric = evaluate.load("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(
            decoded_preds, decoded_labels)

        results = {}
        results['prediction'] = []
        results['label'] = []
        for decode_pred, decode_label in zip(decoded_preds, decoded_labels):
            results['prediction'].append(decode_pred)
            results['label'].append(decode_label)
            result = metric.compute(predictions=[decode_pred],
                                    references=[decode_label], use_stemmer=True)
            result = {k: round(v * 100, 4) for k, v in result.items()}

            for key, value in result.items():
                if key not in results:
                    results[key] = []
                results[key].append(value)
        results_df = pd.DataFrame.from_dict(results)
        global output_decider
        output_suffix = ['eval', 'test']
        output_path = training_args.output_dir + data_args.proxy + "_" + data_args.tester + "_" + \
            output_suffix[output_decider] + '.csv'
        output_decider = 1
        results_df.to_csv(output_path, index=False)
        print(results_df.head(20))
        result = {}
        for key, values in results.items():
            if key in ['prediction', 'label']:
                continue
            result[key] = np.mean(values)

        print(result)

        prediction_lens = [np.count_nonzero(
            pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # Override the decoding parameters of Seq2SeqTrainer
    # training_args.generation_max_length = (
    #    training_args.generation_max_length
    #    if training_args.generation_max_length is not None
    #    else data_args.val_max_target_length
    # )
    # training_args.generation_num_beams = (
    #    data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    # )

    trainer = get_trainer(model_args, data_args, training_args, tokenizer,
                          model, train_dataset, eval_dataset, compute_metrics)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(
                train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(
            eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict")
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(
                predict_dataset)
        )
        metrics["predict_samples"] = min(
            max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(
                    training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))

    return results


def get_trainer(model_args, data_args, training_args, tokenizer, model, train_dataset, eval_dataset, compute_metrics):
    # Data collator
    label_pad_token_id = - \
        100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    # data_collator = DataCollatorForSeq2Seq(
    #    tokenizer,
    #    model=model,
    #    label_pad_token_id=label_pad_token_id,
    #    pad_to_multiple_of=8 if training_args.fp16 else None,
    # )

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        #    data_collator=data_collator,
    )

    return trainer


def resize_embeddings(model_args, data_args, tokenizer, model):
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(
                data_args.lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )


def get_model_sled(model_args):
    tokenizer = SledTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path)
    return tokenizer, model


def get_model(model_args):
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    return tokenizer, model


def get_datasets(model_args, data_args):
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    return raw_datasets


if __name__ == "__main__":
    model_args, data_args, training_args = get_args()

    print(training_args.output_dir)
    print(model_args.model_name_or_path)

    wandb.login(key="a467a3f1c3b5d7d30010d3bde74e5b0df701a213")
    wandb.init(project="ProxyTasks4MS", entity="nlt",
               # settings=wandb.Settings(start_method="fork"), reinit=True,
               config={
                   "architecture": model_args.model_name_or_path,
                   "train dataset": data_args.train_file,
                   "test dataset": data_args.test_file,
                   "epochs": 3,
                   "input-output": "16k - 1k"
               })

    wandb.config.dataset_name = data_args.train_file
    wandb.config.stage = "training"
    wandb.run.name = f"training_{model_args.model_name_or_path}_{data_args.train_file}-{wandb.util.generate_id()}"
    wandb.run.save()
    wandb.init(mode="disabled")
    main(model_args=model_args, data_args=data_args, training_args=training_args)
