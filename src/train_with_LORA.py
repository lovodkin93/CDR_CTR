import gc
import os
import sys
import threading
from dataclasses import dataclass, field
from typing import Optional
import logging

import numpy as np
import psutil
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import datasets
import evaluate

import transformers
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup, set_seed, HfArgumentParser, Seq2SeqTrainingArguments

from peft import LoraConfig, TaskType, get_peft_model
from src.preprocessor import Preprocessor, get_special_tokens_constants

logger = logging.getLogger(__name__)

def levenshtein_distance(str1, str2):
    # TC: O(N^2)
    # SC: O(N^2)
    if str1 == str2:
        return 0
    num_rows = len(str1) + 1
    num_cols = len(str2) + 1
    dp_matrix = np.empty((num_rows, num_cols))
    dp_matrix[0, :] = range(num_cols)
    dp_matrix[:, 0] = range(num_rows)

    for i in range(1, num_rows):
        for j in range(1, num_cols):
            if str1[i - 1] == str2[j - 1]:
                dp_matrix[i, j] = dp_matrix[i - 1, j - 1]
            else:
                dp_matrix[i, j] = min(dp_matrix[i - 1, j - 1], dp_matrix[i - 1, j], dp_matrix[i, j - 1]) + 1

    return dp_matrix[num_rows - 1, num_cols - 1]


def get_closest_label(eval_pred, classes):
    min_id = sys.maxsize
    min_edit_distance = sys.maxsize
    for i, class_label in enumerate(classes):
        edit_distance = levenshtein_distance(eval_pred.strip(), class_label)
        if edit_distance < min_edit_distance:
            min_id = i
            min_edit_distance = edit_distance
    return classes[min_id]




@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    load_in_8bit: bool = field(
        default=False,
        metadata={
            "help": "whether to load model in 8bit precision."
        },
    )


    device_map: str = field(
        default=None,
        metadata={
            "help": "When loading in 8bit precision, maps which layers are saved on which GPUs/CPU. If not passed and load_in_8bit is True, will be set to \"auto\"."
        },
    )

    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
            "the model's position embeddings."
        },
    )
    # NEW from original script
    freeze_embeds: bool = field(
        default=False
    )
    # NEW from original script
    min_length: int = field(
        default=None
    )
    # NEW from original script
    length_penalty: float = field(
        default=None
    )
    # NEW from original script
    early_stopping: bool = field(
        default=False
    )
    # NEW from original script
    no_repeat_ngram_size: int = field(
        default=None
    )

    # NEW from original script (for LongT5)
    local_radius: int = field(
        default=None
    )
    # NEW from original script (for LongT5)
    global_block_size: int = field(
        default=None
    )
    # NEW from original script (for LongT5)
    encoder_attention_type: int = field(
        default=None
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    experiment_type: str = field(default=None)
    lang: str = field(default=None, metadata={
                      "help": "Language id for summarization."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the decoder_start_token_id."
            "Useful for multilingual models like mBART where the first generated token"
            "needs to be the target language token (Usually it is the target language token)"
        },
    )
    # NEW from original script
    add_global_attention: bool = field(
        default=False
    )
    # NEW from original script
    add_global_attention_on_highlights: bool = field(
        default=False
    )
    # NEW from original script
    add_global_attention_on_highlighted_words: bool = field(
        default=False,
        metadata={
            "help": "Decides whether to add global attention not only on the highlight_start and highlight_end tokens, but also on the highlighted tokens themselves"
        }
    )
    # NEW from original script
    should_preprocess_add_highlights: bool = field(
        default=True,
        metadata={
            "help": "Decides whether to add highlight tokens or not"
        }
    )
    # NEW from original script
    should_preprocess_only_sents_with_highlights: bool = field(
        default=False,
        metadata={
            "help": "Decides whether to keep only sentences with highlights"
        }
    )
    # NEW from original script
    should_preprocess_keep_only_highlights: bool = field(
        default=False,
        metadata={
            "help": "Decides whether to keep only highlights"
        }
    )
    # NEW from original script
    eval_with_summac: bool = field(
        default=True
    )

    # NEW from original script
    add_planning_on_concatenation: bool = field(
        default=False
    )
    # NEW from original script
    add_highlight_delim_planning: bool = field(
        default=True
    )
        # NEW from original script
    add_highlight_labels_to_planning: bool = field(
        default=False
    )

        # NEW from original script
    add_CoT_to_output: bool = field(
        default=False
    )

        # NEW from original script
    add_icl_to_input: bool = field(
        default=False
    )


    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError(
                "Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length






# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)


# This context manager is used to track the peak memory usage of the process
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")


def main():

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if (len(sys.argv) == 3 or len(sys.argv) == 2) and sys.argv[-1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file, (when passing 2 and the second is the json file - this is distributed training)
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[-1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    )
    # text_column = "Tweet text"
    label_column = "text_label"
    lr = 3e-3
    num_epochs = 10
    batch_size = 8
    seed = training_args.seed
    # do_test = False
    set_seed(seed)

    accelerator = Accelerator()
    model_name_or_path = model_args.model_name_or_path
    # model_name_or_path = "facebook/bart-large"

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

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info(
            "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return


    model_args_dict = {}
    model_args_dict.update(model_args.__dict__)
    model_args_dict['max_length'] = data_args.max_target_length  # We must add max_length when setting min_length
    model_args_dict = { k: v for k,v in model_args_dict.items() if v is not None} 

    config = AutoConfig.from_pretrained(model_args.config_name if model_args.config_name else model_args.model_name_or_path, **model_args_dict)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None
    )    

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        # device_map=model_args.device_map,
        # load_in_8bit=model_args.load_in_8bit
    )


    
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    summarization_name_mapping = {
        "amazon_reviews_multi": ("review_body", "review_title"),
        "big_patent": ("description", "abstract"),
        "cnn_dailymail": ("article", "highlights"),
        "orange_sum": ("text", "summary"),
        "pn_summary": ("article", "summary"),
        "psc": ("extract_text", "summary_text"),
        "samsum": ("dialogue", "summary"),
        "thaisum": ("body", "summary"),
        "xglue": ("news_body", "news_title"),
        "xsum": ("document", "summary"),
        "wiki_summary": ("article", "highlights"),
    }

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(
        data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )

    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )



    # NEW from original script
    is_t5_model = model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ] or model.config.model_type == 't5'  # Necessary when loading model from directory
    if data_args.source_prefix is None and is_t5_model:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # NEW from original script
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # NEW from original script
    special_tokens_constants = get_special_tokens_constants(is_t5_model)
    preprocessor = Preprocessor(prefix, special_tokens_constants, data_args.should_preprocess_add_highlights, data_args.should_preprocess_only_sents_with_highlights, data_args.should_preprocess_keep_only_highlights, data_args.add_planning_on_concatenation, data_args.add_highlight_delim_planning, data_args.add_highlight_labels_to_planning)



    # NEW from original script
    def preprocess_function(examples):
        # Orig summarization dataset
        if data_args.dataset_name is not None:
            inputs, targets = [], []
            
            for i in range(len(examples[text_column])):
                if examples[text_column][i] is not None and examples[summary_column][i] is not None:
                    inputs.append(examples[text_column][i])
                    targets.append(examples[summary_column][i])

            inputs = [prefix + inp for inp in inputs]            
        else:
            inputs, targets = [], []
            for i in range(len(examples[text_column])):
                # NEW from original script
                curr_input = preprocessor.preprocess_input(examples['doc_text'][i], examples['highlight_spans'][i])
                inputs.append(curr_input)
                curr_output = preprocessor.preprocess_output(examples['summary_text'][i], curr_input)
                targets.append(curr_output)

        model_inputs = tokenizer(
            inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, max_length=max_target_length, padding="max_length", truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]

        # NEW from original script
        global_attention_mask = []
        if data_args.add_global_attention:
            for input_ids in tqdm(model_inputs['input_ids']):
                curr_global_attention_mask = [0 for _ in range(len(input_ids))]
                curr_global_attention_mask[0] = 1

                tkns_with_global_attention = [preprocessor.special_tokens_constants[tkn_key] for tkn_key in ['highlight_start', 'highlight_end']]
                ids_with_global_attention = [special_id for special_id in tokenizer.additional_special_tokens_ids if tokenizer.convert_ids_to_tokens(special_id) in tkns_with_global_attention]
                # ids_with_global_attention = tokenizer.additional_special_tokens_ids

                highlight_start_tkn_id = tokenizer.convert_tokens_to_ids(preprocessor.special_tokens_constants['highlight_start'])
                highlight_end_tkn_id = tokenizer.convert_tokens_to_ids(preprocessor.special_tokens_constants['highlight_end'])


                if data_args.add_global_attention_on_highlights:
                    highlight_began_flag = False
                    for input_id_idx, input_id in enumerate(input_ids):
                        # Put attention on highlight tokens
                        if input_id in ids_with_global_attention: #AVIVSL: play with this (regarding the other special tokens)
                            curr_global_attention_mask[input_id_idx] = 1
                        if data_args.add_global_attention_on_highlighted_words:
                            if input_id == highlight_start_tkn_id:
                                highlight_began_flag = True
                            elif input_id == highlight_end_tkn_id:
                                highlight_began_flag = False
                            elif highlight_began_flag:
                                curr_global_attention_mask[input_id_idx] = 1

                global_attention_mask.append(curr_global_attention_mask)
            model_inputs['global_attention_mask'] = global_attention_mask

        return model_inputs

    ####################################### AVIVSL - return #########################################
    with accelerator.main_process_first():
        if training_args.do_train:
            if "train" not in raw_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = raw_datasets["train"]
            if data_args.max_train_samples is not None:
                max_train_samples = min(
                    len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))
            with training_args.main_process_first(desc="train dataset map pre-processing"):
                train_dataset = train_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on train dataset",
                )

        if training_args.do_eval:
            max_target_length = data_args.val_max_target_length
            if "validation" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets["validation"]
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(
                    len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
            with training_args.main_process_first(desc="validation dataset map pre-processing"):
                eval_dataset = eval_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on validation dataset",
                )

        if training_args.do_predict:
            max_target_length = data_args.val_max_target_length
            if "test" not in raw_datasets:
                raise ValueError("--do_predict requires a test dataset")
            predict_dataset = raw_datasets["test"]
            if data_args.max_predict_samples is not None:
                max_predict_samples = min(
                    len(predict_dataset), data_args.max_predict_samples)
                predict_dataset = predict_dataset.select(
                    range(max_predict_samples))
            with training_args.main_process_first(desc="prediction dataset map pre-processing"):
                # NEW from original_script
                prep_predict_dataset = predict_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on prediction dataset",
                )

    accelerator.wait_for_everyone()
    
    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    if training_args.do_train:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True
        )
    
    if training_args.do_eval:
        eval_dataloader = DataLoader(eval_dataset, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True)
    
    if training_args.do_predict:
        test_dataloader = DataLoader(prep_predict_dataset, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True)
    ################################################################################################

    # creating model
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # lr scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    if training_args.do_train:
        train_dataloader = accelerator.prepare(train_dataloader)
    if training_args.do_eval:
        eval_dataloader = accelerator.prepare(eval_dataloader)
    if training_args.do_predict:
        test_dataloader = accelerator.prepare(test_dataloader)   

    accelerator.print(model)

    is_ds_zero_3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3

    metric = evaluate.load("rouge")

    for epoch in range(num_epochs):
        with TorchTracemalloc() as tracemalloc:
            model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print("GPU Memory before entering the train : {}".format(b2mb(tracemalloc.begin)))
        accelerator.print("GPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.used))
        accelerator.print("GPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.peaked))
        accelerator.print(
            "GPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.peaked + b2mb(tracemalloc.begin)
            )
        )

        accelerator.print("CPU Memory before entering the train : {}".format(b2mb(tracemalloc.cpu_begin)))
        accelerator.print("CPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.cpu_used))
        accelerator.print("CPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.cpu_peaked))
        accelerator.print(
            "CPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
            )
        )
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        accelerator.print(f"{epoch}: {train_ppl} {train_epoch_loss}")

        model.eval()
        eval_preds = []
        with TorchTracemalloc() as tracemalloc:
            for _, batch in enumerate(tqdm(eval_dataloader)):
                batch = {k: v for k, v in batch.items() if k != "labels"}
                with torch.no_grad():
                    outputs = accelerator.unwrap_model(model).generate(
                        **batch, synced_gpus=is_ds_zero_3
                    )  # synced_gpus=True for DS-stage 3
                outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id)
                preds = accelerator.gather_for_metrics(outputs).detach().cpu().numpy()
                eval_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print("GPU Memory before entering the eval : {}".format(b2mb(tracemalloc.begin)))
        accelerator.print("GPU Memory consumed at the end of the eval (end-begin): {}".format(tracemalloc.used))
        accelerator.print("GPU Peak Memory consumed during the eval (max-begin): {}".format(tracemalloc.peaked))
        accelerator.print(
            "GPU Total Peak Memory consumed during the eval (max): {}".format(
                tracemalloc.peaked + b2mb(tracemalloc.begin)
            )
        )

        accelerator.print("CPU Memory before entering the eval : {}".format(b2mb(tracemalloc.cpu_begin)))
        accelerator.print("CPU Memory consumed at the end of the eval (end-begin): {}".format(tracemalloc.cpu_used))
        accelerator.print("CPU Peak Memory consumed during the eval (max-begin): {}".format(tracemalloc.cpu_peaked))
        accelerator.print(
            "CPU Total Peak Memory consumed during the eval (max): {}".format(
                tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
            )
        )

        correct = 0
        total = 0
        assert len(eval_preds) == len(
            raw_datasets["train"][label_column]
        ), f"{len(eval_preds)} != {len(raw_datasets['train'][label_column])}"
        for pred, true in zip(eval_preds, raw_datasets["train"][label_column]):
            if pred.strip() == true.strip():
                correct += 1
            total += 1
        accuracy = correct / total * 100
        accelerator.print(f"{accuracy}")
        accelerator.print(f"{eval_preds[:10]}")
        accelerator.print(f"{raw_datasets['train'][label_column][:10]}")

    if training_args.do_predict:
        model.eval()
        test_preds = []
        for _, batch in enumerate(tqdm(test_dataloader)):
            batch = {k: v for k, v in batch.items() if k != "labels"}
            with torch.no_grad():
                outputs = accelerator.unwrap_model(model).generate(
                    **batch, synced_gpus=is_ds_zero_3
                )  # synced_gpus=True for DS-stage 3
            outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id)
            preds = accelerator.gather(outputs).detach().cpu().numpy()
            test_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

        test_preds_cleaned = []
        for _, pred in enumerate(test_preds):
            test_preds_cleaned.append(get_closest_label(pred, classes))

        test_df = dataset["test"].to_pandas()
        assert len(test_preds_cleaned) == len(test_df), f"{len(test_preds_cleaned)} != {len(test_df)}"
        test_df[label_column] = test_preds_cleaned
        test_df["text_labels_orig"] = test_preds
        accelerator.print(test_df[[text_column, label_column]].sample(20))

        pred_df = test_df[["ID", label_column]]
        pred_df.columns = ["ID", "Label"]

        os.makedirs(f"data/{dataset_name}", exist_ok=True)
        pred_df.to_csv(f"data/{dataset_name}/predictions.csv", index=False)

    accelerator.wait_for_everyone()
    model.push_to_hub(
        "smangrul/"
        + f"{dataset_name}_{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}".replace("/", "_"),
        state_dict=accelerator.get_state_dict(model),
        use_auth_token=True,
    )
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()