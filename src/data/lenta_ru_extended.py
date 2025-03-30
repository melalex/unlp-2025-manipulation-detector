from datetime import datetime
from datasets import load_dataset, Dataset
import multiprocessing

import pandas as pd

REF_DATE = datetime(2022, 2, 24)


def load_lenta_ru_extended_dataset(
    tokenizer,
    rows_count=None,
    max_tokens=512,
):
    def preprocess_function(examples):
        return tokenizer(examples["news"], truncation=False)

    def filter_long_examples(example):
        return len(example["input_ids"]) <= max_tokens

    def filter_invalid_examples(example):
        if example["date"] is None or example["news"] is None:
            return False
        return datetime.strptime(example["date"], "%Y-%m-%d %H:%M:%S") > REF_DATE

    core_count = multiprocessing.cpu_count()

    split = f"train[:{rows_count}]" if rows_count is not None else "train"

    ds = load_dataset(
        "data-silence/lenta.ru_2-extended",
        split=split,
        num_proc=core_count,
    )

    ds = ds.filter(filter_invalid_examples)
    ds = ds.select_columns(["news"])
    ds = ds.map(
        preprocess_function, remove_columns=["news"], num_proc=core_count, batched=True
    )
    ds = ds.filter(filter_long_examples)

    return ds
