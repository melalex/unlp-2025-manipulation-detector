from datasets import load_dataset
import multiprocessing


def load_rus_news_classifier_dataset(
    tokenizer,
    rows_count=None,
    max_tokens=512,
):
    def preprocess_function(examples):
        return tokenizer(examples["news"], truncation=False)

    def filter_long_examples(example):
        return len(example["input_ids"]) <= max_tokens

    core_count = multiprocessing.cpu_count()

    split = f"train[:{rows_count}]" if rows_count is not None else "train"

    ds = load_dataset(
        "data-silence/rus_news_classifier",
        split=split,
        num_proc=core_count,
    )
    ds = ds.select_columns(["news"])
    ds = ds.map(
        preprocess_function, remove_columns=["news"], num_proc=core_count, batched=True
    )
    ds = ds.filter(filter_long_examples)

    return ds
