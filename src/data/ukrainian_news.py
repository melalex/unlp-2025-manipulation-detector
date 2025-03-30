from bs4 import BeautifulSoup
from datasets import load_dataset, Dataset
import multiprocessing


def load_ukrainian_news_dataset(
    tokenizer,
    rows_count=None,
    max_tokens=512,
):
    def remove_html_tags(example):
        soup = BeautifulSoup(example["text"], "html.parser")
        return {"text": soup.get_text()}

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=False)

    def filter_long_examples(example):
        return len(example["input_ids"]) <= max_tokens

    def filter_out_telegram_posts(example):
        return not example["url"].startswith("https://t.me")

    core_count = multiprocessing.cpu_count()

    split = f"train[:{rows_count}]" if rows_count is not None else "train"

    ds = load_dataset(
        "zeusfsx/ukrainian-news",
        split=split,
        num_proc=core_count,
    )
    ds = ds.filter(filter_out_telegram_posts)
    ds = ds.select_columns(["text"])
    ds = ds.map(remove_html_tags, num_proc=core_count)
    ds = ds.map(
        preprocess_function, remove_columns=["text"], num_proc=core_count, batched=True
    )
    ds = ds.filter(filter_long_examples)

    return ds
