from pathlib import Path
from bs4 import BeautifulSoup
from datasets import load_dataset, DatasetDict
import multiprocessing

import torch


def load_ukrainian_news_dataset(cache_folder, original_cache_folder, tokenizer):
    def remove_html_tags(example):
        soup = BeautifulSoup(example["text"], "html.parser")
        return {"text": soup.get_text()}

    cache_path = cache_folder / "ukrainian-news"

    if cache_path.exists():
        return DatasetDict.load_from_disk(cache_path)

    core_count = multiprocessing.cpu_count()

    cache_folder.mkdir(parents=True, exist_ok=True)
    original_cache_folder.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "zeusfsx/ukrainian-news",
        split="train[:10000]",
        cache_dir=original_cache_folder,
        num_proc=core_count,
    )
    dataset = dataset.select_columns(["text"])
    dataset = dataset.map(remove_html_tags, num_proc=core_count)

    dataset.save_to_disk(cache_path)

    return dataset
