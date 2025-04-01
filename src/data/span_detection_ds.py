import ast
from pathlib import Path
import re
import numpy as np
from transformers import PreTrainedTokenizerBase, BertTokenizerFast
from datasets import load_dataset, DatasetDict


class ManipulationDetectionDataset:

    __tokenizer: BertTokenizerFast
    __raw_path: Path
    __processed_path: Path
    __train_ratio: float = 0.9
    __seed: int
    __label2id = {
        "O": 0,
        "I-MANIPULATION": 1,
    }
    __id2label = {v: k for k, v in __label2id.items()}
    __exclude_tail: bool = True
    __load_existing: bool = False
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        raw_path: Path,
        processed_path: Path,
        exclude_tail: bool = True,
        train_ratio: float = 0.9,
        seed: int = 42,
        load_existing: bool = False,
        do_split: bool = True,
        lang: str = None
    ):
        self.__tokenizer = tokenizer
        self.__raw_path = raw_path
        self.__processed_path = processed_path
        self.__exclude_tail = exclude_tail
        self.__train_ratio = train_ratio
        self.__seed = seed
        self.__load_existing = load_existing
        self.__do_split = do_split
        self.__lang = lang

    @property
    def label2id(self):
        return self.__label2id

    @property
    def id2label(self):
        return self.__id2label

    def read(self):
        if self.__processed_path.exists() and self.__load_existing:
            return DatasetDict.load_from_disk(str(self.__processed_path))

        ds = self.__load_ds()
        ds.save_to_disk(str(self.__processed_path))

        return ds

    def __load_ds(self):
        dataset = load_dataset(
            "parquet", split="train", data_files=str(self.__raw_path)
        )
        dataset = dataset.shuffle(self.__seed)
        if self.__do_split:
            dataset = dataset.train_test_split(train_size=self.__train_ratio, seed=self.__seed)

        if self.__lang:
            dataset = dataset.filter(lambda x: x["lang"] == self.__lang)

        dataset = dataset.map(self.__encode_labels, batched=True, remove_columns=["lang", "manipulative", "techniques", "trigger_words"])

        return dataset

    def __encode_labels(self, data):
        tokenized_inputs = self.__tokenizer(
            data["content"],
            truncation=True,
            return_offsets_mapping=True,
        )

        labels = []

        for i, offsets in enumerate(tokenized_inputs["offset_mapping"]):
            example_labels = [0] * len(offsets)
            trigger_words = data["trigger_words"][i]
            trigger_words = trigger_words if trigger_words is not None else []
            for start, end in trigger_words:
                for idx, (offset_start, offset_end) in enumerate(offsets):
                    if offset_start >= start and offset_end <= end:
                        example_labels[idx] = 1

            word_ids = tokenized_inputs.word_ids(i)

            previous_word_id = None

            for j, id in enumerate(word_ids):
                if self.__exclude_tail:
                    if id is None or id == previous_word_id:
                        example_labels[j] = -100
                else:
                    if id is None:
                        example_labels[j] = -100
                previous_word_id = id

            labels.append(example_labels)

        tokenized_inputs["labels"] = labels

        del tokenized_inputs["offset_mapping"]

        return tokenized_inputs
