import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from transformers import PreTrainedTokenizerBase, BertTokenizerFast
from datasets import Dataset, load_dataset


class ManipulationDetectionDataset:

    __tokenizer: BertTokenizerFast
    __seed: int
    __label2id = {
        "O": 0,
        "MANIPULATION": 1,
    }
    __id2label = {v: k for k, v in __label2id.items()}

    def __init__(self, tokenizer: PreTrainedTokenizerBase, seed: int = 42):
        self.__tokenizer = tokenizer
        self.__seed = seed

    @property
    def label2id(self):
        return self.__label2id

    @property
    def id2label(self):
        return self.__id2label

    def read(self, ds_path: Path, train_ratio: float = 0.9):
        dataset = load_dataset("parquet", split="train", data_files=str(ds_path))
        dataset = dataset.shuffle(self.__seed)
        dataset = dataset.train_test_split(train_size=train_ratio)

        return dataset.map(self.__encode_labels, batched=True)

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
                if id is None or id == previous_word_id:
                    example_labels[j] = -100
                previous_word_id = id

            labels.append(example_labels)

        tokenized_inputs["labels"] = labels

        del tokenized_inputs["offset_mapping"]

        return tokenized_inputs
