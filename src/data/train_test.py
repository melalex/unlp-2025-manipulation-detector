import multiprocessing
import pandas as pd

from datasets import Dataset


def load_train_test_dataset(data_folder, tokenizer):

    def preprocess_function(examples):
        return tokenizer(examples["content"], truncation=True)

    core_count = multiprocessing.cpu_count()

    train = pd.read_parquet(data_folder / "span-detection.parquet")
    test = pd.read_csv(data_folder / "test.csv")
    train = train["content"]
    test = test["content"]

    df = pd.concat([train, test], ignore_index=True)

    ds = Dataset.from_pandas(df.to_frame())
    ds = ds.map(
        preprocess_function,
        remove_columns=["content"],
        num_proc=core_count,
        batched=True,
    )

    return ds
