import os
import csv
from pathlib import Path

import pandas as pd


class EvaluatingReport:

    __report_columns = [
        "timestamp",
        "eval_loss",
        "eval_precision",
        "eval_recall",
        "eval_f1",
        "eval_accuracy",
        "eval_runtime",
        "eval_samples_per_second",
        "eval_steps_per_second",
        "epoch",
    ]

    __path: Path

    def __init__(self, path):
        self.__path = path

    def write_to_report(self, output, now):
        os.makedirs(os.path.dirname(self.__path), exist_ok=True)

        row = [now] + [output[col] for col in self.__report_columns[1:]]

        with open(self.__path, "a") as the_file:
            writer = csv.writer(the_file)
            writer.writerow(row)

    def read_report(self):
        return pd.read_csv(
            self.__path,
            names=self.__report_columns,
        )
