from enum import Enum
import os


ERROR_TYPE_TO_COLOR = {1: "green", 2: "yellow", 3: "red"}


class VisualizationMode(Enum):
    BERT = 1
    ROBERTA = 2


class MarkdownVisualizer:

    def __init__(self, tokenizer, path, visualization_mode):
        self.tokenizer = tokenizer
        self.path = path
        self.visualization_mode = visualization_mode

    def visualize_as_markdown_and_save(
        self,
        dataset,
        predictions,
    ):
        entries = self.visualize_as_markdown(dataset, predictions)

        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        with open(self.path, "a") as the_file:
            for it in entries:
                the_file.write(it)
                the_file.write("\n\n")

    def visualize_as_markdown(self, dataset, predictions):
        return [
            self.__visualize_single_datapoint(id, labels, input_ids, prediction)
            for id, labels, input_ids, prediction in zip(
                dataset["id"], dataset["labels"], dataset["input_ids"], predictions
            )
        ]

    def __visualize_single_datapoint(self, id, labels, input_ids, prediction):
        content = self.tokenizer.convert_ids_to_tokens(input_ids)
        prediction_arr = self.__predictions_as_array(prediction, len(content))
        confusion_arr = self.__extract_confusion(content, labels, prediction_arr)

        def to_md(i):
            token = content[i]

            if self.visualization_mode == VisualizationMode.ROBERTA:
                if token == "<s>" or token == "</s>":
                    return ""

            confusion = confusion_arr[i]

            if self.visualization_mode == VisualizationMode.BERT:
                result = token[2:] if token.startswith("##") else " " + token
            else:
                result = " " + token[1:] if token.startswith("▁") else token

            if confusion > 0:
                return f"<mark style='background-color:{ERROR_TYPE_TO_COLOR[confusion]}'>{result}</mark>"

            return result

        return id + "<br>" + "".join([to_md(i) for i in range(len(content))])

    def __extract_confusion(self, content, labels, predictions):
        def extract_label(source, index, prev):
            if self.visualization_mode == VisualizationMode.BERT:
                if content[index].startswith("##"):
                    return prev
                else:
                    return source[index]
            else:
                if not content[index].startswith("▁"):
                    return prev
                else:
                    return source[index]

        confusion = [0] * len(labels)

        prev_y_true = 0
        prev_y_pred = 0

        for i in range(len(confusion)):
            y_true = extract_label(labels, i, prev_y_true)
            y_pred = extract_label(predictions, i, prev_y_pred)

            if y_true == 1 and y_pred == 1:
                confusion[i] = 1  ## TRUE POSITIVE
            elif y_true == 1 and y_pred == 0:
                confusion[i] = 2  ## FALSE NEGATIVE
            elif y_true == 0 and y_pred == 1:
                confusion[i] = 3  ## FALSE POSITIVE

            prev_y_true = y_true
            prev_y_pred = y_pred

        return confusion

    def __predictions_as_array(self, prediction, tokens_count):
        pred_arr = [0] * tokens_count

        for pred_entry in prediction:
            pred_arr[pred_entry["index"]] = 1

        return pred_arr
