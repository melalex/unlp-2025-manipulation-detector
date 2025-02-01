import os


ERROR_TYPE_TO_COLOR = {1: "green", 2: "yellow", 3: "red"}


def visualize_as_markdown_and_save(dataset, predictions, tokenizer, path):
    entries = visualize_as_markdown(dataset, predictions, tokenizer)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "a") as the_file:
        for it in entries:
            the_file.write(it)
            the_file.write("\n\n")


def visualize_as_markdown(dataset, predictions, tokenizer):
    print(dataset)
    return [
        visualize_single_datapoint(id, labels, input_ids, prediction, tokenizer)
        for id, labels, input_ids, prediction in zip(
            dataset["id"], dataset["labels"], dataset["input_ids"], predictions
        )
    ]


def visualize_single_datapoint(id, labels, input_ids, prediction, tokenizer):
    content = tokenizer.convert_ids_to_tokens(input_ids)
    prediction_arr = predictions_as_array(prediction, len(content))
    confusion_arr = extract_confusion(content, labels, prediction_arr)

    def to_md(i):
        token = content[i]
        confusion = confusion_arr[i]
        result = token[2:] if token.startswith("##") else " " + token

        if confusion > 0:
            return f"<mark style='background-color:{ERROR_TYPE_TO_COLOR[confusion]}'>{result}</mark>"

        return result

    return id + "<br>" + "".join([to_md(i) for i in range(len(content))])


def extract_confusion(content, labels, predictions):
    def extract_label(source, index, prev):
        if content[index].startswith("##"):
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


def predictions_as_array(prediction, tokens_count):
    pred_arr = [0] * tokens_count

    for pred_entry in prediction:
        pred_arr[pred_entry["index"]] = 1

    return pred_arr
