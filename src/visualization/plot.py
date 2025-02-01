from matplotlib import pyplot as plt
import pandas as pd


def plot_loss(trainer, size=(12, 6)) -> None:
    history = trainer.state.log_history

    plt.figure(figsize=size)
    plt.plot(
        [it["loss"] for it in history if "loss" in it],
        label="Training Loss",
    )
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_model_progress(full_report, size=(12, 6)):
    plt.figure(figsize=size)
    plt.plot(full_report["eval_f1"], label="F1 score")
    plt.title("Eval F1 score")
    plt.xlabel("Run")
    plt.ylabel("F1")
    plt.legend()
    plt.show()
