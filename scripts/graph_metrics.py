import matplotlib.pyplot as plt
import pandas as pd


def graph_model_metrics(output: pd.DataFrame) -> plt.Figure:
    """
    A function to graph the model's train and validation losses
    and metrics.

    :param output: The DataFrame output from `train_and_validate_model`
    :return: A Figure containing the model metrics and losses. To plot
        it inline in Jupyter Notebook, use `result.plot()`.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))

    output.plot.line(
        x='epoch',
        y=['train-loss', 'validation-loss'],
        ax=ax1,
    )
    ax1.set_title("Loss")

    output.plot.line(
        x='epoch',
        y=['train-metrics', 'validation-metrics'],
        ax=ax2,
    )
    ax2.set_title("Metrics")

    plt.close() # This prevents the graph from displaying twice
    return fig
