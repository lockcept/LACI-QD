"""
plot methods for visualizing training logs or arena score.
"""

import pandas as pd
import matplotlib.pyplot as plt


def plot_train_loss(
    csv_path="logs/train.csv",
    output_image_path="logs/train_loss_graph.png",
):
    """
    Reads a CSV file containing train logs and plots the loss graph with dividers
    whenever num_iter changes.
    """

    # Load the CSV file
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File '{csv_path}' not found.")
        return

    # Ensure the expected columns are present
    expected_columns = {"num_iter", "num_epoch", "pi_loss", "v_loss"}
    if not expected_columns.issubset(data.columns):
        print(f"Error: CSV file must contain the following columns: {expected_columns}")
        return

    # Sort data by 'num_iter' and 'num_epoch' and reset index
    data = data.sort_values(by=["num_iter", "num_epoch"]).reset_index(drop=True)

    # Create the main figure and axis
    _, ax1 = plt.subplots(figsize=(10, 6))

    # Plot pi_loss on the primary y-axis
    ax1.plot(
        data.index,
        data["pi_loss"],
        label="Policy Loss",
        color="blue",
        marker="o",
        markersize=1,
        alpha=0.3,
    )
    ax1.set_ylabel("Policy Loss", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Create a secondary y-axis for v_loss
    ax2 = ax1.twinx()
    ax2.plot(
        data.index,
        data["v_loss"],
        label="Value Loss",
        color="orange",
        marker="s",
        markersize=1,
        alpha=0.3,
    )
    ax2.set_ylabel("Value Loss", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    # Add vertical lines and labels where num_iter changes
    previous_num_iter = None
    for idx, num_iter in enumerate(data["num_iter"]):
        if num_iter != previous_num_iter:
            ax1.axvline(x=idx, color="gray", linestyle="--", linewidth=0.8)
            ax1.text(
                idx,
                ax1.get_ylim()[0],
                f"{num_iter}",
                verticalalignment="bottom",
                fontsize=10,
                color="black",
            )
            previous_num_iter = num_iter

    # Add labels, legend, and title
    ax1.set_xlabel("Index")
    plt.title("Policy and Value Loss")
    ax1.grid(True)

    # Save the plot as an image
    plt.tight_layout()
    plt.savefig(output_image_path)
    plt.close()

    print(f"Loss graph saved as '{output_image_path}'")


if __name__ == "__main__":
    plot_train_loss()
