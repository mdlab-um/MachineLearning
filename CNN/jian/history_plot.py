import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_training_history(csv_filename):
    """
    Load training history from CSV and create plots.

    Args:
        csv_filename: Path to the CSV file (e.g., 'training_history_20241121_143022.csv')
    """
    # Load the data
    history_df = pd.read_csv(csv_filename)

    # Create side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss
    ax1.plot(history_df['epoch'], history_df['train_loss'], 'b-',
             label='Train Loss', linewidth=2)
    ax1.plot(history_df['epoch'], history_df['val_loss'], 'r-',
             label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(history_df['epoch'], history_df['train_acc'], 'b-',
             label='Train Accuracy', linewidth=2)
    ax2.plot(history_df['epoch'], history_df['val_acc'], 'r-',
             label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    # output_filename = csv_filename.replace('.csv', '_plot.png')
    # plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    # print(f"Plot saved to: {output_filename}")

    # Show plot
    plt.show()

    # Print summary statistics
    print(f"\nTraining Summary:")
    print(f"Best Train Accuracy: {history_df['train_acc'].max():.4f} at epoch {history_df['train_acc'].idxmax() + 1}")
    print(f"Best Val Accuracy: {history_df['val_acc'].max():.4f} at epoch {history_df['val_acc'].idxmax() + 1}")
    print(f"Final Train Loss: {history_df['train_loss'].iloc[-1]:.4f}")
    print(f"Final Val Loss: {history_df['val_loss'].iloc[-1]:.4f}")


# Usage
if __name__ == "__main__":
    # Replace with your CSV filename
    plot_training_history(sys.argv[1])
