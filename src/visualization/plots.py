import os
import matplotlib.pyplot as plt
import torch


def plot_and_save_training_curves(metrics, experiment_dir):
    """
    Plots and saves training and validation curves for Loss, Accuracy, and F1 Score.

    Args:
        metrics (dict): Dictionary containing keys: 'train_losses', 'val_losses',
                        'val_accuracies', 'val_f1_scores'.
        experiment_dir (str): Directory path where the plots should be saved.
    """
    epochs = range(1, len(metrics['train_losses']) + 1)

    plt.figure(figsize=(15, 5))

    # ---- Plot Loss Curve ----
    plt.subplot(1, 3, 1)
    plt.plot(epochs, metrics['train_losses'], 'b-o', label='Train Loss')
    plt.plot(epochs, metrics['val_losses'], 'r-o', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # ---- Plot Accuracy Curve ----
    plt.subplot(1, 3, 2)
    plt.plot(epochs, metrics['val_accuracies'], 'g-o', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)

    # ---- Plot F1 Score Curve ----
    plt.subplot(1, 3, 3)
    plt.plot(epochs, metrics['val_f1_scores'], 'm-o', label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # ---- Save plot as image ----
    save_path = os.path.join(experiment_dir, 'training_curves.png')
    plt.savefig(save_path)

    # plt.show()

    print(f"Training curves saved at: {save_path}")


def visualize_channel_attention(model, experiments_dir, feature_names=None):
    """
    Visualize channel attention weights from the model.

    Args:
        model: Your trained LSTMClassifier model.
        feature_names: Optional list of feature names (e.g., sensor names like ['acc_x', 'acc_y', ...])
        save_path: Path to save the attention plot image. If None, the plot will just display.
    """

    # Extract attention weights
    with torch.no_grad():
        attn_weights = torch.sigmoid(model.channel_attention).cpu().numpy()

    num_channels = len(attn_weights)

    # Generate x-axis labels
    if feature_names is None:
        feature_names = [f"Channel {i}" for i in range(num_channels)]

    # Plot
    plt.figure(figsize=(10, 4))
    plt.bar(range(num_channels), attn_weights, tick_label=feature_names)
    plt.title("Learned Channel Attention Weights")
    plt.ylabel("Attention Weight (After Sigmoid)")
    plt.xlabel("Channels / Features")
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_path = os.path.join(experiments_dir, "attention_weights.png")
    if save_path:
        plt.savefig(save_path)
        print(f"Attention weights plot saved to: {save_path}")

    # plt.show()

