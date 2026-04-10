import matplotlib.pyplot as plt
import numpy as np

def plot_all_results(results, cm, labels):
    models = list(results.keys())

    accuracy = [results[m]["accuracy"] for m in models]
    precision = [results[m]["precision"] for m in models]
    recall = [results[m]["recall"] for m in models]
    f1 = [results[m]["f1"] for m in models]

    x = np.arange(len(models))
    width = 0.2

    # ---------------------------
    # COMPARISON BAR GRAPH
    # ---------------------------
    plt.figure(figsize=(10, 5))

    plt.bar(x - 1.5*width, accuracy, width, label="Accuracy")
    plt.bar(x - 0.5*width, precision, width, label="Precision")
    plt.bar(x + 0.5*width, recall, width, label="Recall")
    plt.bar(x + 1.5*width, f1, width, label="F1-Score")

    plt.xticks(x, models)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison on METR-LA Dataset")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------------------------
    # CONFUSION MATRIX (DIFFUSION)
    # ---------------------------
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()

    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Diffusion Model Confusion Matrix (500 METR-LA Samples)")
    plt.tight_layout()
    plt.show()
