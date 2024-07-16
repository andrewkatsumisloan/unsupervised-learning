import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100, 25),
            nn.ReLU(),
            nn.Linear(25, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


def load_and_preprocess(file_path):
    data = pd.read_csv(file_path)
    X = data.drop("target", axis=1).values
    y = data["target"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor


def calculate_accuracy(model, X, y):
    with torch.no_grad():
        y_pred = model(X)
        y_pred_class = (y_pred > 0.5).float()
        accuracy = accuracy_score(y.numpy(), y_pred_class.numpy())
    return accuracy


def train_and_evaluate(X_train, X_test, y_train, y_test, model_name):
    input_size = X_train.shape[1]
    model = NeuralNetwork(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    num_epochs = 65
    batch_size = 64

    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i : i + batch_size]
            batch_y = y_train[i : i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Calculate and store accuracies
        train_accuracy = calculate_accuracy(model, X_train, y_train)
        test_accuracy = calculate_accuracy(model, X_test, y_test)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}"
            )

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred_class = (y_pred > 0.5).float()
        accuracy = accuracy_score(y_test.numpy(), y_pred_class.numpy())

    print(f"\nResults for {model_name}:")
    print(f"Final Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test.numpy(), y_pred_class.numpy()))

    # Plot accuracy vs epochs
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, num_epochs + 1), test_accuracies, label="Test Accuracy")
    plt.title(f"Accuracy vs Epochs - {model_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(
        f'charts/nn/clustering/accuracy_vs_epochs_{model_name.lower().replace(" ", "_")}.png'
    )
    plt.close()

    return model, accuracy, train_accuracies, test_accuracies


# Load and process datasets
em_X_train, em_X_test, em_y_train, em_y_test = load_and_preprocess(
    "data/clustered/credit_card_default_tw_clustered_em.csv"
)
km_X_train, km_X_test, km_y_train, km_y_test = load_and_preprocess(
    "data/clustered/credit_card_default_tw_clustered_kmeans.csv"
)

# Train and evaluate models
em_model, em_accuracy, em_train_acc, em_test_acc = train_and_evaluate(
    em_X_train, em_X_test, em_y_train, em_y_test, "EM Clustering"
)
km_model, km_accuracy, km_train_acc, km_test_acc = train_and_evaluate(
    km_X_train, km_X_test, km_y_train, km_y_test, "K-Means Clustering"
)

# Compare accuracies
models = ["EM Clustering", "K-Means Clustering"]
accuracies = [em_accuracy, km_accuracy]

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies)
plt.title("Model Accuracy Comparison (Clustered Data)")
plt.xlabel("Clustering Technique")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f"{v:.4f}", ha="center")
plt.savefig("charts/nn/clustering/model_comparison_clustered.png")
plt.close()

print("\nComparison of model accuracies:")
for model, accuracy in zip(models, accuracies):
    print(f"{model}: {accuracy:.4f}")

print("\nModel comparison plot saved as 'model_comparison_clustered.png'")
print("Accuracy vs Epochs plots saved for each clustering technique")
