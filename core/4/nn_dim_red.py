import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

sns.set_theme(style="whitegrid")

# Create the charts/nn directory if it doesn't exist
os.makedirs("charts/nn", exist_ok=True)


def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)

    X = data.iloc[:, :-1].values  # Features
    Y = data.iloc[:, -1].values  # Target (last column)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(1)

    return X_train, X_test, Y_train, Y_test


class ANN(nn.Module):
    def __init__(self, input_size, layer1_size, layer2_size, layer3_size):
        super(ANN, self).__init__()
        self.layer_1 = nn.Linear(input_size, layer1_size)
        self.layer_2 = nn.Linear(layer1_size, layer2_size)
        self.layer_3 = nn.Linear(layer2_size, layer3_size)
        self.layer_4 = nn.Linear(layer3_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.sigmoid(self.layer_4(x))
        return x


def train_and_evaluate_model(
    X_train,
    Y_train,
    X_test,
    Y_test,
    lr,
    num_epochs,
    layer1_size,
    layer2_size,
    layer3_size,
):
    model = ANN(X_train.shape[1], layer1_size, layer2_size, layer3_size)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_accuracies = []
    test_accuracies = []

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()

        outputs = model(X_train)
        loss = criterion(outputs, Y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_outputs = (outputs > 0.5).float()
        train_accuracy = (train_outputs == Y_train).sum().item() / Y_train.shape[0]
        train_accuracies.append(train_accuracy)

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, Y_test)

            test_outputs = (test_outputs > 0.5).float()
            test_accuracy = (test_outputs == Y_test).sum().item() / Y_test.shape[0]
            test_accuracies.append(test_accuracy)

        if (epoch + 1) % 1 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}"
            )

    end_time = time.time()
    wall_clock_time = end_time - start_time

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_outputs = (test_outputs > 0.5).float()
        accuracy = (test_outputs == Y_test).sum().item() / Y_test.shape[0]
        Y_test_np = Y_test.numpy()
        test_outputs_np = test_outputs.numpy()
        class_report = classification_report(
            Y_test_np,
            test_outputs_np,
            target_names=["Class 0", "Class 1"],
            output_dict=True,
        )
        return (
            accuracy,
            class_report,
            model,
            train_accuracies,
            test_accuracies,
            wall_clock_time,
        )


def plot_accuracies(train_accuracies, test_accuracies, num_epochs, filename):
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_accuracies, label="Training Accuracy")
    plt.plot(epochs, test_accuracies, label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Test Accuracy vs Epochs")
    plt.legend()
    plt.savefig(os.path.join("charts/nn/dimensionality-reduction", filename))
    plt.close()


def main():
    dim_reduction_methods = ["ica", "pca", "rp"]

    results = []

    for method in dim_reduction_methods:
        file_path = f"data/reduced/{method}_credit_card_default.csv"

        print(f"\nTraining on {method.upper()} reduced data:")

        X_train, X_test, Y_train, Y_test = load_and_preprocess_data(file_path)

        # lr = 0.055
        # num_epochs = 30
        # layer1_size = 32
        # layer2_size = 16
        # layer3_size = 1

        lr = 0.015
        num_epochs = 55
        layer1_size = 128
        layer2_size = 32
        layer3_size = 16

        (
            accuracy,
            class_report,
            model,
            train_accuracies,
            test_accuracies,
            wall_clock_time,
        ) = train_and_evaluate_model(
            X_train,
            Y_train,
            X_test,
            Y_test,
            lr,
            num_epochs,
            layer1_size,
            layer2_size,
            layer3_size,
        )

        print(f"Model accuracy: {accuracy:.4f}")
        print(f"Wall clock time: {wall_clock_time:.2f} seconds")
        print("Classification Report:")
        print(
            classification_report(
                Y_test.numpy(),
                (model(X_test) > 0.5).float().numpy(),
                target_names=["Class 0", "Class 1"],
            )
        )

        filename = f"{method}_credit_card_default_accuracy_plot.png"
        plot_accuracies(train_accuracies, test_accuracies, num_epochs, filename)

        # Save the classification report
        report_df = pd.DataFrame(class_report).transpose()
        # c

        results.append(
            {"method": method, "accuracy": accuracy, "wall_clock_time": wall_clock_time}
        )

    # Save summary results
    results_df = pd.DataFrame(results)
    # results_df.to_csv(os.path.join("charts/nn", "summary_results.csv"), index=False)


if __name__ == "__main__":
    main()
