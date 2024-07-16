import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os
import time


def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)

    X = data.iloc[:, :-1].values  # Features
    Y = data.iloc[:, -1].values  # Target (last column)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=144
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

    start_time = time.time()

    train_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        train_outputs = (outputs > 0.5).float()
        accuracy = (train_outputs == Y_train).sum().item() / Y_train.shape[0]
        train_accuracies.append(accuracy)

    wall_clock_time = time.time() - start_time

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

    return accuracy, class_report, wall_clock_time, train_accuracies


def create_bar_plot(data, y_key, title, ylabel, filename):
    plt.figure(figsize=(12, 6))
    plt.bar(data.index, data[y_key])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"charts/nn/{filename}.png")
    plt.close()


def run_model_on_dataset(
    file_path, lr, num_epochs, layer1_size, layer2_size, layer3_size
):
    X_train, X_test, Y_train, Y_test = load_and_preprocess_data(file_path)
    accuracy, class_report, wall_clock_time, train_accuracies = (
        train_and_evaluate_model(
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
    )
    return {
        "accuracy": accuracy,
        "f1_score": class_report["weighted avg"]["f1-score"],
        "wall_clock_time": wall_clock_time,
        "train_accuracies": train_accuracies,
    }


def plot_accuracy(all_results):
    epochs = range(1, len(next(iter(all_results.values()))["train_accuracies"]) + 1)
    plt.figure(figsize=(12, 6))

    colors = plt.cm.tab20.colors  # Get a list of colors for plotting

    for idx, (dataset_name, results) in enumerate(all_results.items()):
        plt.plot(
            epochs,
            results["train_accuracies"],
            label=f"{dataset_name} Accuracy",
            color=colors[idx % len(colors)],
        )

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training Accuracy for All Datasets")
    plt.savefig(os.path.join("charts/nn/combined_nn"))
    plt.close()


def main():
    lr = 0.0015
    num_epochs = 55
    layer1_size = 110
    layer2_size = 55
    layer3_size = 10

    # lr = 0.01
    # num_epochs = 35
    # layer1_size = 120
    # layer2_size = 48
    # layer3_size = 16

    datasets = {
        "Original": "data/credit_card_default_tw.csv",
        "ICA": "data/reduced/ica_credit_card_default.csv",
        "PCA": "data/reduced/pca_credit_card_default.csv",
        "RP": "data/reduced/rp_credit_card_default.csv",
        "EM": "data/clustered/credit_card_default_tw_clustered_em.csv",
        "KM": "data/clustered/credit_card_default_tw_clustered_kmeans.csv",
    }

    all_results = {}
    for name, file_path in datasets.items():
        print(f"Running model on {name} dataset...")
        all_results[name] = run_model_on_dataset(
            file_path, lr, num_epochs, layer1_size, layer2_size, layer3_size
        )

    # Create a DataFrame for easy comparison
    df_results = pd.DataFrame(all_results).T
    df_results = df_results.sort_values("accuracy", ascending=False)

    print("\nResults:")
    print(df_results)

    # Save results to CSV
    # df_results.to_csv("nn_comparison_results.csv")

    # Create bar plots
    create_bar_plot(
        df_results,
        "accuracy",
        "Model Accuracy Comparison",
        "Accuracy",
        "accuracy_comparison.png",
    )
    create_bar_plot(
        df_results,
        "wall_clock_time",
        "Model Training Time Comparison",
        "Wall Clock Time (seconds)",
        "wall_clock_time_comparison.png",
    )

    # Plot accuracy for all datasets
    plot_accuracy(all_results)


if __name__ == "__main__":
    main()
