import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os


def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1]  # All columns except the last one
    y = df.iloc[:, -1]  # Last column as the target variable
    return X, y


def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def find_optimal_clusters(X, max_clusters=10, method="both"):
    n_clusters_range = range(2, max_clusters + 1)
    silhouette_scores = {"em": [], "kmeans": []}

    for n_clusters in n_clusters_range:
        if method in ["em", "both"]:
            gmm = GaussianMixture(n_components=n_clusters, random_state=7)
            cluster_labels = gmm.fit_predict(X)
            silhouette_scores["em"].append(silhouette_score(X, cluster_labels))

        if method in ["kmeans", "both"]:
            kmeans = KMeans(n_clusters=n_clusters, random_state=33, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            silhouette_scores["kmeans"].append(silhouette_score(X, cluster_labels))

    optimal_clusters = {
        "em": (
            n_clusters_range[np.argmax(silhouette_scores["em"])]
            if "em" in silhouette_scores
            else None
        ),
        "kmeans": (
            n_clusters_range[np.argmax(silhouette_scores["kmeans"])]
            if "kmeans" in silhouette_scores
            else None
        ),
    }
    return optimal_clusters, silhouette_scores


def train_model(X, n_clusters, method):
    if method == "em":
        model = GaussianMixture(n_components=n_clusters, random_state=42)
    elif method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    model.fit(X)
    return model


def evaluate_model(model, X):
    labels = model.predict(X)
    silhouette_avg = silhouette_score(X, labels)
    return silhouette_avg


def plot_silhouette_scores(n_clusters_range, silhouette_scores, dataset_name, method):
    plt.figure(figsize=(10, 6))
    plt.plot(n_clusters_range, silhouette_scores)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title(
        f"Silhouette Score vs Number of Clusters - {dataset_name} ({method.upper()})"
    )

    os.makedirs(f"charts/clustering/{method}", exist_ok=True)
    plt.savefig(
        f"charts/clustering/{method}/silhouette_scores_{method}_{dataset_name}.png"
    )
    plt.close()


def calculate_variance_explained(X, max_clusters=10):
    variance_explained = []
    tss = np.sum((X - X.mean(axis=0)) ** 2)

    for n_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X)
        variance_explained.append((tss - kmeans.inertia_) / tss * 100)

    return variance_explained


def plot_elbow_curve(n_clusters_range, variance_explained, dataset_name):
    plt.figure(figsize=(10, 6))
    plt.plot(n_clusters_range, variance_explained, marker="o")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Percent of Variance Explained")
    plt.title(f"Elbow Method - {dataset_name} (K-Means)")

    os.makedirs("charts/clustering/kmeans", exist_ok=True)
    plt.savefig(f"charts/clustering/kmeans/elbow_curve_kmeans_{dataset_name}.png")
    plt.close()


def save_clustered_data(X, y, model, scaler, dataset_name, method):
    # Get cluster assignments and probabilities/distances
    cluster_labels = model.predict(X)
    if method == "em":
        cluster_probs = model.predict_proba(X)
    else:
        distances = model.transform(X)

    # Create DataFrame with original features
    original_data = pd.DataFrame(scaler.inverse_transform(X), columns=X.columns)

    # Add cluster assignments and probabilities/distances
    clustered_data = original_data.copy()
    clustered_data["cluster"] = cluster_labels
    if method == "em":
        for i in range(model.n_components):
            clustered_data[f"prob_cluster_{i}"] = cluster_probs[:, i]
    else:
        for i in range(model.n_clusters):
            clustered_data[f"distance_cluster_{i}"] = distances[:, i]

    # Add target variable
    clustered_data["target"] = y

    # Save to CSV
    os.makedirs("data/clustered", exist_ok=True)
    output_path = f"data/clustered/{dataset_name}_clustered_{method}.csv"
    clustered_data.to_csv(output_path, index=False)
    print(f"Saved clustered data to {output_path}")


# def save_clustered_data(X, y, model, scaler, dataset_name, method):
#     # Get cluster assignments and probabilities/distances
#     cluster_labels = model.predict(X)
#     if method == "em":
#         cluster_probs = model.predict_proba(X)
#     else:
#         distances = model.transform(X)

#     # Create DataFrame with original features
#     original_data = pd.DataFrame(
#         scaler.inverse_transform(X), columns=[f"feature_{i}" for i in range(X.shape[1])]
#     )

#     # Add cluster assignments and probabilities/distances
#     clustered_data = original_data.copy()
#     clustered_data["cluster"] = cluster_labels
#     if method == "em":
#         for i in range(model.n_components):
#             clustered_data[f"prob_cluster_{i}"] = cluster_probs[:, i]
#     else:
#         for i in range(model.n_clusters):
#             clustered_data[f"distance_cluster_{i}"] = distances[:, i]

#     # Add target variable
#     clustered_data["target"] = y

#     # Save to CSV
#     os.makedirs("data/clustered/new", exist_ok=True)
#     output_path = f"data/clustered/new/{dataset_name}_clustered_{method}.csv"
#     clustered_data.to_csv(output_path, index=False)
#     print(f"Saved clustered data to {output_path}")


def plot_comparison_chart(n_clusters_range, silhouette_scores, dataset_name):
    plt.figure(figsize=(12, 6))
    plt.plot(n_clusters_range, silhouette_scores["em"], label="EM", marker="o")
    plt.plot(n_clusters_range, silhouette_scores["kmeans"], label="K-Means", marker="s")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title(f"EM vs K-Means Clustering Comparison - {dataset_name}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    os.makedirs("charts/clustering/comparison", exist_ok=True)
    plt.savefig(f"charts/clustering/comparison/em_vs_kmeans_{dataset_name}.png")
    plt.close()


def main(file_path):
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]

    # Load and preprocess dataå
    X, y = load_data(file_path)
    X_scaled, scaler = preprocess_data(X)

    # Find optimal number of clusters
    max_clusters = 10
    optimal_clusters, silhouette_scores = find_optimal_clusters(X_scaled, max_clusters)

    # Plot comparison chart
    plot_comparison_chart(range(2, max_clusters + 1), silhouette_scores, dataset_name)

    for method in ["em", "kmeans"]:
        print(f"\nAnalyzing {method.upper()} for dataset: {dataset_name}")

        # Plot and save silhouette scores
        plot_silhouette_scores(
            range(2, max_clusters + 1), silhouette_scores[method], dataset_name, method
        )

        # Train model with optimal number of clusters
        model = train_model(X_scaled, optimal_clusters[method], method)

        # Evaluate the model
        silhouette_avg = evaluate_model(model, X_scaled)

        print(f"Optimal number of clusters: {optimal_clusters[method]}")
        print(f"Silhouette score: {silhouette_avg:.4f}")

        # Additional analysis: compare cluster assignments with target variable
        cluster_labels = model.predict(X_scaled)
        cluster_df = pd.DataFrame({"Cluster": cluster_labels, "Target": y})
        print("\nCluster distribution:")
        print(
            cluster_df.groupby("Cluster")["Target"]
            .value_counts(normalize=True)
            .unstack()
        )

        # Save clustered data
        save_clustered_data(X, y, model, scaler, dataset_name, method)

        if method == "kmeans":
            # Calculate and plot elbow curve
            variance_explained = calculate_variance_explained(X_scaled, max_clusters)
            plot_elbow_curve(
                range(1, max_clusters + 1), variance_explained, dataset_name
            )


if __name__ == "__main__":
    file_paths = ["data/credit_card_default_tw.csv", "data/diabetes_balanced.csv"]

    for file_path in file_paths:
        print(f"\nAnalyzing dataset: {file_path}")
        main(file_path)
