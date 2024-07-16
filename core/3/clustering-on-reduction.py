import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os


def load_reduced_data(file_path):
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1]  # All columns except the last one
    y = df.iloc[:, -1]  # Last column as the target variable
    return X, y


def apply_kmeans(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, labels)
    inertia = kmeans.inertia_
    return silhouette_avg, inertia


def apply_em(X, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    labels = gmm.fit_predict(X)
    silhouette_avg = silhouette_score(X, labels)
    bic = gmm.bic(X)
    return silhouette_avg, bic


def analyze_clustering(X, max_clusters=10):
    n_clusters_range = range(2, max_clusters + 1)
    kmeans_results = {"silhouette": [], "inertia": []}
    em_results = {"silhouette": [], "bic": []}

    for n_clusters in n_clusters_range:
        # K-means
        sil, inertia = apply_kmeans(X, n_clusters)
        kmeans_results["silhouette"].append(sil)
        kmeans_results["inertia"].append(inertia)

        # EM
        sil, bic = apply_em(X, n_clusters)
        em_results["silhouette"].append(sil)
        em_results["bic"].append(bic)

    return n_clusters_range, kmeans_results, em_results


def plot_comparative_metrics(results, dataset_name):
    os.makedirs("charts/clustering-reduced", exist_ok=True)

    metrics = ["silhouette", "inertia/bic"]
    titles = ["Silhouette Score", "Inertia (K-means) / BIC (EM)"]

    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f"Clustering Metrics Comparison - {dataset_name}", fontsize=16)

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        for j, algorithm in enumerate(["kmeans", "em"]):  # Changed to lowercase
            ax = axs[i, j]
            for method, data in results.items():
                n_clusters_range = data["n_clusters_range"]
                if metric == "silhouette":
                    ax.plot(
                        n_clusters_range,
                        data[algorithm][metric],  # Removed .lower()
                        marker="o",
                        label=method,
                    )
                elif metric == "inertia/bic":
                    if algorithm == "kmeans":  # Changed to lowercase
                        ax.plot(
                            n_clusters_range,
                            data[algorithm]["inertia"],  # Removed .lower()
                            marker="o",
                            label=method,
                        )
                    else:
                        ax.plot(
                            n_clusters_range,
                            data[algorithm]["bic"],  # Removed .lower()
                            marker="o",
                            label=method,
                        )

            ax.set_xlabel("Number of Clusters")
            ax.set_ylabel(title)
            ax.set_title(
                f"{title} - {algorithm.capitalize()}"
            )  # Capitalize for display
            ax.legend()

    plt.tight_layout()
    plt.savefig(f"charts/clustering-reduced/comparative_metrics_{dataset_name}.png")
    plt.close()


def analyze_reduced_dataset(file_path):
    dataset_name = os.path.basename(file_path).split("_", 1)[1].replace(".csv", "")
    method = os.path.basename(file_path).split("_")[0].upper()

    print(f"\nAnalyzing {method} reduced dataset: {dataset_name}")

    X, y = load_reduced_data(file_path)

    # Analyze clustering
    n_clusters_range, kmeans_results, em_results = analyze_clustering(X)

    return method, {
        "n_clusters_range": n_clusters_range,
        "kmeans": kmeans_results,
        "em": em_results,
    }


def determine_best_combinations(results):
    best_combinations = []
    for dataset_name, dataset_results in results.items():
        for method, data in dataset_results.items():
            kmeans_best_silhouette = max(data["kmeans"]["silhouette"])
            em_best_silhouette = max(data["em"]["silhouette"])
            kmeans_best_clusters = data["n_clusters_range"][
                np.argmax(data["kmeans"]["silhouette"])
            ]
            em_best_clusters = data["n_clusters_range"][
                np.argmax(data["em"]["silhouette"])
            ]

            best_combinations.append(
                {
                    "Dataset": dataset_name,
                    "Method": method,
                    "Algorithm": "K-means",
                    "Best Silhouette": kmeans_best_silhouette,
                    "Best Clusters": kmeans_best_clusters,
                    "Inertia": data["kmeans"]["inertia"][kmeans_best_clusters - 2],
                }
            )
            best_combinations.append(
                {
                    "Dataset": dataset_name,
                    "Method": method,
                    "Algorithm": "EM",
                    "Best Silhouette": em_best_silhouette,
                    "Best Clusters": em_best_clusters,
                    "BIC": data["em"]["bic"][em_best_clusters - 2],
                }
            )

    best_df = pd.DataFrame(best_combinations)
    best_df = best_df.sort_values("Best Silhouette", ascending=False)
    return best_df


def main():
    reduced_file_paths = [
        "data/reduced/ica_credit_card_default.csv",
        "data/reduced/pca_credit_card_default.csv",
        "data/reduced/rp_credit_card_default.csv",
        "data/reduced/ica_diabetes.csv",
        "data/reduced/pca_diabetes.csv",
        "data/reduced/rp_diabetes.csv",
    ]

    results = {}

    for file_path in reduced_file_paths:
        method, data = analyze_reduced_dataset(file_path)
        dataset_name = os.path.basename(file_path).split("_", 1)[1].replace(".csv", "")
        if dataset_name not in results:
            results[dataset_name] = {}
        results[dataset_name][method] = data

    # Plot comparative metrics for each dataset
    for dataset_name, dataset_results in results.items():
        plot_comparative_metrics(dataset_results, dataset_name)

    # Determine best combinations
    best_combinations = determine_best_combinations(results)
    best_combinations.to_csv("best_clustering_combinations.csv", index=False)
    print("\nBest Clustering Combinations:")
    print(best_combinations)


if __name__ == "__main__":
    main()
