import pandas as pd
import numpy as np
from sklearn.decomposition import FastICA, PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
import os


def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1]  # All columns except the last
    y = df.iloc[:, -1]  # Last column as target
    return X, y


def apply_ica(X, max_components):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reconstruction_errors = []
    for n_components in range(1, max_components + 1):
        ica = FastICA(n_components=n_components, random_state=42)
        X_ica_reduced = ica.fit_transform(X_scaled)
        X_ica_reconstructed = ica.inverse_transform(X_ica_reduced)
        reconstruction_errors.append(mean_squared_error(X_scaled, X_ica_reconstructed))

    kurtosis_scores = [
        np.mean(np.abs(kurtosis(ica.fit_transform(X_scaled))))
        for ica in [
            FastICA(n_components=i, random_state=42)
            for i in range(1, max_components + 1)
        ]
    ]
    n_components = np.argmax(kurtosis_scores) + 1
    ica = FastICA(n_components=n_components, random_state=42)
    X_ica_reduced = ica.fit_transform(X_scaled)

    return X_ica_reduced, kurtosis_scores, n_components, reconstruction_errors


def apply_pca(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1

    reconstruction_errors = []
    for i in range(1, len(explained_variance_ratio) + 1):
        pca_temp = PCA(n_components=i)
        X_pca_reduced = pca_temp.fit_transform(X_scaled)
        X_pca_reconstructed = pca_temp.inverse_transform(X_pca_reduced)
        reconstruction_errors.append(mean_squared_error(X_scaled, X_pca_reconstructed))

    pca = PCA(n_components=n_components)
    X_pca_reduced = pca.fit_transform(X_scaled)

    return X_pca_reduced, cumulative_variance_ratio, n_components, reconstruction_errors


def apply_rp(X, max_components):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    def compute_reconstruction_error(n_components):
        rp = GaussianRandomProjection(n_components=n_components, random_state=42)
        X_proj = rp.fit_transform(X_scaled)
        X_rec = np.dot(X_proj, rp.components_)
        return mean_squared_error(X_scaled, X_rec)

    n_samples, n_features = X_scaled.shape
    min_components = int(np.ceil(np.log2(n_samples)))
    component_range = range(min_components, min(max_components, n_features) + 1)
    reconstruction_errors = [compute_reconstruction_error(i) for i in component_range]
    n_components = min_components + np.argmin(np.diff(reconstruction_errors)) + 1
    rp = GaussianRandomProjection(n_components=n_components, random_state=42)
    X_rp_reduced = rp.fit_transform(X_scaled)

    # Final reconstruction error
    reconstruction_error = compute_reconstruction_error(n_components)

    return (
        X_rp_reduced,
        reconstruction_errors,
        n_components,
        component_range,
        reconstruction_error,
    )


def plot_results(x, y, xlabel, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.close()


def plot_combined_reconstruction_errors(
    ica_errors, pca_errors, rp_errors, rp_component_range, dataset_name
):
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(ica_errors) + 1), ica_errors, label="ICA")
    plt.plot(range(1, len(pca_errors) + 1), pca_errors, label="PCA")
    plt.plot(rp_component_range, rp_errors, label="RP")
    plt.xlabel("Number of Components")
    plt.ylabel("Reconstruction Error")
    plt.title(f"Reconstruction Error vs. Number of Components - {dataset_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        f'charts/dimensionality-reduction/combined_reconstruction_errors_{dataset_name.replace(" ", "_").lower()}.png'
    )
    plt.close()


def save_reduced_dataset(X_reduced, y, filename):
    df_reduced = pd.DataFrame(
        X_reduced, columns=[f"Component_{i+1}" for i in range(X_reduced.shape[1])]
    )
    df_reduced["target"] = y
    df_reduced.to_csv(filename, index=False)
    print(f"Saved reduced dataset to {filename}")


def plot_component_histograms(X_reduced, n_components, title, filename):
    plt.figure(figsize=(12, 4))
    for i in range(min(3, n_components)):
        plt.subplot(1, 3, i + 1)
        plt.hist(X_reduced[:, i], bins=50)
        plt.title(f"Component {i+1}")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def process_dataset(file_path, dataset_name):
    print(f"\nProcessing {dataset_name} dataset...")
    X, y = load_and_preprocess(file_path)

    # Create directories if they don't exist
    os.makedirs("charts/dimensionality-reduction", exist_ok=True)
    os.makedirs("data/reduced", exist_ok=True)

    # ICA
    max_components_ica = min(X.shape[1], 50)
    X_ica_reduced, kurtosis_scores, n_components_ica, ica_reconstruction_errors = (
        apply_ica(X, max_components_ica)
    )
    plot_results(
        range(1, len(kurtosis_scores) + 1),
        kurtosis_scores,
        "Number of Components",
        "Average Kurtosis",
        f"Average Kurtosis vs. Number of Components - {dataset_name}",
        f'charts/dimensionality-reduction/ica_{dataset_name.replace(" ", "_").lower()}.png',
    )
    plot_component_histograms(
        X_ica_reduced,
        n_components_ica,
        f"ICA Component Histograms - {dataset_name}",
        f'charts/dimensionality-reduction/ica_histograms_{dataset_name.replace(" ", "_").lower()}.png',
    )
    save_reduced_dataset(
        X_ica_reduced,
        y,
        f'data/reduced/ica_{dataset_name.replace(" ", "_").lower()}.csv',
    )
    print(f"ICA - Number of components chosen: {n_components_ica}")
    print(f"ICA - Shape of reduced dataset: {X_ica_reduced.shape}")
    print(
        f"ICA - Final reconstruction error: {ica_reconstruction_errors[n_components_ica - 1]}"
    )

    # PCA
    (
        X_pca_reduced,
        cumulative_variance_ratio,
        n_components_pca,
        pca_reconstruction_errors,
    ) = apply_pca(X)
    plot_results(
        range(1, len(cumulative_variance_ratio) + 1),
        cumulative_variance_ratio,
        "Number of Components",
        "Cumulative Explained Variance Ratio",
        f"Explained Variance Ratio vs. Number of Components - {dataset_name}",
        f'charts/dimensionality-reduction/pca_{dataset_name.replace(" ", "_").lower()}.png',
    )
    plot_component_histograms(
        X_pca_reduced,
        n_components_pca,
        f"PCA Component Histograms - {dataset_name}",
        f'charts/dimensionality-reduction/pca_histograms_{dataset_name.replace(" ", "_").lower()}.png',
    )
    save_reduced_dataset(
        X_pca_reduced,
        y,
        f'data/reduced/pca_{dataset_name.replace(" ", "_").lower()}.csv',
    )
    print(f"PCA - Number of components chosen: {n_components_pca}")
    print(f"PCA - Shape of reduced dataset: {X_pca_reduced.shape}")
    print(
        f"PCA - Final reconstruction error: {pca_reconstruction_errors[n_components_pca - 1]}"
    )

    # Random Projection
    max_components_rp = min(X.shape[1], 100)
    (
        X_rp_reduced,
        rp_reconstruction_errors,
        n_components_rp,
        component_range,
        rp_reconstruction_error,
    ) = apply_rp(X, max_components_rp)
    plot_results(
        component_range,
        rp_reconstruction_errors,
        "Number of Components",
        "Reconstruction Error",
        f"Reconstruction Error vs. Number of Components - {dataset_name}",
        f'charts/dimensionality-reduction/rp_{dataset_name.replace(" ", "_").lower()}.png',
    )
    plot_component_histograms(
        X_rp_reduced,
        n_components_rp,
        f"RP Component Histograms - {dataset_name}",
        f'charts/dimensionality-reduction/rp_histograms_{dataset_name.replace(" ", "_").lower()}.png',
    )
    save_reduced_dataset(
        X_rp_reduced, y, f'data/reduced/rp_{dataset_name.replace(" ", "_").lower()}.csv'
    )
    print(f"RP - Number of components chosen: {n_components_rp}")
    print(f"RP - Shape of reduced dataset: {X_rp_reduced.shape}")
    print(f"RP - Final reconstruction error: {rp_reconstruction_error}")

    # Adjust component range to be consistent across methods
    max_components = max(
        len(ica_reconstruction_errors),
        len(pca_reconstruction_errors),
        len(rp_reconstruction_errors),
    )
    component_range = range(1, max_components + 1)

    # Adjust reconstruction errors to match the same length
    ica_reconstruction_errors = np.pad(
        ica_reconstruction_errors,
        (0, max_components - len(ica_reconstruction_errors)),
        "edge",
    )
    pca_reconstruction_errors = np.pad(
        pca_reconstruction_errors,
        (0, max_components - len(pca_reconstruction_errors)),
        "edge",
    )
    rp_reconstruction_errors = np.pad(
        rp_reconstruction_errors,
        (0, max_components - len(rp_reconstruction_errors)),
        "edge",
    )

    # Plot combined reconstruction errors
    plot_combined_reconstruction_errors(
        ica_reconstruction_errors,
        pca_reconstruction_errors,
        rp_reconstruction_errors,
        component_range,
        dataset_name,
    )


if __name__ == "__main__":
    file_paths = ["data/credit_card_default_tw.csv", "data/diabetes_balanced.csv"]
    dataset_names = ["Credit Card Default", "Diabetes"]

    for file_path, dataset_name in zip(file_paths, dataset_names):
        process_dataset(file_path, dataset_name)
