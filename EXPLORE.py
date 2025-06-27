# hybrid_dnn_eqic_clusters.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from itertools import product
import warnings
warnings.filterwarnings("ignore")

# -------------------- Utility Functions --------------------
def preprocess_df(df):
    df = df.copy()
    df = df.drop(columns=df.select_dtypes(include=['datetime64', 'datetime']).columns, errors='ignore')
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
    df = df.loc[:, df.nunique(dropna=False) > 1]
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df = df.fillna(df.median(numeric_only=True))
    return df

def normalize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std = np.where(std == 0, 1e-8, std)
    return (data - mean) / std

def calculate_distance(point, centroid, alpha, beta, gamma, weights):
    weighted_diff = weights * np.abs(point - centroid)
    dist = np.sqrt(np.sum(weighted_diff**2))
    exp_term = np.exp(-alpha * dist)
    inv_term = beta / (1 + gamma * dist)
    return exp_term + inv_term

def centroid_interaction(c1, c2, kappa):
    dist = np.linalg.norm(c1 - c2)
    return np.exp(-kappa * dist)

def update_dimension_weights(clusters, data):
    weights = np.ones(data.shape[1])
    epsilon = 1e-8
    for cluster in clusters:
        if cluster:
            cluster_data = np.array(cluster)
            var = np.var(cluster_data, axis=0)
            mean_var = np.mean(var) if np.mean(var) != 0 else epsilon
            weights *= var / mean_var
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
    sum_weights = np.sum(weights)
    return weights / sum_weights if sum_weights >= epsilon else np.ones_like(weights) / len(weights)

def update_centroids(clusters, centroids, gamma, kappa):
    new_centroids = []
    for i, cluster in enumerate(clusters):
        if cluster:
            mean = np.mean(cluster, axis=0)
            interaction = sum(
                centroid_interaction(centroids[i], centroids[j], kappa) * centroids[j]
                for j in range(len(centroids)) if j != i
            ) / (len(centroids) - 1)
            new_c = gamma * mean + (1 - gamma) * (centroids[i] + interaction)
            new_centroids.append(new_c)
        else:
            new_centroids.append(centroids[i])
    return np.array(new_centroids)

def assign_clusters(data, centroids, weights, alpha, beta, gamma):
    labels = []
    for point in data:
        dists = [calculate_distance(point, c, alpha, beta, gamma, weights) for c in centroids]
        labels.append(np.argmax(dists))
    return np.array(labels)

def enhanced_quantum_clustering(data, n_clusters=2, alpha=2.0, beta=0.5, gamma=0.9, kappa=0.1, tol=1e-4, max_iter=100):
    data = normalize_data(data)
    idx = np.random.choice(len(data), n_clusters, replace=False)
    centroids = data[idx]
    weights = np.ones(data.shape[1])

    for _ in range(max_iter):
        clusters = [[] for _ in range(n_clusters)]
        for x in data:
            dists = [calculate_distance(x, c, alpha, beta, gamma, weights) for c in centroids]
            clusters[np.argmax(dists)].append(x)

        if _ > 0:
            weights = update_dimension_weights(clusters, data)

        new_centroids = update_centroids(clusters, centroids, gamma, kappa)
        if np.max(np.abs(new_centroids - centroids)) < tol:
            break
        centroids = new_centroids

    labels = assign_clusters(data, centroids, weights, alpha, beta, gamma)
    return labels, centroids, weights

def cluster_summary(df, labels):
    df['Cluster'] = labels
    st.markdown("### 📊 Cluster Summary")
    st.write(df['Cluster'].value_counts().rename("Size"))
    st.dataframe(df.groupby('Cluster').mean().T)

def centroid_stats(centroids, features, scaler):
    unscaled = scaler.inverse_transform(centroids)
    df = pd.DataFrame(unscaled, columns=features)
    st.markdown("### 🎯 Cluster Centroids (Unscaled)")
    st.dataframe(df)

def cluster_feature_stats(df, labels):
    df['Cluster'] = labels
    st.markdown("### 📌 Cluster-wise Feature Statistics")
    for cluster_id in sorted(df['Cluster'].unique()):
        st.subheader(f"Cluster {cluster_id}")
        st.dataframe(df[df['Cluster'] == cluster_id].describe().T)

def cluster_correlations(df, labels):
    df['Cluster'] = labels
    st.markdown("### 🔍 Intra-cluster Feature Correlations")
    for cluster_id in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster_id].drop(columns='Cluster')
        st.subheader(f"Cluster {cluster_id} Correlation Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cluster_data.corr(), annot=False, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

def pairwise_distributions(df, labels):
    df['Cluster'] = labels
    st.markdown("### 📈 Pairwise Feature Distributions")
    try:
        import seaborn as sns
        sns.set(style="ticks")
        sampled_df = df.sample(n=min(500, len(df)), random_state=42)
        fig = sns.pairplot(sampled_df, hue='Cluster', corner=True)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not display pairplot: {e}")

# -------------------- Streamlit App --------------------
st.set_page_config(page_title="Hybrid DNN-EQIC Unsupervised Analyzer", layout="wide")
st.title("🔎 Hybrid DNN-EQIC Unsupervised Cluster Analyzer")

uploaded_file = st.file_uploader("📂 Upload your dataset (CSV/XLSX)", type=["csv", "xlsx"])
if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    df.columns = df.columns.astype(str)
    st.markdown("### 📄 Preview")
    st.dataframe(df.head())

    df_clean = preprocess_df(df)
    features = df_clean.columns.tolist()

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df_clean)

    st.markdown("### ⚙️ Clustering Parameters")
    n_clusters = st.slider("Number of Clusters", 2, 10, 3)
    param_grid = {'alpha': [1.0, 2.0], 'beta': [0.3, 0.5], 'gamma': [0.7, 0.9], 'kappa': [0.05, 0.1]}

    if st.button("🚀 Run Clustering"):
        labels, centroids, weights = enhanced_quantum_clustering(X_scaled, n_clusters)

        cluster_summary(df_clean.copy(), labels)
        centroid_stats(centroids, features, scaler)
        cluster_feature_stats(df_clean.copy(), labels)
        cluster_correlations(df_clean.copy(), labels)
        pairwise_distributions(df_clean.copy(), labels)

        st.markdown("### 📉 Clustering Metrics")
        st.write(f"Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")
        st.write(f"Calinski-Harabasz Score: {calinski_harabasz_score(X_scaled, labels):.2f}")
        st.write(f"Davies-Bouldin Score: {davies_bouldin_score(X_scaled, labels):.2f}")

        st.markdown("### 🧬 Dimensionality Reduction")
        pca = PCA(n_components=2)
        pca_proj = pca.fit_transform(X_scaled)
        tsne = TSNE(n_components=2, perplexity=30)
        tsne_proj = tsne.fit_transform(X_scaled)

        fig1, ax1 = plt.subplots()
        for c in np.unique(labels):
            ax1.scatter(pca_proj[labels == c, 0], pca_proj[labels == c, 1], label=f"Cluster {c}", alpha=0.6)
        ax1.set_title("PCA Projection")
        ax1.legend()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        for c in np.unique(labels):
            ax2.scatter(tsne_proj[labels == c, 0], tsne_proj[labels == c, 1], label=f"Cluster {c}", alpha=0.6)
        ax2.set_title("t-SNE Projection")
        ax2.legend()
        st.pyplot(fig2)

        st.markdown("### 💾 Download Clustered Data")
        df_out = df.copy()
        df_out['Cluster'] = labels
        csv = df_out.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Clustered CSV", csv, file_name="clustered_output.csv")
