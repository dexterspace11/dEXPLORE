# --- Updated Streamlit App with Adaptive Cluster Count and Enhanced DNN-EQIC Interpretation ---
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from itertools import product
import warnings
warnings.filterwarnings("ignore")

# -------------------- Utility Functions --------------------
def preprocess_df(df, target_column=None):
    df = df.copy()
    df = df.drop(columns=df.select_dtypes(include=['datetime64', 'datetime']).columns, errors='ignore')
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
    df = df.loc[:, df.nunique(dropna=False) > 1]

    target = df[target_column] if target_column and target_column in df.columns else None
    if target_column and target_column in df.columns:
        df = df.drop(columns=[target_column])

    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df = df.fillna(df.median(numeric_only=True))

    if target_column and target is not None:
        df[target_column] = target

    usable_cols = [col for col in df.columns if col != target_column]
    if not usable_cols:
        fallback = ['open', 'high', 'low', 'close', 'volume', 'VWAP']
        fallback_cols = [col for col in fallback if col in df.columns]
        if not fallback_cols:
            raise ValueError("No features available after preprocessing. Please check your dataset.")
        return df[fallback_cols + ([target_column] if target_column else [])]

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

def hyperparameter_search(data, cluster_range, param_grid):
    best_score = -np.inf
    best_params = None
    best_labels = None
    best_n_clusters = None
    for n_clusters in cluster_range:
        for alpha, beta, gamma, kappa in product(param_grid['alpha'], param_grid['beta'], param_grid['gamma'], param_grid['kappa']):
            labels, _, _ = enhanced_quantum_clustering(
                data, n_clusters=n_clusters, alpha=alpha, beta=beta, gamma=gamma, kappa=kappa
            )
            try:
                score = silhouette_score(data, labels)
            except:
                score = -1
            if score > best_score:
                best_score = score
                best_params = {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'kappa': kappa}
                best_labels = labels
                best_n_clusters = n_clusters
    return best_labels, best_params, best_score, best_n_clusters

def interpret_clusters(df, features):
    interpretations = []
    stats = df.groupby('Cluster')[features].agg(['mean', 'std'])
    for cluster_id in sorted(df['Cluster'].unique()):
        desc = f"\n**Cluster {cluster_id} Summary:**\n"
        cluster_stats = stats.loc[cluster_id]
        top_mean = cluster_stats['mean'].sort_values(ascending=False).head(3)
        low_mean = cluster_stats['mean'].sort_values().head(3)

        desc += f"- High Feature Means: {', '.join(top_mean.index)}\n"
        desc += f"- Low Feature Means: {', '.join(low_mean.index)}\n"

        corrs = df[df['Cluster'] == cluster_id][features].corr()
        strong_corr = ((corrs > 0.8) | (corrs < -0.8)) & (corrs != 1.0)
        high_corr_pairs = [(i, j) for i in corrs.columns for j in corrs.columns if strong_corr.loc[i, j]]
        if high_corr_pairs:
            desc += f"- Strong Correlations: {', '.join([f'{a}<->{b}' for a,b in high_corr_pairs])}\n"
        else:
            desc += "- No strong intra-cluster correlations found.\n"

        interpretations.append(desc)
    return "\n".join(interpretations)

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="\U0001f9d0 Hybrid DNN-EQIC Clustering", layout="wide")
st.title(":brain: Hybrid DNN-EQIC Clustering with Interpretation")

uploaded_file = st.file_uploader("\U0001f4c4 Upload your dataset (CSV/XLSX)", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    df.columns = df.columns.astype(str)
    st.dataframe(df.head())

    target_column = st.selectbox("Select target column (optional)", ["None"] + list(df.columns))
    target_column = None if target_column == "None" else target_column
    auto_cluster = st.checkbox("Automatically determine optimal number of clusters", value=True)
    if not auto_cluster:
        n_clusters = st.slider("Manual number of clusters", 2, 10, 3)

    if st.button("\U0001f680 Start Clustering"):
        df_clean = preprocess_df(df, target_column)
        features = [col for col in df_clean.columns if col != target_column]
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(df_clean[features])

        param_grid = {'alpha': [1.0, 2.0], 'beta': [0.3, 0.5], 'gamma': [0.7, 0.9], 'kappa': [0.05, 0.1]}
        if auto_cluster:
            cluster_range = range(2, 8)
        else:
            cluster_range = [n_clusters]

        labels, best_params, sil_score, best_n_clusters = hyperparameter_search(X_scaled, cluster_range, param_grid)

        st.success(f"\u2705 Best Silhouette Score: {sil_score:.4f} with {best_n_clusters} clusters")
        st.json(best_params)

        labels, centroids, weights = enhanced_quantum_clustering(X_scaled, best_n_clusters, **best_params)
        df_clean['Cluster'] = labels

        st.markdown("### \U0001f4ca Cluster Metrics")
        st.write(f"Calinski-Harabasz Score: {calinski_harabasz_score(X_scaled, labels):.2f}")
        st.write(f"Davies-Bouldin Score: {davies_bouldin_score(X_scaled, labels):.2f}")

        st.markdown("### \U0001f4ca Cluster Sizes")
        st.write(df_clean['Cluster'].value_counts().reset_index().rename(columns={'index': 'Cluster', 'Cluster': 'Count'}))

        st.markdown("### \U0001f9e0 Cluster Feature Statistics")
        st.write(df_clean.groupby('Cluster')[features].agg(['mean', 'std', 'min', 'max']))

        st.markdown("### \U0001f52c Centroid Feature Analysis")
        centroids_df = pd.DataFrame(scaler.inverse_transform(centroids), columns=features)
        st.dataframe(centroids_df)

        st.markdown("### \U0001f9ea Correlation Heatmaps (Per Cluster)")
        for cluster in sorted(df_clean['Cluster'].unique()):
            st.markdown(f"#### Cluster {cluster}")
            corr = df_clean[df_clean['Cluster'] == cluster][features].corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, cmap='coolwarm', center=0, annot=False, ax=ax)
            st.pyplot(fig)

        st.markdown("### \U0001f4c8 Dimensionality Reduction")
        pca = PCA(n_components=2).fit_transform(X_scaled)
        tsne = TSNE(n_components=2, perplexity=30).fit_transform(X_scaled)

        fig1, ax1 = plt.subplots()
        for c in np.unique(labels):
            ax1.scatter(pca[labels == c, 0], pca[labels == c, 1], label=f"Cluster {c}")
        ax1.set_title("PCA Projection")
        ax1.legend()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        for c in np.unique(labels):
            ax2.scatter(tsne[labels == c, 0], tsne[labels == c, 1], label=f"Cluster {c}")
        ax2.set_title("t-SNE Projection")
        ax2.legend()
        st.pyplot(fig2)

        if len(features) <= 10:
            st.markdown("### \U0001f4c9 Pairwise Feature Distributions")
            plot_df = df_clean[features + ['Cluster']]
            pair_fig = sns.pairplot(plot_df, hue='Cluster', palette='tab10')
            st.pyplot(pair_fig)

        st.markdown("### \U0001fdf0 Interpretation Report")
        interpretation_text = interpret_clusters(df_clean, features)
        st.markdown(interpretation_text)

        csv = df_clean.to_csv(index=False).encode('utf-8')
        st.download_button("\U0001f4c5 Download Clustered Data", csv, file_name="clustered_output.csv")