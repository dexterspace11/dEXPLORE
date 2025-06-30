# ---------------- Enhanced CNN-EQIC Cluster Analysis App with Interpretive Insights ----------------

# Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression
from datetime import datetime
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, Alignment

# ---------------- Memory Structures ----------------
class EpisodicMemory:
    def __init__(self):
        self.episodes = {}
        self.current_episode = None

    def create_episode(self, timestamp):
        self.current_episode = timestamp
        self.episodes[timestamp] = {'patterns': [], 'emotional_tags': []}

    def store_pattern(self, pattern, emotional_tag):
        if self.current_episode is None:
            self.create_episode(datetime.now())
        self.episodes[self.current_episode]['patterns'].append(pattern)
        self.episodes[self.current_episode]['emotional_tags'].append(emotional_tag)

class WorkingMemory:
    def __init__(self, capacity=20):
        self.capacity = capacity
        self.short_term_patterns = []
        self.temporal_context = []

    def store(self, pattern, timestamp):
        if len(self.short_term_patterns) >= self.capacity:
            self.short_term_patterns.pop(0)
            self.temporal_context.pop(0)
        self.short_term_patterns.append(pattern)
        self.temporal_context.append(timestamp)

# ---------------- Neural Unit ----------------
class HybridNeuralUnit:
    def __init__(self, position, decay_rate=100.0):
        self.position = position
        self.age = 0
        self.usage_count = 0
        self.last_spike_time = None
        self.decay_rate = decay_rate
        self.connections = []

    def distance(self, input_pattern):
        diff = np.abs(input_pattern - self.position)
        dist = np.sqrt(np.sum(diff ** 2))
        decay = np.exp(-self.age / self.decay_rate)
        return (np.exp(-2.0 * dist) + 0.5 / (1 + 0.9 * dist)) * decay

    def update_spike_time(self):
        self.last_spike_time = datetime.now()

    def communicate(self, other_units):
        for other in other_units:
            if other != self:
                dist = np.linalg.norm(self.position - other.position)
                if dist < 0.5 and other not in self.connections:
                    self.connections.append(other)

# ---------------- Neural Network ----------------
class HybridNeuralNetwork:
    def __init__(self, working_memory_capacity=20, decay_rate=100.0):
        self.units = []
        self.episodic_memory = EpisodicMemory()
        self.working_memory = WorkingMemory(capacity=working_memory_capacity)
        self.decay_rate = decay_rate

    def generate_unit(self, position):
        unit = HybridNeuralUnit(position, decay_rate=self.decay_rate)
        unit.communicate(self.units)
        self.units.append(unit)
        return unit

    def process_input(self, input_pattern):
        if not self.units:
            return self.generate_unit(input_pattern), 0.0

        similarities = [(unit, unit.distance(input_pattern)) for unit in self.units]
        similarities.sort(key=lambda x: x[1], reverse=True)
        best_unit, best_similarity = similarities[0]

        self.episodic_memory.store_pattern(input_pattern, best_similarity)
        self.working_memory.store(input_pattern, datetime.now())
        best_unit.age = 0
        best_unit.usage_count += 1
        best_unit.update_spike_time()

        if best_similarity < 0.6:
            return self.generate_unit(input_pattern), best_similarity

        return best_unit, best_similarity

# ---------------- Forecasting ----------------
def forecast_next(values, steps=1):
    model = LinearRegression()
    X = np.arange(len(values)).reshape(-1, 1)
    y = values
    model.fit(X, y)
    X_pred = np.arange(len(values), len(values) + steps).reshape(-1, 1)
    return model.predict(X_pred)

# ---------------- Clustering Evaluation ----------------
def evaluate_clustering(X, labels):
    try:
        if len(np.unique(labels)) > 1 and len(X) > len(np.unique(labels)):
            sil = silhouette_score(X, labels)
            db = davies_bouldin_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
            return {'Silhouette': sil, 'Davies-Bouldin': db, 'Calinski-Harabasz': ch}
        else:
            return {'Silhouette': np.nan, 'Davies-Bouldin': np.nan, 'Calinski-Harabasz': np.nan}
    except:
        return {'Silhouette': np.nan, 'Davies-Bouldin': np.nan, 'Calinski-Harabasz': np.nan}

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Enhanced CNN-EQIC Cluster App", layout="wide")
st.title("🔍 Hybrid CNN-EQIC Clustering Explorer")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    st.markdown("### Data Preview")
    st.dataframe(df.head())

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected = st.multiselect("Select features", numerical_cols, default=numerical_cols[:4])
    window_size = st.slider("Window Size", 2, 20, 5)

    clean = SimpleImputer().fit_transform(df[selected])
    scaled = MinMaxScaler().fit_transform(clean)

    param_grid = [{'working_memory_capacity': c, 'decay_rate': d} for c in [10, 20] for d in [50.0, 100.0]]
    def evaluate(net, data):
        scores = []
        for i in range(window_size, len(data)):
            pattern = data[i - window_size:i].flatten()
            _, sim = net.process_input(pattern)
            scores.append(sim)
        return np.mean(scores)

    best_score = -np.inf
    best_params = {}
    for params in param_grid:
        net = HybridNeuralNetwork(**params)
        score = evaluate(net, scaled)
        if score > best_score:
            best_params, best_score = params, score

    net = HybridNeuralNetwork(**best_params)
    similarities, timestamps = [], []
    for i in range(window_size, len(scaled)):
        pattern = scaled[i - window_size:i].flatten()
        _, sim = net.process_input(pattern)
        similarities.append(sim)
        timestamps.append(i)

    st.success(f"Best Params: {best_params}, Best Similarity: {best_score:.4f}")

    patterns = [p for e in net.episodic_memory.episodes.values() for p in e['patterns']]
    if patterns:
        patterns = np.array(patterns)
        pcs = PCA(n_components=2).fit_transform(patterns)

        st.markdown("### Neural Cluster Summary")
        cnn_labels = [i for i in range(len(net.units)) for _ in range(net.units[i].usage_count)]
        cnn_metrics = evaluate_clustering(pcs, cnn_labels[:len(pcs)])
        kmeans = KMeans(n_clusters=5).fit(patterns)
        kmeans_metrics = evaluate_clustering(pcs, kmeans.labels_)
        db = DBSCAN(eps=0.4, min_samples=5).fit(patterns)
        dbscan_metrics = evaluate_clustering(pcs, db.labels_)

        summary_df = pd.DataFrame({
            'Algorithm': ['CNN-EQIC', 'KMeans', 'DBSCAN'],
            'Clusters': [len(net.units), len(np.unique(kmeans.labels_)), len(np.unique(db.labels_)) - (1 if -1 in db.labels_ else 0)],
            'Outliers': [0, 0, sum(db.labels_ == -1)],
            'Silhouette': [cnn_metrics['Silhouette'], kmeans_metrics['Silhouette'], dbscan_metrics['Silhouette']],
            'Davies-Bouldin': [cnn_metrics['Davies-Bouldin'], kmeans_metrics['Davies-Bouldin'], dbscan_metrics['Davies-Bouldin']],
            'Calinski-Harabasz': [cnn_metrics['Calinski-Harabasz'], kmeans_metrics['Calinski-Harabasz'], dbscan_metrics['Calinski-Harabasz']]
        })
        st.dataframe(summary_df)

        st.markdown("### PCA Projections")
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        ax[0].scatter(pcs[:, 0], pcs[:, 1], c=cnn_labels[:len(pcs)], cmap='tab20', s=20)
        ax[0].set_title("CNN-EQIC Clusters")
        ax[1].scatter(pcs[:, 0], pcs[:, 1], c=kmeans.labels_, cmap='Set2', s=20)
        ax[1].set_title("KMeans Clusters")
        ax[2].scatter(pcs[:, 0], pcs[:, 1], c=db.labels_, cmap='coolwarm', s=20)
        ax[2].set_title("DBSCAN Clusters")
        st.pyplot(fig)

        st.markdown("### Feature Relationships by Cluster")
        patterns_df = pd.DataFrame(patterns)
        patterns_df['KMeans_Label'] = kmeans.labels_
        patterns_df['DBSCAN_Label'] = db.labels_
        patterns_df['CNN_Label'] = cnn_labels[:len(patterns)]

        st.markdown("#### KMeans Feature Importance")
        for label in np.unique(kmeans.labels_):
            st.write(f"Cluster {label} Mean Profile")
            st.bar_chart(patterns_df[patterns_df['KMeans_Label'] == label].mean())

        st.markdown("#### CNN-EQIC Feature Importance")
        for label in np.unique(cnn_labels[:len(patterns)]):
            st.write(f"CNN Unit {label} Mean Profile")
            st.bar_chart(patterns_df[patterns_df['CNN_Label'] == label].mean())

        st.markdown("### Correlation Heatmap (All Data)")
        fig, ax = plt.subplots()
        sns.heatmap(pd.DataFrame(clean, columns=selected).corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No patterns found in episodic memory.")
