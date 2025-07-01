# ---------------- Enhanced CNN-EQIC EDA with Narrative Cluster Analysis for Research ----------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from datetime import datetime
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, Alignment
import os

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

# ---------------- Narrative Generator ----------------
def generate_cluster_narrative(centroids, features):
    narrative = []
    for idx, row in enumerate(centroids):
        summary = f"Cluster {idx}: "
        trait_summary = []
        for f, val in zip(features, row):
            level = "high" if val > 0.75 else "moderate" if val > 0.5 else "low"
            trait_summary.append(f"{level} {f}")
        summary += ", ".join(trait_summary)
        narrative.append(summary)
    return "\n".join(narrative)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="CNN-EQIC EDA", layout="wide")
st.title("📊 CNN-EQIC: Research-Grade Cluster Interpretation")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    st.markdown("### Dataset Preview")
    st.dataframe(df.head())

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected = st.multiselect("Select features", numerical_cols, default=numerical_cols[:4])
    window_size = st.slider("Window size", 2, 20, 5)

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

    best_score, best_params = -np.inf, {}
    for params in param_grid:
        net = HybridNeuralNetwork(**params)
        score = evaluate(net, scaled)
        if score > best_score:
            best_params, best_score = params, score

    net = HybridNeuralNetwork(**best_params)
    similarities = []
    for i in range(window_size, len(scaled)):
        pattern = scaled[i - window_size:i].flatten()
        _, sim = net.process_input(pattern)
        similarities.append(sim)

    patterns = [p for e in net.episodic_memory.episodes.values() for p in e['patterns']]
    if patterns:
        patterns = np.array(patterns)
        k = st.slider("KMeans Clusters (k)", 2, 10, 5)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(patterns)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        silhouette = silhouette_score(patterns, labels)
        db = davies_bouldin_score(patterns, labels)
        ch = calinski_harabasz_score(patterns, labels)

        st.markdown("### Clustering Evaluation")
        st.metric("Silhouette", f"{silhouette:.3f}")
        st.metric("Davies-Bouldin", f"{db:.3f}")
        st.metric("Calinski-Harabasz", f"{ch:.1f}")

        pca = PCA(n_components=2)
        pcs = pca.fit_transform(patterns)
        df_pca = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
        df_pca['Cluster'] = labels

        fig, ax = plt.subplots()
        sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_pca, palette='tab10', ax=ax)
        ax.set_title("Cluster Projection via PCA")
        st.pyplot(fig)

        st.markdown("### Cluster Narratives")
        cluster_narrative = generate_cluster_narrative(centroids, [f"F{i+1}" for i in range(centroids.shape[1])])
        st.text(cluster_narrative)

        export_path = r"C:\\Users\\oliva\\OneDrive\\Documents\\Excel doc\\DNNanalysis.xlsx"
        if st.button("Export Narrative & Clusters to Excel"):
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Cluster Narratives"
            for line in cluster_narrative.split("\n"):
                ws.append([line])
            wb.save(export_path)
            st.success(f"Results saved to {export_path}")
    else:
        st.warning("No patterns stored.")
