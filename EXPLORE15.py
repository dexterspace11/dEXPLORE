# ----------------- Modified CNN-EQIC Clustering Focus + KMeans/DBSCAN Comparison with Excel Export ------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from scipy.stats import zscore
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

# ---------------- Excel Export ----------------
def export_cluster_results(save_path, df, selected, usage_counts, unit_assignments, pcs, kmeans_labels, dbscan_labels, cluster_summary):
    wb = openpyxl.Workbook()

    # Sheet 1: Clean Data
    ws1 = wb.active
    ws1.title = "Clean Data"
    for r in dataframe_to_rows(df[selected], index=False, header=True):
        ws1.append(r)
    for cell in ws1[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal='center')

    # Sheet 2: CNN-EQIC Unit Usage
    ws2 = wb.create_sheet("CNN Unit Usage")
    for i, count in enumerate(usage_counts):
        ws2.append([f"Unit {i}", count])

    # Sheet 3: Cluster Assignments
    ws3 = wb.create_sheet("Cluster Assignments")
    for i, unit in enumerate(unit_assignments):
        ws3.append([i, unit])

    # Sheet 4: PCA Projection
    ws4 = wb.create_sheet("PCA Projection")
    ws4.append(['PC1', 'PC2', 'CNN Unit', 'KMeans Label', 'DBSCAN Label'])
    for i in range(len(pcs)):
        row = list(pcs[i]) + [unit_assignments[i], kmeans_labels[i], dbscan_labels[i]]
        ws4.append(row)

    # Sheet 5: Cluster Averages
    ws5 = wb.create_sheet("CNN Cluster Averages")
    for r in dataframe_to_rows(cluster_summary, index=True, header=True):
        ws5.append(r)

    # Sheet 6: Summary
    ws6 = wb.create_sheet("Summary")
    ws6.append(["Total Patterns", len(unit_assignments)])
    ws6.append(["CNN Units", len(usage_counts)])
    ws6.append(["KMeans Clusters", len(set(kmeans_labels))])
    ws6.append(["DBSCAN Outliers", sum(np.array(dbscan_labels) == -1)])

    wb.save(save_path)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="CNN-EQIC Clustering Engine", layout="wide")
st.title("🧠 CNN-EQIC: Memory-Based Cluster Analysis & Comparison")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    st.markdown("### Preview")
    st.dataframe(df.head())

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected = st.multiselect("Select features", numerical_cols, default=numerical_cols[:4])
    window_size = st.slider("Window Size", 2, 20, 5)

    clean = SimpleImputer().fit_transform(df[selected])
    scaled = MinMaxScaler().fit_transform(clean)

    st.markdown("### Running CNN-EQIC Neural Clustering")
    net = HybridNeuralNetwork(working_memory_capacity=20, decay_rate=100.0)
    similarities, unit_assignments = [], []

    for i in range(window_size, len(scaled)):
        pattern = scaled[i - window_size:i].flatten()
        unit, sim = net.process_input(pattern)
        similarities.append(sim)
        unit_assignments.append(net.units.index(unit))

    patterns = [p for e in net.episodic_memory.episodes.values() for p in e['patterns']]
    pcs = PCA(n_components=2).fit_transform(patterns)

    st.markdown("### CNN-EQIC Cluster Centroid Visualization")
    fig, ax = plt.subplots()
    scatter = ax.scatter(pcs[:, 0], pcs[:, 1], c=unit_assignments, cmap='tab10')
    ax.set_title("CNN-EQIC Neural Cluster Projection (PCA)")
    st.pyplot(fig)

    st.markdown("### Neural Unit Usage Frequency")
    usage_counts = [u.usage_count for u in net.units]
    st.bar_chart(usage_counts)

    st.markdown("### CNN-EQIC Cluster Profiles")
    cluster_df = pd.DataFrame(patterns)
    cluster_df['Unit'] = unit_assignments
    cluster_summary = cluster_df.groupby('Unit').mean()
    st.dataframe(cluster_summary)

    st.markdown("### KMeans & DBSCAN Comparison")
    kmeans = KMeans(n_clusters=min(5, len(patterns))).fit(patterns)
    db = DBSCAN(eps=0.4, min_samples=5).fit(patterns)

    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
    ax2[0].scatter(pcs[:, 0], pcs[:, 1], c=kmeans.labels_, cmap='Set1')
    ax2[0].set_title("KMeans Clusters")
    ax2[1].scatter(pcs[:, 0], pcs[:, 1], c=db.labels_, cmap='Spectral')
    ax2[1].set_title("DBSCAN Clusters")
    st.pyplot(fig2)

    st.markdown("### Export Analysis to Excel")
    export_path = "C:/Users/oliva/OneDrive/Documents/Excel doc/CNN_EQIC_Clusters.xlsx"
    if st.button("Export to Excel"):
        export_cluster_results(
            export_path,
            df=df,
            selected=selected,
            usage_counts=usage_counts,
            unit_assignments=unit_assignments,
            pcs=pcs,
            kmeans_labels=kmeans.labels_,
            dbscan_labels=db.labels_,
            cluster_summary=cluster_summary
        )
        st.success(f"Exported to {export_path}")