# ---------------- Fully Optimized EDA App with CNN-EQIC and Deep Analysis ----------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from scipy.stats import zscore
from datetime import datetime
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
        self.emotional_weight = 1.0
        self.last_spike_time = None
        self.decay_rate = decay_rate

    def quantum_inspired_distance(self, input_pattern):
        diff = np.abs(input_pattern - self.position)
        dist = np.sqrt(np.sum(diff ** 2))
        decay = np.exp(-self.age / self.decay_rate)
        return (np.exp(-2.0 * dist) + 0.5 / (1 + 0.9 * dist)) * decay

    def update_spike_time(self):
        self.last_spike_time = datetime.now()

# ---------------- Neural Network ----------------
class HybridNeuralNetwork:
    def __init__(self, working_memory_capacity=20, decay_rate=100.0):
        self.units = []
        self.episodic_memory = EpisodicMemory()
        self.working_memory = WorkingMemory(capacity=working_memory_capacity)
        self.decay_rate = decay_rate

    def generate_unit(self, position):
        unit = HybridNeuralUnit(position, decay_rate=self.decay_rate)
        self.units.append(unit)
        return unit

    def process_input(self, input_pattern):
        if not self.units:
            return self.generate_unit(input_pattern), 0.0

        similarities = [(unit, unit.quantum_inspired_distance(input_pattern)) for unit in self.units]
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

# ---------------- Parameter Tuning ----------------
def evaluate_network(network, input_data, window_size=5):
    scores = []
    for i in range(window_size, len(input_data)):
        input_seq = input_data[i - window_size:i].flatten()
        _, sim = network.process_input(input_seq)
        scores.append(sim)
    return np.mean(scores)

def tune_parameters(input_data, param_grid):
    best_score = -np.inf
    best_params = {}

    for params in param_grid:
        network = HybridNeuralNetwork(**params)
        score = evaluate_network(network, input_data)
        if score > best_score:
            best_score = score
            best_params = params
    return best_params, best_score

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Optimized EDA with CNN-EQIC", layout="wide")
st.title("\U0001F50D Deep Exploratory Data Analysis with CNN-EQIC & Parameter Tuning")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)

    st.markdown("### Preview of Data")
    st.dataframe(df.head())

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = st.multiselect("Select features for analysis", numerical_cols, default=numerical_cols[:4])
    window_size = st.slider("Window Size", 2, 20, 5)

    imputer = SimpleImputer()
    clean_data = imputer.fit_transform(df[selected_features])
    scaled_data = MinMaxScaler().fit_transform(clean_data)

    # ---------------- Parameter Tuning ----------------
    param_grid = [{'working_memory_capacity': c, 'decay_rate': d} for c in [10, 20, 30] for d in [50.0, 100.0, 150.0]]
    best_params, best_score = tune_parameters(scaled_data, param_grid)
    st.success(f"Best Parameters: {best_params}, Score: {best_score:.4f}")

    # ---------------- Run Analysis ----------------
    network = HybridNeuralNetwork(**best_params)
    similarities, timestamps = [], []
    for i in range(window_size, len(scaled_data)):
        input_seq = scaled_data[i - window_size:i].flatten()
        _, sim = network.process_input(input_seq)
        similarities.append(sim)
        timestamps.append(i)

    # ---------------- Visualize Similarity ----------------
    st.markdown("### Pattern Similarity Timeline")
    fig1, ax1 = plt.subplots()
    ax1.plot(timestamps, similarities, label='Similarity', color='blue')
    ax1.axhline(0.6, linestyle='--', color='red', label='Threshold')
    ax1.set_title("Similarity Over Time")
    st.pyplot(fig1)

    # ---------------- CNN Usage ----------------
    usage = [unit.usage_count for unit in network.units]
    st.markdown("### CNN Unit Usage")
    st.bar_chart(usage)

    # ---------------- Clustering & Centroid Analysis ----------------
    patterns = np.array([p for e in network.episodic_memory.episodes.values() for p in e['patterns']])
    if len(patterns) > 0:
        # KMeans
        kmeans = KMeans(n_clusters=min(5, len(patterns)))
        labels = kmeans.fit_predict(patterns)
        st.markdown("### KMeans Clustering")
        st.dataframe(pd.DataFrame({'Cluster': labels}))

        fig2, ax2 = plt.subplots()
        scatter = ax2.scatter(patterns[:, 0], patterns[:, 1], c=labels, cmap='viridis')
        ax2.set_title("Clustered Patterns (KMeans)")
        st.pyplot(fig2)

        centroids_df = pd.DataFrame(kmeans.cluster_centers_, columns=[f"F{i+1}" for i in range(kmeans.cluster_centers_.shape[1])])
        st.markdown("### Cluster Centroids")
        st.dataframe(centroids_df)

        # DBSCAN
        st.markdown("### DBSCAN Clustering")
        dbscan = DBSCAN(eps=0.5, min_samples=5).fit(patterns)
        db_labels = dbscan.labels_
        fig3, ax3 = plt.subplots()
        scatter_db = ax3.scatter(patterns[:, 0], patterns[:, 1], c=db_labels, cmap='tab10')
        ax3.set_title("Clustered Patterns (DBSCAN)")
        st.pyplot(fig3)

    # ---------------- Correlation ----------------
    st.markdown("### Correlation Heatmap")
    corr = pd.DataFrame(clean_data, columns=selected_features).corr()
    fig4, ax4 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax4)
    st.pyplot(fig4)

    # ---------------- Inverse Relationships ----------------
    st.markdown("### Strong Inverse Correlations")
    inverse = corr[corr < -0.5].stack().reset_index()
    inverse.columns = ['Feature1', 'Feature2', 'Correlation']
    inverse = inverse[inverse['Feature1'] != inverse['Feature2']]
    st.dataframe(inverse)

    # ---------------- PCA ----------------
    st.markdown("### PCA Projection")
    pcs = PCA(n_components=2).fit_transform(scaled_data)
    fig5, ax5 = plt.subplots()
    ax5.scatter(pcs[:, 0], pcs[:, 1], alpha=0.6)
    ax5.set_title("PCA (2 Components)")
    st.pyplot(fig5)

    # ---------------- Mutual Information ----------------
    st.markdown("### Mutual Information")
    mi = mutual_info_regression(clean_data, clean_data[:, 0])
    mi_df = pd.DataFrame({'Feature': selected_features, 'MI Score vs ' + selected_features[0]: mi})
    st.dataframe(mi_df)

    # ---------------- Outliers ----------------
    st.markdown("### Outlier Detection")
    z_scores = np.abs(zscore(clean_data))
    outliers = pd.DataFrame(clean_data[(z_scores > 3).any(axis=1)], columns=selected_features)
    st.dataframe(outliers)

    # ---------------- Export ----------------
    save_path = os.path.expanduser(r"C:\\Users\\oliva\\OneDrive\\Documents\\Excel doc\\CNNanalysis.xlsx")
    with pd.ExcelWriter(save_path) as writer:
        df[selected_features].to_excel(writer, sheet_name='Clean Data', index=False)
        pd.DataFrame({'Timestamp': timestamps, 'Similarity': similarities}).to_excel(writer, sheet_name='Similarity', index=False)
        pd.DataFrame(usage, columns=['Usage']).to_excel(writer, sheet_name='Usage', index=False)
        corr.to_excel(writer, sheet_name='Correlation')
        if 'centroids_df' in locals():
            centroids_df.to_excel(writer, sheet_name='Centroids')
        mi_df.to_excel(writer, sheet_name='Mutual Info', index=False)
        outliers.to_excel(writer, sheet_name='Outliers', index=False)
    st.success(f"Results saved to: {save_path}")
