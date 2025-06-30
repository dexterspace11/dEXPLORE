# ---------------- Final Streamlit App with CNN-EQIC + Advanced Clustering ----------------
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

# ---------------- Evaluation & Tuning ----------------
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
st.set_page_config(page_title="Hybrid CNN-EQIC Analyzer", layout="wide")
st.title("\U0001F4A1 Hybrid CNN-EQIC EDA with Clustering & Tuning")

uploaded_file = st.file_uploader("Upload CSV/Excel file", type=["csv", "xlsx"])
if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    st.dataframe(df.head())
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = st.multiselect("Select features", numerical_cols, default=numerical_cols[:4])
    window_size = st.slider("Window size", 2, 20, 5)

    clean = SimpleImputer().fit_transform(df[selected_features])
    scaled = MinMaxScaler().fit_transform(clean)

    # Parameter tuning
    param_grid = [{'working_memory_capacity': c, 'decay_rate': d} for c in [10, 20] for d in [50.0, 100.0]]
    best_params, best_score = tune_parameters(scaled, param_grid)
    st.success(f"Best Parameters: {best_params}, Score: {best_score:.4f}")

    network = HybridNeuralNetwork(**best_params)
    similarities, timestamps = [], []
    for i in range(window_size, len(scaled)):
        input_seq = scaled[i - window_size:i].flatten()
        _, sim = network.process_input(input_seq)
        similarities.append(sim)
        timestamps.append(i)

    st.subheader("Pattern Similarity Timeline")
    fig, ax = plt.subplots()
    ax.plot(timestamps, similarities, label="Similarity", color="orange")
    ax.axhline(y=0.6, color='red', linestyle='--')
    st.pyplot(fig)

    all_patterns = np.array([p for e in network.episodic_memory.episodes.values() for p in e['patterns']])
    if len(all_patterns) > 1:
        st.subheader("Hybrid CNN-Based Clustering")
        # KMeans
        kmeans = KMeans(n_clusters=min(5, len(all_patterns)))
        k_labels = kmeans.fit_predict(all_patterns)
        st.write("KMeans Cluster Centroids:", kmeans.cluster_centers_)
        # DBSCAN
        db = DBSCAN(eps=0.5, min_samples=5).fit(all_patterns)
        db_labels = db.labels_
        # Plot
        fig2, ax2 = plt.subplots()
        ax2.scatter(all_patterns[:, 0], all_patterns[:, 1], c=k_labels, cmap='Set1')
        ax2.set_title("CNN-EQIC KMeans Clusters")
        st.pyplot(fig2)

        # Correlation
        st.subheader("Correlation Matrix")
        corr = pd.DataFrame(clean, columns=selected_features).corr()
        fig3, ax3 = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax3)
        st.pyplot(fig3)

        # PCA
        st.subheader("PCA Projection")
        pcs = PCA(n_components=2).fit_transform(scaled)
        fig4, ax4 = plt.subplots()
        ax4.scatter(pcs[:, 0], pcs[:, 1], alpha=0.6)
        st.pyplot(fig4)

        # Mutual Info
        st.subheader("Mutual Information")
        mi_scores = mutual_info_regression(clean, clean[:, 0])
        st.dataframe(pd.DataFrame({'Feature': selected_features, 'MI Score': mi_scores}))

        # Outliers
        st.subheader("Outlier Detection (Z-score)")
        z_scores = np.abs(zscore(clean))
        outliers = df[(z_scores > 3).any(axis=1)]
        st.write(f"Detected {outliers.shape[0]} outliers")
        st.dataframe(outliers)

        # Save Output
        save_path = os.path.expanduser(r"C:\\Users\\oliva\\OneDrive\\Documents\\Excel doc\\CNNanalysis.xlsx")
        with pd.ExcelWriter(save_path) as writer:
            df.to_excel(writer, index=False, sheet_name="Raw")
            pd.DataFrame(all_patterns).to_excel(writer, sheet_name="Patterns")
            pd.DataFrame(pcs, columns=['PC1', 'PC2']).to_excel(writer, sheet_name="PCA")
            pd.DataFrame({'Similarity': similarities}).to_excel(writer, sheet_name="Similarity")
            pd.DataFrame(kmeans.cluster_centers_).to_excel(writer, sheet_name="Centroids")
            outliers.to_excel(writer, sheet_name="Outliers")
        st.success(f"Results saved to: {save_path}")