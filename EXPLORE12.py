# ---------------- Advanced CNN-EQIC EDA with Interpretation & Auto-Analysis ----------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import zscore
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression
from datetime import datetime
import os
import time

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
        self.connections = []  # Inter-unit links

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

# ---------------- Main Analysis Function ----------------
def run_analysis(df, selected, window_size, save_path):
    clean = SimpleImputer().fit_transform(df[selected])
    scaled = MinMaxScaler().fit_transform(clean)

    # Parameter grid for tuning
    param_grid = [
        {'working_memory_capacity': c, 'decay_rate': d}
        for c in [10, 20] for d in [50.0, 100.0]
    ]
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

    patterns = [p for e in net.episodic_memory.episodes.values() for p in e['patterns']]

    if patterns:
        patterns = np.array(patterns)
        kmeans = KMeans(n_clusters=min(5, len(patterns))).fit(patterns)
        dbscan = DBSCAN(eps=0.4, min_samples=5).fit(patterns)
        pcs = PCA(n_components=2).fit_transform(patterns)

        # Forecasting last similarities
        forecast_values = forecast_next(similarities[-10:], 3)

        # Z-score outliers in cleaned data
        zs = np.abs(zscore(clean))
        outliers = pd.DataFrame(clean)[(zs > 3).any(axis=1)]

        # Save to Excel
        with pd.ExcelWriter(save_path) as writer:
            df[selected].to_excel(writer, sheet_name='Clean Data', index=False)
            pd.DataFrame({"Timestamps": timestamps, "Similarities": similarities}).to_excel(writer, sheet_name='Similarities', index=False)
            pd.DataFrame({'Cluster': kmeans.labels_}).to_excel(writer, sheet_name='KMeans Clusters', index=False)
            pd.DataFrame({'DBSCAN Labels': dbscan.labels_}).to_excel(writer, sheet_name='DBSCAN Labels', index=False)
            pd.DataFrame(pcs, columns=['PC1', 'PC2']).to_excel(writer, sheet_name='PCA', index=False)
            outliers.to_excel(writer, sheet_name='Outliers', index=False)
            pd.DataFrame([u.usage_count for u in net.units], columns=['Unit Usage Count']).to_excel(writer, sheet_name='Unit Usage', index=False)

        return {
            'best_params': best_params,
            'best_score': best_score,
            'similarities': similarities,
            'timestamps': timestamps,
            'unit_usage': [u.usage_count for u in net.units],
            'patterns': patterns,
            'kmeans_labels': kmeans.labels_,
            'dbscan_labels': dbscan.labels_,
            'pca_components': pcs,
            'forecast': forecast_values,
            'outliers': outliers,
            'clean_data': clean,
            'selected_features': selected
        }
    else:
        return None

# ---------------- Interpretation Function ----------------
def interpret_results(results):
    st.markdown("## 🔍 Interpretation of Analysis Results")
    if results is None:
        st.warning("No patterns stored in episodic memory. Not enough data or processing issues.")
        return

    st.markdown(f"- **Best Hyperparameters:** Working Memory Capacity = {results['best_params']['working_memory_capacity']}, Decay Rate = {results['best_params']['decay_rate']}")
    st.markdown(f"- **Best Similarity Score:** {results['best_score']:.4f}")
    st.markdown(f"- **Total Neural Units Generated:** {len(results['unit_usage'])}")
    st.markdown(f"- **Unit Usage Distribution:** Units with higher usage indicate frequent or significant pattern matches.")

    st.markdown(f"- **KMeans Clusters:** {len(np.unique(results['kmeans_labels']))} clusters detected. Clusters indicate grouping of similar patterns in the dataset.")
    st.markdown(f"- **DBSCAN Outliers:** {np.sum(results['dbscan_labels'] == -1)} detected as anomalies or noise, representing unusual or rare patterns.")
    st.markdown(f"- **PCA Explained:** Visualization of high-dimensional pattern clusters in 2D to observe group separations.")

    st.markdown(f"- **Forecasting:** Next 3 similarity values predicted to be approximately: {np.round(results['forecast'], 4)}. Useful for anticipating pattern stability or changes.")
    st.markdown(f"- **Outlier Rows:** {len(results['outliers'])} rows detected as outliers via Z-score method in original features.")

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Advanced CNN-EQIC EDA + Interpretation + Auto-Analysis", layout="wide")
st.title("📊 CNN-EQIC: Dynamic Clustering + Forecasting + Anomaly Detection")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    st.markdown("### Data Preview")
    st.dataframe(df.head())

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected = st.multiselect("Select features", numerical_cols, default=numerical_cols[:4])
    window_size = st.slider("Window Size", 2, 20, 5)

    save_path = os.path.expanduser(r"C:\\Users\\oliva\\OneDrive\\Documents\\Excel doc\\CNNanalysis.xlsx")

    # Auto-analysis option
    auto_analysis = st.checkbox("Enable Auto-Analysis (rerun every 60 seconds)", value=False)

    def perform_full_analysis():
        with st.spinner("Running full analysis..."):
            results = run_analysis(df, selected, window_size, save_path)
            if results:
                # Display charts and tables
                st.markdown("### Pattern Similarity Over Time")
                st.line_chart(pd.DataFrame({"Similarity": results['similarities']}, index=results['timestamps']))

                st.markdown("### CNN Unit Usage Frequency")
                st.bar_chart(results['unit_usage'])

                st.markdown("### Clustering and Anomaly Detection")
                cluster_df = pd.DataFrame({
                    'Pattern Index': np.arange(len(results['patterns'])),
                    'KMeans Cluster': results['kmeans_labels'],
                    'DBSCAN Label': results['dbscan_labels']
                })
                st.dataframe(cluster_df)

                st.markdown("### Forecasting Next 3 Similarity Steps")
                st.write(np.round(results['forecast'], 4))

                st.markdown("### PCA Projection")
                fig, ax = plt.subplots()
                scatter = ax.scatter(results['pca_components'][:, 0], results['pca_components'][:, 1], c=results['kmeans_labels'], cmap='Set1')
                ax.set_title("PCA Clustering Projection")
                st.pyplot(fig)

                st.markdown("### Correlation Matrix")
                fig2, ax2 = plt.subplots()
                sns.heatmap(pd.DataFrame(results['clean_data'], columns=results['selected_features']).corr(), annot=True, cmap="coolwarm", ax=ax2)
                st.pyplot(fig2)

                st.markdown("### Z-score Outliers")
                st.dataframe(results['outliers'])

                # Interpret results
                interpret_results(results)
            else:
                st.warning("Not enough patterns stored to perform analysis.")

    perform_full_analysis()

    if auto_analysis:
        st.markdown("### Auto-Analysis Enabled: The analysis will rerun every 60 seconds.")
        while True:
            time.sleep(60)
            st.experimental_rerun()
else:
    st.info("Please upload a CSV or Excel file to start analysis.")
