# ---------------- Fully Enhanced Unsupervised EDA App Using Hybrid DNN-EQIC ----------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from scipy.stats import zscore
from datetime import datetime
import io

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
    def __init__(self, position):
        self.position = position
        self.age = 0
        self.usage_count = 0
        self.emotional_weight = 1.0
        self.last_spike_time = None

    def quantum_inspired_distance(self, input_pattern):
        diff = np.abs(input_pattern - self.position)
        dist = np.sqrt(np.sum(diff ** 2))
        decay = np.exp(-self.age / 100.0)
        return (np.exp(-2.0 * dist) + 0.5 / (1 + 0.9 * dist)) * decay

    def update_spike_time(self):
        self.last_spike_time = datetime.now()

# ---------------- Neural Network ----------------
class HybridNeuralNetwork:
    def __init__(self):
        self.units = []
        self.episodic_memory = EpisodicMemory()
        self.working_memory = WorkingMemory()

    def generate_unit(self, position):
        unit = HybridNeuralUnit(position)
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

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Unsupervised EDA with Hybrid DNN-EQIC", layout="wide")
st.title("\U0001F50D Fully Enhanced EDA Using Cognitive Neural Pattern Discovery")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.markdown("### Preview of Data")
    st.dataframe(df.head())

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numerical_cols:
        st.warning("No numerical columns found!")
    else:
        selected_features = st.multiselect("Select features to analyze", numerical_cols, default=numerical_cols[:4])
        window_size = st.slider("Input Window Size", 2, 20, 5)

        imputer = SimpleImputer(strategy='mean')
        clean_data = imputer.fit_transform(df[selected_features])
        scaled_data = MinMaxScaler().fit_transform(clean_data)

        df_clean = pd.DataFrame(clean_data, columns=selected_features)

        network = HybridNeuralNetwork()
        similarities, timestamps = [], []

        for i in range(window_size, len(scaled_data)):
            input_seq = scaled_data[i - window_size:i].flatten()
            _, sim = network.process_input(input_seq)
            similarities.append(sim)
            timestamps.append(i)

        st.markdown("### Pattern Similarity Over Time")
        fig, ax = plt.subplots()
        ax.plot(timestamps, similarities, label='Pattern Similarity', color='orange')
        ax.axhline(y=0.6, color='red', linestyle='--', label='Generation Threshold')
        ax.set_ylabel("Similarity")
        ax.set_xlabel("Timestep")
        ax.set_title("Pattern Recognition Dynamics")
        ax.legend()
        st.pyplot(fig)

        st.markdown("### Episodic Memory Heatmap")
        all_patterns = np.array([p for e in network.episodic_memory.episodes.values() for p in e['patterns']])
        if len(all_patterns) > 0:
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            im = ax2.imshow(all_patterns, aspect='auto', cmap='viridis')
            ax2.set_title("Stored Pattern Encodings")
            fig2.colorbar(im)
            st.pyplot(fig2)

            st.markdown("### KMeans Clustering of Stored Patterns")
            n_clusters = min(5, len(all_patterns))
            kmeans = KMeans(n_clusters=n_clusters)
            labels = kmeans.fit_predict(all_patterns)
            cluster_df = pd.DataFrame(all_patterns)
            cluster_df['Cluster'] = labels
            st.dataframe(cluster_df)
            fig3, ax3 = plt.subplots()
            scatter = ax3.scatter(all_patterns[:, 0], all_patterns[:, 1], c=labels, cmap='Set1')
            ax3.set_title("Clustering of Neural Pattern Encodings")
            st.pyplot(fig3)

            st.markdown("### Correlation Matrix Among Selected Features")
            corr_matrix = df_clean.corr()
            fig4, ax4 = plt.subplots()
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax4)
            ax4.set_title("Correlation Matrix")
            st.pyplot(fig4)
            st.dataframe(corr_matrix)

            st.markdown("### Mutual Information Scores (Nonlinear Relationships)")
            try:
                mi_scores = mutual_info_regression(df_clean, df_clean.iloc[:, 0])
                mi_df = pd.DataFrame({'Feature': selected_features, 'MI Score vs ' + selected_features[0]: mi_scores})
                st.dataframe(mi_df)
            except ValueError as e:
                st.warning(f"Mutual Information could not be computed: {e}")

            st.markdown("### PCA Insights (Top 2 Components)")
            pca = PCA(n_components=2)
            pcs = pca.fit_transform(scaled_data)
            pca_df = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
            fig5, ax5 = plt.subplots()
            ax5.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.6)
            ax5.set_title("PCA Projection")
            st.pyplot(fig5)
            st.dataframe(pca_df.head())

            st.markdown("### Outlier Detection using Z-score")
            z_scores = np.abs(zscore(df_clean))
            outlier_flags = (z_scores > 3).any(axis=1)
            outliers_df = df_clean[outlier_flags]
            st.dataframe(outliers_df)
            st.markdown(f"**Detected {outliers_df.shape[0]} outliers.**")

            st.markdown("### CNN-Derived Feature Insights")
            usage_counts = [unit.usage_count for unit in network.units]
            if usage_counts:
                st.write("**Unit Usage Frequency:**")
                st.bar_chart(usage_counts)

            st.markdown("### Export All Insights to Excel")
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Raw Data')
                corr_matrix.to_excel(writer, sheet_name='Correlation')
                if 'mi_df' in locals():
                    mi_df.to_excel(writer, sheet_name='Mutual Info')
                cluster_df.to_excel(writer, sheet_name='Clusters')
                pca_df.to_excel(writer, sheet_name='PCA')
                outliers_df.to_excel(writer, sheet_name='Outliers')
            st.download_button("📥 Download Full Analysis", data=output.getvalue(), file_name="EDA_Insights.xlsx")

            st.markdown("### Interpretations")
            st.markdown("- **High Pattern Similarity** indicates recurring sequences or trends.")
            st.markdown("- **Strong MI Scores** imply nonlinear relationships.")
            st.markdown("- **Outliers** may reflect anomalies or impactful events.")
            st.markdown("- **Clustering** reveals data groupings and behaviors.")
            st.markdown("- **Usage Frequency** highlights important encoded patterns in the CNN.")
        else:
            st.info("No patterns stored in episodic memory yet.")
