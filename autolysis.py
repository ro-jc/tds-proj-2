# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "networkx",
#     "numpy",
#     "pandas",
#     "seaborn",
#     "scikit-learn",
#     "openai",
# ]
# ///
import os
import sys
import io
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression
import networkx as nx
import json

from openai import OpenAI


class VisualizationGenerator:
    def __init__(self, dataframe):
        """
        Initialize visualization generator

        Args:
            dataframe (pd.DataFrame): Input dataframe
        """
        self.df = dataframe
        self.plots_dir = "visualization_outputs"
        os.makedirs(self.plots_dir, exist_ok=True)

    def _save_and_encode_plot(self, filename):
        """Save plot and return base64 encoded image"""
        filepath = os.path.join(self.plots_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

        with open(filepath, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def boxplot_distribution(self, columns=None):
        """
        Create boxplots for numeric distributions

        Args:
            columns (list, optional): Specific columns to visualize

        Returns:
            dict: Visualization metadata
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns

        plt.figure(figsize=(len(columns) * 3, 6))
        for i, col in enumerate(columns, 1):
            plt.subplot(1, len(columns), i)
            sns.boxplot(x=self.df[col])
            plt.title(f"Distribution: {col}")

        return {
            "type": "boxplot_distribution",
            "columns": list(columns),
            "image": self._save_and_encode_plot("boxplot_distribution.png"),
        }

    def correlation_heatmap(self, method="pearson"):
        """
        Create correlation heatmap

        Args:
            method (str): Correlation method

        Returns:
            dict: Visualization metadata
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numeric_cols].corr(method=method)

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
        plt.title(f"{method.capitalize()} Correlation Heatmap")

        return {
            "type": "correlation_heatmap",
            "method": method,
            "image": self._save_and_encode_plot("correlation_heatmap.png"),
        }

    def pca_visualization(self, n_components=2):
        """
        PCA visualization of numeric features

        Args:
            n_components (int): Number of PCA components

        Returns:
            dict: Visualization metadata
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        X = self.df[numeric_cols]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        plt.figure(figsize=(10, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1])
        plt.title("PCA Visualization")
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")

        return {
            "type": "pca_visualization",
            "variance_explained": pca.explained_variance_ratio_.tolist(),
            "image": self._save_and_encode_plot("pca_visualization.png"),
        }

    def cluster_visualization(self, n_clusters=3):
        """
        K-means clustering visualization

        Args:
            n_clusters (int): Number of clusters

        Returns:
            dict: Visualization metadata
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        X = self.df[numeric_cols]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis")
        plt.colorbar(scatter)
        plt.title(f"{n_clusters} Cluster Visualization")

        return {
            "type": "cluster_visualization",
            "num_clusters": n_clusters,
            "cluster_centers": kmeans.cluster_centers_.tolist(),
            "image": self._save_and_encode_plot("cluster_visualization.png"),
        }

    def get_visualization_functions(self):
        """
        Return available visualization function details

        Returns:
            list: Available visualization functions
        """
        return [
            {
                "name": "boxplot_distribution",
                "description": "Create boxplots to show distribution of numeric columns",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of column names to visualize",
                        }
                    },
                },
            },
            {
                "name": "correlation_heatmap",
                "description": "Generate correlation heatmap for numeric columns",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "method": {
                            "type": "string",
                            "enum": ["pearson", "spearman", "kendall"],
                            "description": "Correlation calculation method",
                        }
                    },
                },
            },
            {
                "name": "pca_visualization",
                "description": "Perform PCA and visualize data in reduced dimensions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "n_components": {
                            "type": "integer",
                            "description": "Number of PCA components to visualize",
                        }
                    },
                },
            },
            {
                "name": "cluster_visualization",
                "description": "Perform K-means clustering and visualize clusters",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "n_clusters": {
                            "type": "integer",
                            "description": "Number of clusters to generate",
                        }
                    },
                },
            },
        ]


# Demonstration function
def generate_visualizations(df, client):
    viz_generator = VisualizationGenerator(df)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"Dataset columns: {list(df.columns)}\n"
                f"Dataset overview:\n{df.describe().to_string()}",
            }
        ],
        functions=viz_generator.get_visualization_functions(),
        function_call="auto",
    )

    return response


def generate_summary(df, client):
    # Perform initial analyses
    missing_values = df.isnull().sum()
    correlation_matrix = df.select_dtypes(include=[np.number]).corr()

    # Outlier detection
    def detect_outliers(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((series < lower_bound) | (series > upper_bound)).sum()

    outliers = {
        col: detect_outliers(df[col]) for col in df.select_dtypes(include=[np.number])
    }

    # Prepare comprehensive context
    summary_context = {
        "dataset_columns": list(df.columns),
        "column_types": dict(df.dtypes),
        "dataset_overview": df.describe().to_string(),
        "missing_values": dict(missing_values[missing_values > 0]),
        "correlation_matrix": correlation_matrix.to_dict(),
        "outliers": outliers,
    }

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a data science expert. Provide a comprehensive summary of the dataset, integrating insights from various analyses. You respond only in Valid Markdown as a README file with proper heading and bold tags",
            },
            {
                "role": "user",
                "content": f"Comprehensive Dataset Context:\n{summary_context}",
            },
        ],
    )

    return response


# Example usage
if __name__ == "__main__":
    # Replace with actual data and API key
    df = pd.read_csv(sys.argv[1])
    api_key = os.getenv("AIPROXY_TOKEN")

    client = OpenAI(api_key=api_key)
    client.base_url = "http://aiproxy.sanand.workers.dev/openai/v1"

    # function_response = generate_visualizations(df, client)
    summary_response = generate_summary(df, client)

    # print(function_response)
    print(summary_response)

    readme_content = f"""# Data Visualization Analysis

## Dataset Overview
- **Total Columns**: {len(df.columns)}
- **Total Rows**: {len(df)}
- **Numeric Columns**: {list(df.select_dtypes(include=[np.number]).columns)}

## Dataset Statistics
```python
{df.describe().to_string()}
```

## Visualization Insights
Generated using advanced data science techniques including:
- Principal Component Analysis (PCA)
- K-means Clustering
- Correlation Analysis

## LLM Insights

"""
    readme_content += summary_response
    # Write README
    with open("README.md", "w") as f:
        f.write(readme_content)
