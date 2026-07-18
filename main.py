
"""

Real Estate Buyer Segmentation - Advanced ML Pipeline

Complete workflow: EDA → Preprocessing → Clustering → Interpretation 
→ Predictive Modeling → Export
Author: Yunes Abdulghani Mohammed Ghaleb
Organization: Unified Mentor | Parcl Co. Limited
Date: 2026

"""

import os
import pickle
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not installed. Run: pip install xgboost")

# Import utilities
from utils import (
    create_directories, load_raw_data, clean_clients_data, clean_properties_data,
    merge_and_engineer_features, prepare_features_for_clustering,
    save_figure, save_processed_data, save_model, load_model,
    CLUSTER_COLORS, CLUSTER_NAMES
)

# Suppress warnings
warnings.filterwarnings('ignore')


# MAIN EXECUTION


def main():
    """
    Main execution pipeline for Real Estate Buyer Segmentation.
    """
    print("\n" + "=" * 80)
    print(" " * 15 + "🏢 REAL ESTATE BUYER SEGMENTATION PROJECT")
    print(" " * 5 + "Machine Learning based Buyer Segmentation and Investment Profiling")
    print(" " * 20 + "Unified Mentor | Parcl Co. Limited")
    print("=" * 80)
    
    # Create directories
    create_directories()
    
    
    # PHASE 1: DATA LOADING & CLEANING
    
    
    # Load raw data
    clients_raw, properties_raw = load_raw_data()
    
    # Clean data
    clients_clean = clean_clients_data(clients_raw)
    properties_clean = clean_properties_data(properties_raw)
    
    # Save cleaned clients data
    save_processed_data(clients_clean, 'data/processed_data.csv')
    
    
    # PHASE 2: FEATURE ENGINEERING
    
    
    features_df = merge_and_engineer_features(clients_clean, properties_clean)
    
    
    # PHASE 3: EXPLORATORY DATA ANALYSIS (EDA)

    run_eda(features_df)
    
    
    # PHASE 4: FEATURE PREPARATION
    
    
    scaled_features, feature_cols, scaler, encoders = prepare_features_for_clustering(features_df)
    
    # Save scaler
    save_model(scaler, 'models/scaler.pkl')
    
    
    # PHASE 5: OPTIMAL CLUSTER SELECTION
    
    
    optimal_k = find_optimal_clusters(scaled_features)
    
    
    # PHASE 6: CLUSTERING MODELS
    
    
    kmeans_labels, kmeans_model = run_kmeans(scaled_features, optimal_k)
    hierarchical_labels = run_hierarchical(scaled_features, optimal_k)
    
    # Save K-Means model
    save_model(kmeans_model, 'models/kmeans_model.pkl')
    
    
    # PHASE 7: MODEL EVALUATION & COMPARISON
    
    
    compare_models(scaled_features, kmeans_labels, hierarchical_labels)
    
    
    # PHASE 8: CLUSTER INTERPRETATION
    
    
    features_df = interpret_clusters(features_df, kmeans_labels, scaled_features)
    
    # Save clustered data
    features_df.to_csv('models/clustered_data.csv', index=False)
    print("\n✅ Saved clustered data to models/clustered_data.csv")
    
    
    # PHASE 9: ADVANCED VISUALIZATIONS
    
    
    create_advanced_visualizations(features_df, scaled_features, kmeans_labels, kmeans_model)
    
    
    # PHASE 11: PREDICTIVE MODELING
    
    
    run_predictive_modeling(features_df)
    
    
    # PHASE 10: FINAL SUMMARY
    
    
    print_final_summary(features_df)
    
    print("\n" + "=" * 80)
    print(" " * 25 + "🎉 PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\n📁 Generated Files:")
    print("   • data/processed_data.csv")
    print("   • models/kmeans_model.pkl")
    print("   • models/scaler.pkl")
    print("   • models/random_forest_model.pkl")
    print("   • models/xgboost_model.pkl")
    print("   • models/neural_network_model.pkl")
    print("   • models/nn_scaler.pkl")
    print("   • models/clustered_data.csv")
    print("   • reports/figures/eda_charts.png")
    print("   • reports/figures/elbow_plot.png")
    print("   • reports/figures/silhouette_plot.png")
    print("   • reports/figures/dendrogram.png")
    print("   • reports/figures/pca_visualization.png")
    print("   • reports/figures/cluster_profiles.png")
    print("   • reports/figures/model_comparison.png")
    print("   • reports/figures/confusion_matrices.png")
    print("   • reports/model_comparison.csv")
    print("\n🚀 Next Step: Run 'streamlit run app.py' to launch the dashboard")
    print("=" * 80)



# PHASE 3: EXPLORATORY DATA ANALYSIS


def run_eda(features_df: pd.DataFrame) -> None:
    """
    Comprehensive Exploratory Data Analysis with visualizations.
    """
    print("\n" + "=" * 70)
    print("PHASE 3: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Real Estate Buyers - Comprehensive EDA', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # Row 1: Demographics
    # 1. Age Distribution
    ax1 = plt.subplot(3, 4, 1)
    sns.histplot(data=features_df, x='age', bins=20, kde=True, color='skyblue', ax=ax1)
    ax1.set_title('Age Distribution', fontweight='bold')
    ax1.set_xlabel('Age (years)')
    
    # 2. Age Group Distribution
    ax2 = plt.subplot(3, 4, 2)
    age_group_counts = features_df['age_group'].value_counts()
    colors_age = plt.cm.Set3(np.linspace(0, 1, len(age_group_counts)))
    ax2.pie(age_group_counts, labels=age_group_counts.index, autopct='%1.1f%%',
            colors=colors_age, startangle=90)
    ax2.set_title('Age Groups', fontweight='bold')
    
    # 3. Gender Distribution
    ax3 = plt.subplot(3, 4, 3)
    gender_counts = features_df['gender'].value_counts()
    ax3.bar(gender_counts.index, gender_counts.values, color=['#FF6B6B', '#4ECDC4'])
    ax3.set_title('Gender Distribution', fontweight='bold')
    ax3.set_ylabel('Count')
    
    # 4. Client Type
    ax4 = plt.subplot(3, 4, 4)
    type_counts = features_df['client_type'].value_counts()
    ax4.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%',
            colors=['#45B7D1', '#96CEB4'], startangle=90)
    ax4.set_title('Client Type', fontweight='bold')
    
    # Row 2: Financial
    # 5. Average Sale Price
    ax5 = plt.subplot(3, 4, 5)
    sns.histplot(data=features_df, x='avg_sale_price', bins=25, kde=True, 
                 color='mediumpurple', ax=ax5)
    ax5.set_title('Average Sale Price Distribution', fontweight='bold')
    ax5.set_xlabel('Price ($)')
    
    # 6. Total Investment
    ax6 = plt.subplot(3, 4, 6)
    sns.histplot(data=features_df, x='total_investment', bins=25, kde=True,
                 color='gold', ax=ax6)
    ax6.set_title('Total Investment Distribution', fontweight='bold')
    ax6.set_xlabel('Investment ($)')
    
    # 7. Price per Sqft
    ax7 = plt.subplot(3, 4, 7)
    sns.histplot(data=features_df, x='avg_price_per_sqft', bins=20, kde=True,
                 color='lightcoral', ax=ax7)
    ax7.set_title('Price per Sqft', fontweight='bold')
    ax7.set_xlabel('$/sqft')
    
    # 8. Property Count
    ax8 = plt.subplot(3, 4, 8)
    prop_counts = features_df['property_count'].value_counts().sort_index()
    ax8.bar(prop_counts.index, prop_counts.values, color='lightgreen')
    ax8.set_title('Properties per Client', fontweight='bold')
    ax8.set_xlabel('Number of Properties')
    ax8.set_ylabel('Count')
    
    # Row 3: Behavioral
    # 9. Acquisition Purpose
    ax9 = plt.subplot(3, 4, 9)
    purpose_counts = features_df['acquisition_purpose'].value_counts()
    ax9.pie(purpose_counts, labels=purpose_counts.index, autopct='%1.1f%%',
            colors=['#FFEAA7', '#DDA0DD'], startangle=90)
    ax9.set_title('Acquisition Purpose', fontweight='bold')
    
    # 10. Loan Application
    ax10 = plt.subplot(3, 4, 10)
    loan_counts = features_df['loan_applied'].value_counts()
    ax10.bar(loan_counts.index, loan_counts.values, color=['#FF6B6B', '#4ECDC4'])
    ax10.set_title('Loan Application', fontweight='bold')
    ax10.set_ylabel('Count')
    
    # 11. Satisfaction Score
    ax11 = plt.subplot(3, 4, 11)
    sns.histplot(data=features_df, x='satisfaction_score', bins=5, discrete=True,
                 color='orange', ax=ax11)
    ax11.set_title('Satisfaction Score', fontweight='bold')
    ax11.set_xlabel('Score (1-5)')
    
    # 12. Referral Channel
    ax12 = plt.subplot(3, 4, 12)
    ref_counts = features_df['referral_channel'].value_counts().head(5)
    ax12.barh(ref_counts.index, ref_counts.values, color='teal')
    ax12.set_title('Top Referral Channels', fontweight='bold')
    ax12.set_xlabel('Count')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_figure(fig, 'reports/figures/eda_charts.png', dpi=300)
    plt.close()
    
    print("✅ EDA charts saved to reports/figures/eda_charts.png")
    
    # Statistical summary
    print("\n📊 Statistical Summary:")
    print(features_df[['age', 'avg_sale_price', 'total_investment', 
                       'property_count', 'satisfaction_score']].describe().round(2))



# PHASE 5: OPTIMAL CLUSTER SELECTION


def find_optimal_clusters(scaled_features: np.ndarray, max_k: int = 10) -> int:
    """
    Find optimal number of clusters using multiple evaluation metrics.
    """
    print("\n" + "=" * 70)
    print("PHASE 5: OPTIMAL CLUSTER SELECTION")
    print("=" * 70)
    
    K_range = range(2, max_k + 1)
    
    metrics = {
        'inertias': [],
        'silhouette_scores': [],
        'davies_bouldin_scores': [],
        'calinski_harabasz_scores': []
    }
    
    print("\n  → Evaluating K from 2 to", max_k)
    
    for k in K_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels_temp = kmeans_temp.fit_predict(scaled_features)
        
        metrics['inertias'].append(kmeans_temp.inertia_)
        metrics['silhouette_scores'].append(silhouette_score(scaled_features, labels_temp))
        metrics['davies_bouldin_scores'].append(davies_bouldin_score(scaled_features, labels_temp))
        metrics['calinski_harabasz_scores'].append(calinski_harabasz_score(scaled_features, labels_temp))
        
        print(f"     K={k}: Silhouette={metrics['silhouette_scores'][-1]:.3f}, "
              f"DB={metrics['davies_bouldin_scores'][-1]:.3f}")
    
    # Plot all metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cluster Evaluation Metrics', fontsize=16, fontweight='bold')
    
    # Elbow Method
    axes[0, 0].plot(K_range, metrics['inertias'], 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Number of Clusters (K)')
    axes[0, 0].set_ylabel('Inertia (WCSS)')
    axes[0, 0].set_title('Elbow Method', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Silhouette Score
    axes[0, 1].plot(K_range, metrics['silhouette_scores'], 'ro-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Number of Clusters (K)')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].set_title('Silhouette Score (Higher is Better)', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Davies-Bouldin
    axes[1, 0].plot(K_range, metrics['davies_bouldin_scores'], 'go-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Number of Clusters (K)')
    axes[1, 0].set_ylabel('Davies-Bouldin Index')
    axes[1, 0].set_title('Davies-Bouldin Index (Lower is Better)', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Calinski-Harabasz
    axes[1, 1].plot(K_range, metrics['calinski_harabasz_scores'], 'mo-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Number of Clusters (K)')
    axes[1, 1].set_ylabel('Calinski-Harabasz Score')
    axes[1, 1].set_title('Calinski-Harabasz (Higher is Better)', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, 'reports/figures/elbow_plot.png', dpi=300)
    plt.close()
    
    # Find optimal K
    best_k_silhouette = list(K_range)[np.argmax(metrics['silhouette_scores'])]
    best_k_db = list(K_range)[np.argmin(metrics['davies_bouldin_scores'])]
    best_k_ch = list(K_range)[np.argmax(metrics['calinski_harabasz_scores'])]
    
    print(f"\n📊 Optimal K Results:")
    print(f"   • By Silhouette Score: K={best_k_silhouette} (Score: {max(metrics['silhouette_scores']):.3f})")
    print(f"   • By Davies-Bouldin: K={best_k_db} (Score: {min(metrics['davies_bouldin_scores']):.3f})")
    print(f"   • By Calinski-Harabasz: K={best_k_ch} (Score: {max(metrics['calinski_harabasz_scores']):.1f})")
    
    # Use K=4 as specified in project requirements
    optimal_k = 4
    print(f"\n✅ Selected K={optimal_k} clusters (as per project requirements)")
    
    return optimal_k



# PHASE 6: CLUSTERING MODELS


def run_kmeans(scaled_features: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, KMeans]:
    """
    Run K-Means clustering algorithm.
    """
    print("\n" + "=" * 70)
    print("PHASE 6A: K-MEANS CLUSTERING")
    print("=" * 70)
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
        max_iter=300,
        algorithm='lloyd'
    )
    
    labels = kmeans.fit_predict(scaled_features)
    
    print(f"\n📊 K-Means Results:")
    print(f"   • Number of iterations: {kmeans.n_iter_}")
    print(f"   • Inertia (WCSS): {kmeans.inertia_:.2f}")
    
    # Cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n   Cluster Distribution:")
    for u, c in zip(unique, counts):
        print(f"      Cluster {u}: {c} clients ({c/len(labels)*100:.1f}%)")
    
    return labels, kmeans


def run_hierarchical(scaled_features: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Run Hierarchical clustering algorithm.
    """
    print("\n" + "=" * 70)
    print("PHASE 6B: HIERARCHICAL CLUSTERING")
    print("=" * 70)
    
    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward',
        metric='euclidean'
    )
    
    labels = hierarchical.fit_predict(scaled_features)
    
    print(f"\n📊 Hierarchical Clustering Results:")
    unique, counts = np.unique(labels, return_counts=True)
    print(f"   Cluster Distribution:")
    for u, c in zip(unique, counts):
        print(f"      Cluster {u}: {c} clients ({c/len(labels)*100:.1f}%)")
    
    # Create dendrogram
    print("\n  → Generating dendrogram...")
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Sample data for dendrogram (too many points = slow)
    sample_size = min(200, len(scaled_features))
    sample_idx = np.random.choice(len(scaled_features), sample_size, replace=False)
    sample_data = scaled_features[sample_idx]
    
    linkage_matrix = linkage(sample_data, method='ward')
    dendrogram(
        linkage_matrix,
        truncate_mode='lastp',
        p=n_clusters,
        leaf_rotation=90,
        leaf_font_size=12,
        ax=ax,
        color_threshold=0.7 * max(linkage_matrix[:, 2])
    )
    
    ax.set_title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')
    ax.set_xlabel('Cluster Index', fontsize=12)
    ax.set_ylabel('Ward Distance', fontsize=12)
    
    save_figure(fig, 'reports/figures/dendrogram.png', dpi=300)
    plt.close()
    
    print("✅ Dendrogram saved to reports/figures/dendrogram.png")
    
    return labels



# PHASE 7: MODEL EVALUATION


def compare_models(scaled_features: np.ndarray, 
                   kmeans_labels: np.ndarray,
                   hierarchical_labels: np.ndarray) -> None:
    """
    Compare K-Means and Hierarchical clustering performance.
    """
    print("\n" + "=" * 70)
    print("PHASE 7: MODEL EVALUATION & COMPARISON")
    print("=" * 70)
    
    # Calculate metrics for both models
    metrics_comparison = {
        'K-Means': {
            'Silhouette Score': silhouette_score(scaled_features, kmeans_labels),
            'Davies-Bouldin Index': davies_bouldin_score(scaled_features, kmeans_labels),
            'Calinski-Harabasz Score': calinski_harabasz_score(scaled_features, kmeans_labels)
        },
        'Hierarchical': {
            'Silhouette Score': silhouette_score(scaled_features, hierarchical_labels),
            'Davies-Bouldin Index': davies_bouldin_score(scaled_features, hierarchical_labels),
            'Calinski-Harabasz Score': calinski_harabasz_score(scaled_features, hierarchical_labels)
        }
    }
    
    print("\n📊 Model Comparison:")
    print("-" * 60)
    print(f"{'Metric':<25} {'K-Means':>15} {'Hierarchical':>15}")
    print("-" * 60)
    
    for metric in ['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Score']:
        km_val = metrics_comparison['K-Means'][metric]
        h_val = metrics_comparison['Hierarchical'][metric]
        print(f"{metric:<25} {km_val:>15.3f} {h_val:>15.3f}")
    
    print("-" * 60)
    print("💡 Interpretation:")
    print("   • Silhouette: Higher is better (range: -1 to 1)")
    print("   • Davies-Bouldin: Lower is better")
    print("   • Calinski-Harabasz: Higher is better")
    
    # Agreement between models
    agreement = np.mean(kmeans_labels == hierarchical_labels) * 100
    print(f"\n   Model Agreement: {agreement:.1f}% labels match between K-Means and Hierarchical")



# PHASE 8: CLUSTER INTERPRETATION


def interpret_clusters(features_df: pd.DataFrame, 
                       labels: np.ndarray,
                       scaled_features: np.ndarray) -> pd.DataFrame:
    """
    Deep interpretation of each cluster with business insights.
    """
    print("\n" + "=" * 70)
    print("PHASE 8: CLUSTER INTERPRETATION")
    print("=" * 70)
    
    df = features_df.copy()
    df['cluster'] = labels
    df['cluster_name'] = df['cluster'].map(CLUSTER_NAMES)
    
    print("\n" + "=" * 70)
    print("DETAILED CLUSTER ANALYSIS")
    print("=" * 70)
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        name = CLUSTER_NAMES.get(cluster_id, f'Cluster {cluster_id}')
        
        print(f"\n{'🔷' * 25}")
        print(f"  CLUSTER {cluster_id}: {name.upper()}")
        print(f"{'🔷' * 25}")
        
        # Basic stats
        print(f"\n  📊 Demographics:")
        print(f"     • Total Clients: {len(cluster_data):,} ({len(cluster_data)/len(df)*100:.1f}%)")
        print(f"     • Avg Age: {cluster_data['age'].mean():.1f} years (±{cluster_data['age'].std():.1f})")
        print(f"     • Gender: {cluster_data['gender'].value_counts().to_dict()}")
        print(f"     • Age Group: {cluster_data['age_group'].value_counts().to_dict()}")
        
        # Financial stats
        print(f"\n  💰 Financial Profile:")
        print(f"     • Avg Sale Price: ${cluster_data['avg_sale_price'].mean():,.2f}")
        print(f"     • Total Investment: ${cluster_data['total_investment'].mean():,.2f}")
        print(f"     • Avg Price/sqft: ${cluster_data['avg_price_per_sqft'].mean():.2f}")
        print(f"     • Price Range: ${cluster_data['price_range'].mean():,.2f}")
        
        # Behavioral stats
        print(f"\n  🏠 Behavioral Patterns:")
        print(f"     • Avg Properties: {cluster_data['property_count'].mean():.1f}")
        print(f"     • Property Diversity: {cluster_data['property_diversity'].mean():.2f}")
        print(f"     • Investment Span: {cluster_data['investment_span_years'].mean():.1f} years")
        print(f"     • Investment/Year: ${cluster_data['investment_per_year'].mean():,.2f}")
        
        # Business stats
        print(f"\n  🏢 Business Insights:")
        print(f"     • Client Type: {cluster_data['client_type'].value_counts().to_dict()}")
        print(f"     • Acquisition Purpose: {cluster_data['acquisition_purpose'].value_counts().to_dict()}")
        print(f"     • Loan Applied: {cluster_data['loan_applied'].value_counts().to_dict()}")
        print(f"     • Avg Satisfaction: {cluster_data['satisfaction_score'].mean():.2f}/5")
        print(f"     • Top Regions: {cluster_data['region'].value_counts().head(3).to_dict()}")
        print(f"     • Top Referral: {cluster_data['referral_channel'].value_counts().head(2).to_dict()}")
    
    return df



# PHASE 9: ADVANCED VISUALIZATIONS


def create_advanced_visualizations(features_df: pd.DataFrame,
                                    scaled_features: np.ndarray,
                                    labels: np.ndarray,
                                    kmeans_model: KMeans) -> None:
    """
    Create advanced visualizations for cluster analysis.
    """
    print("\n" + "=" * 70)
    print("PHASE 9: ADVANCED VISUALIZATIONS")
    print("=" * 70)
    
    df = features_df.copy()
    
    # --- PCA Visualization ---
    print("\n  → Creating PCA visualization...")
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_features)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    for cluster_id in sorted(df['cluster'].unique()):
        mask = labels == cluster_id
        name = CLUSTER_NAMES.get(cluster_id, f'Cluster {cluster_id}')
        color = CLUSTER_COLORS.get(cluster_id, '#333333')
        
        ax.scatter(
            pca_features[mask, 0],
            pca_features[mask, 1],
            c=color,
            label=f'{name} (C{cluster_id})',
            alpha=0.7,
            s=100,
            edgecolors='black',
            linewidth=0.5
        )
    
    # Plot centroids
    centroids_pca = pca.transform(kmeans_model.cluster_centers_)
    ax.scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        c='black',
        marker='X',
        s=400,
        label='Centroids',
        edgecolors='white',
        linewidth=2,
        zorder=5
    )
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    ax.set_title('Buyer Segments - PCA Visualization', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    save_figure(fig, 'reports/figures/pca_visualization.png', dpi=300)
    plt.close()
    
    print(f"     PCA Explained Variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    print("✅ PCA visualization saved")
    
    # --- Cluster Profiles Radar Chart ---
    print("\n  → Creating cluster profile charts...")
    
    # Normalize features for radar chart
    profile_features = ['age', 'avg_sale_price', 'total_investment', 
                        'property_count', 'satisfaction_score', 'avg_price_per_sqft']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), subplot_kw=dict(projection='polar'))
    fig.suptitle('Cluster Profiles - Radar Charts', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, cluster_id in enumerate(sorted(df['cluster'].unique())):
        ax = axes[idx]
        name = CLUSTER_NAMES.get(cluster_id, f'Cluster {cluster_id}')
        
        cluster_data = df[df['cluster'] == cluster_id]
        
        # Calculate normalized values (0-1 scale)
        values = []
        for feat in profile_features:
            feat_min = df[feat].min()
            feat_max = df[feat].max()
            feat_range = feat_max - feat_min if feat_max != feat_min else 1
            val = (cluster_data[feat].mean() - feat_min) / feat_range
            values.append(val)
        
        values += values[:1]  # Close the polygon
        
        angles = np.linspace(0, 2 * np.pi, len(profile_features), endpoint=False).tolist()
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color=CLUSTER_COLORS.get(cluster_id, '#333'))
        ax.fill(angles, values, alpha=0.25, color=CLUSTER_COLORS.get(cluster_id, '#333'))
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f.replace('_', '\n') for f in profile_features], fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(f'{name} (C{cluster_id})', fontsize=12, fontweight='bold', pad=20)
        ax.grid(True)
    
    plt.tight_layout()
    save_figure(fig, 'reports/figures/cluster_profiles.png', dpi=300)
    plt.close()
    print("✅ Cluster profiles saved")
    
    # --- Feature Importance per Cluster ---
    print("\n  → Creating feature importance heatmap...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    cluster_means = df.groupby('cluster')[profile_features].mean()
    cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
    
    sns.heatmap(
        cluster_means_norm.T,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=0.5,
        ax=ax,
        cbar_kws={'label': 'Normalized Value (0-1)'}
    )
    
    ax.set_title('Cluster Feature Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    
    # Rename x-tick labels
    ax.set_xticklabels([f'C{i}\n{CLUSTER_NAMES.get(i, "")}' for i in cluster_means.index], rotation=0)
    
    plt.tight_layout()
    save_figure(fig, 'reports/figures/feature_heatmap.png', dpi=300)
    plt.close()
    print("✅ Feature heatmap saved")



# PHASE 11: PREDICTIVE MODELING


def run_predictive_modeling(features_df: pd.DataFrame) -> None:
    """
    Advanced Predictive Modeling for Buyer Segment Classification.
    Models: Random Forest, XGBoost (if available), Neural Network
    """
    print("\n" + "=" * 70)
    print("PHASE 11: ADVANCED PREDICTIVE MODELING")
    print("=" * 70)
    
    # Prepare data
    df = features_df.copy()
    
    # Select features for prediction (exclude target and non-numeric)
    exclude_cols = ['client_id', 'cluster', 'cluster_name', 'age_group', 'gender', 
                    'client_type', 'country', 'region', 'acquisition_purpose', 
                    'loan_applied', 'referral_channel']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
    
    X = df[feature_cols].fillna(0)
    y = df['cluster']
    
    print(f"\n📊 Dataset for Prediction:")
    print(f"   • Features: {len(feature_cols)}")
    print(f"   • Samples: {len(X)}")
    print(f"   • Classes: {len(y.unique())}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n   • Training: {len(X_train)} samples")
    print(f"   • Testing: {len(X_test)} samples")
    

    # Model 1: Random Forest
    
    print("\n" + "-" * 50)
    print("MODEL 1: RANDOM FOREST CLASSIFIER")
    print("-" * 50)
    
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='accuracy')
    
    print(f"\n📊 Random Forest CV Accuracy: {rf_cv_scores.mean():.4f} (±{rf_cv_scores.std():.4f})")
    
    # Train and predict
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred, average='weighted')
    
    print(f"📊 Test Accuracy: {rf_accuracy:.4f}")
    print(f"📊 Test F1-Score: {rf_f1:.4f}")
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔝 Top 10 Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   • {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    save_model(rf_model, 'models/random_forest_model.pkl')
    
    
    # Model 2: XGBoost (if available)
    
    xgb_accuracy = 0
    xgb_f1 = 0
    xgb_cv_scores = np.array([0])
    xgb_pred = rf_pred  # fallback
    
    if XGBOOST_AVAILABLE:
        print("\n" + "-" * 50)
        print("MODEL 2: XGBOOST CLASSIFIER")
        print("-" * 50)
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        # Cross-validation
        xgb_cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='accuracy')
        
        print(f"\n📊 XGBoost CV Accuracy: {xgb_cv_scores.mean():.4f} (±{xgb_cv_scores.std():.4f})")
        
        # Train and predict
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        xgb_f1 = f1_score(y_test, xgb_pred, average='weighted')
        
        print(f"📊 Test Accuracy: {xgb_accuracy:.4f}")
        print(f"📊 Test F1-Score: {xgb_f1:.4f}")
        
        # Save model
        save_model(xgb_model, 'models/xgboost_model.pkl')
    else:
        print("\n⚠️ XGBoost skipped (not installed)")
    
    
    # Model 3: Neural Network (MLP)

    print("\n" + "-" * 50)
    print("MODEL 3: NEURAL NETWORK (MLP)")
    print("-" * 50)
    
    # Scale features for neural network
    scaler_nn = StandardScaler()
    X_train_scaled = scaler_nn.fit_transform(X_train)
    X_test_scaled = scaler_nn.transform(X_test)
    
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=32,
        learning_rate='adaptive',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20
    )
    
    # Cross-validation
    mlp_cv_scores = cross_val_score(mlp_model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    
    print(f"\n📊 Neural Network CV Accuracy: {mlp_cv_scores.mean():.4f} (±{mlp_cv_scores.std():.4f})")
    
    # Train and predict
    mlp_model.fit(X_train_scaled, y_train)
    mlp_pred = mlp_model.predict(X_test_scaled)
    mlp_accuracy = accuracy_score(y_test, mlp_pred)
    mlp_f1 = f1_score(y_test, mlp_pred, average='weighted')
    
    print(f"📊 Test Accuracy: {mlp_accuracy:.4f}")
    print(f"📊 Test F1-Score: {mlp_f1:.4f}")
    
    # Save model and scaler
    save_model(mlp_model, 'models/neural_network_model.pkl')
    save_model(scaler_nn, 'models/nn_scaler.pkl')
    
    
    # Model Comparison
    
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    
    if XGBOOST_AVAILABLE:
        comparison = pd.DataFrame({
            'Model': ['Random Forest', 'XGBoost', 'Neural Network'],
            'CV Accuracy': [rf_cv_scores.mean(), xgb_cv_scores.mean(), mlp_cv_scores.mean()],
            'CV Std': [rf_cv_scores.std(), xgb_cv_scores.std(), mlp_cv_scores.std()],
            'Test Accuracy': [rf_accuracy, xgb_accuracy, mlp_accuracy],
            'Test F1-Score': [rf_f1, xgb_f1, mlp_f1]
        })
    else:
        comparison = pd.DataFrame({
            'Model': ['Random Forest', 'Neural Network'],
            'CV Accuracy': [rf_cv_scores.mean(), mlp_cv_scores.mean()],
            'CV Std': [rf_cv_scores.std(), mlp_cv_scores.std()],
            'Test Accuracy': [rf_accuracy, mlp_accuracy],
            'Test F1-Score': [rf_f1, mlp_f1]
        })
    
    print("\n" + comparison.to_string(index=False))
    
    # Best model
    best_idx = comparison['Test Accuracy'].idxmax()
    best_model_name = comparison.loc[best_idx, 'Model']
    best_accuracy = comparison.loc[best_idx, 'Test Accuracy']
    
    print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
    
    
    # Visualization: Model Comparison
    
    print("\n  → Creating model comparison visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy comparison
    models = comparison['Model']
    x_pos = np.arange(len(models))
    
    axes[0].bar(x_pos - 0.2, comparison['CV Accuracy'], 0.4, 
                label='CV Accuracy', color='skyblue', alpha=0.8)
    axes[0].bar(x_pos + 0.2, comparison['Test Accuracy'], 0.4, 
                label='Test Accuracy', color='lightcoral', alpha=0.8)
    
    for i, (cv, test) in enumerate(zip(comparison['CV Accuracy'], comparison['Test Accuracy'])):
        axes[0].text(i - 0.2, cv + 0.01, f'{cv:.3f}', ha='center', fontsize=10)
        axes[0].text(i + 0.2, test + 0.01, f'{test:.3f}', ha='center', fontsize=10)
    
    axes[0].set_xlabel('Model', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(models, rotation=15)
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3)
    
    # Feature Importance (Random Forest)
    top_features = feature_importance.head(15)
    axes[1].barh(range(len(top_features)), top_features['importance'], color='mediumseagreen')
    axes[1].set_yticks(range(len(top_features)))
    axes[1].set_yticklabels(top_features['feature'], fontsize=9)
    axes[1].set_xlabel('Importance', fontsize=12)
    axes[1].set_title('Top 15 Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    save_figure(fig, 'reports/figures/model_comparison.png', dpi=300)
    plt.close()
    
    print("✅ Model comparison saved to reports/figures/model_comparison.png")
    
    
    # Confusion Matrix for All Models
    
    print("\n  → Creating confusion matrices...")
    
    if XGBOOST_AVAILABLE:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        models_data = [
            (rf_pred, 'Random Forest', rf_accuracy),
            (xgb_pred, 'XGBoost', xgb_accuracy),
            (mlp_pred, 'Neural Network', mlp_accuracy)
        ]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        models_data = [
            (rf_pred, 'Random Forest', rf_accuracy),
            (mlp_pred, 'Neural Network', mlp_accuracy)
        ]
    
    for idx, (pred, name, acc) in enumerate(models_data):
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'{name}\nAccuracy: {acc:.3f}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    save_figure(fig, 'reports/figures/confusion_matrices.png', dpi=300)
    plt.close()
    
    print("✅ Confusion matrices saved to reports/figures/confusion_matrices.png")
    
    
    # Classification Report for Best Model
    
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT - BEST MODEL")
    print("=" * 70)
    
    if best_model_name == 'Random Forest':
        best_pred = rf_pred
    elif best_model_name == 'XGBoost':
        best_pred = xgb_pred
    else:
        best_pred = mlp_pred
    
    target_names = [CLUSTER_NAMES[i] for i in sorted(CLUSTER_NAMES.keys())]
    print(f"\n{best_model_name}:")
    print(classification_report(y_test, best_pred, target_names=target_names))
    
    # Save comparison to CSV
    comparison.to_csv('reports/model_comparison.csv', index=False)
    print("\n✅ Model comparison saved to reports/model_comparison.csv")



# PHASE 10: FINAL SUMMARY


def print_final_summary(features_df: pd.DataFrame) -> None:
    """
    Print final project summary.
    """
    print("\n" + "=" * 70)
    print("PHASE 10: PROJECT SUMMARY")
    print("=" * 70)
    
    total_clients = len(features_df)
    
    print(f"\n📊 Dataset Summary:")
    print(f"   • Total Clients Analyzed: {total_clients:,}")
    print(f"   • Total Properties: {features_df['property_count'].sum():,.0f}")
    print(f"   • Total Investment Volume: ${features_df['total_investment'].sum():,.2f}")
    print(f"   • Average Investment per Client: ${features_df['total_investment'].mean():,.2f}")
    
    print(f"\n🏷️ Segment Distribution:")
    for cluster_id in sorted(features_df['cluster'].unique()):
        count = (features_df['cluster'] == cluster_id).sum()
        name = CLUSTER_NAMES.get(cluster_id, f'Cluster {cluster_id}')
        pct = count / total_clients * 100
        print(f"   • {name}: {count:,} clients ({pct:.1f}%)")
    
    print(f"\n💡 Key Business Insights:")
    
    # Find highest spending segment
    highest_spending = features_df.groupby('cluster_name')['total_investment'].mean().idxmax()
    print(f"   • Highest Spending Segment: {highest_spending}")
    
    # Find most satisfied segment
    most_satisfied = features_df.groupby('cluster_name')['satisfaction_score'].mean().idxmax()
    print(f"   • Most Satisfied Segment: {most_satisfied}")
    
    # Find largest segment
    largest = features_df['cluster_name'].value_counts().idxmax()
    print(f"   • Largest Segment: {largest}")
    
    # Investment vs Personal use
    investment_pct = (features_df['acquisition_purpose'] == 'Investment').mean() * 100
    print(f"   • Investment Purpose: {investment_pct:.1f}% of clients")
    
    # Loan dependency
    loan_pct = (features_df['loan_applied'] == 'Yes').mean() * 100
    print(f"   • Loan Dependency: {loan_pct:.1f}% of clients")
    
    print(f"\n🎯 Recommendations:")
    print(f"   1. Target {CLUSTER_NAMES[3]} with premium property offerings")
    print(f"   2. Develop first-time buyer programs for {CLUSTER_NAMES[1]}")
    print(f"   3. Create corporate partnership packages for {CLUSTER_NAMES[2]}")
    print(f"   4. Expand global marketing for {CLUSTER_NAMES[0]}")



# ENTRY POINT


if __name__ == "__main__":
    main()