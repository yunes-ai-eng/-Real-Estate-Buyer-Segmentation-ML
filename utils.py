"""

Utility Functions for Real Estate Buyer Segmentation Project

Complete utilities: Data Loading, Cleaning, Feature Engineering, 
Encoding, Scaling, and Model Management
Author: Yunes Abdulghani Mohammed Ghaleb
Organization: Unified Mentor | Parcl Co. Limited
Date: 2026

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import pickle
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# CLUSTER CONFIGURATION

CLUSTER_COLORS = {
    0: '#FF6B6B',  # Global Investors
    1: '#4ECDC4',  # First-Time Buyers
    2: '#45B7D1',  # Corporate Buyers
    3: '#96CEB4'   # Luxury Investors
}

CLUSTER_NAMES = {
    0: 'Global Investors',
    1: 'First-Time Buyers',
    2: 'Corporate Buyers',
    3: 'Luxury Investors'
}


# PRINT HELPERS

def print_section(title, width=70):
    """Print a formatted section header"""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)

def print_subsection(title, width=50):
    """Print a formatted subsection header"""
    print("\n" + "-" * width)
    print(f"  {title}")
    print("-" * width)


# DIRECTORY CREATION

def create_directories():
    """Create project directories if they don't exist"""
    dirs = ['models', 'reports/figures', 'screenshots', 'data']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✅ All directories created successfully")


# DATA LOADING

def load_data(clients_path='data/clients.csv', properties_path='data/properties.csv'):
    """Load and merge clients and properties datasets"""
    try:
        clients = pd.read_csv(clients_path)
        properties = pd.read_csv(properties_path)
        print(f"✅ Loaded {len(clients)} client records")
        print(f"✅ Loaded {len(properties)} property records")
        return clients, properties
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        raise

def load_raw_data(clients_path='data/clients.csv', properties_path='data/properties.csv'):
    """Alias for load_data - used by main.py"""
    return load_data(clients_path, properties_path)


# DATE PARSING

def parse_dates(date_series):
    """Parse dates with multiple formats"""
    formats = ['%m-%d-%Y', '%m/%d/%Y', '%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d']
    parsed = pd.to_datetime(date_series, errors='coerce')
    
    mask = parsed.isna()
    if mask.any():
        for fmt in formats:
            try:
                temp = pd.to_datetime(date_series[mask], format=fmt, errors='coerce')
                parsed.loc[mask] = temp
                mask = parsed.isna()
                if not mask.any():
                    break
            except:
                continue
    
    return parsed


# CLIENTS CLEANING

def clean_clients_data(clients_df):
    """Clean clients dataset"""
    print_section("STEP 2: CLEANING CLIENTS DATA")
    
    df = clients_df.copy()
    original_count = len(df)
    print(f"📊 Original records: {original_count:,}")
    
    print("  → Parsing dates...")
    df['date_of_birth'] = parse_dates(df['date_of_birth'])
    
    current_year = pd.Timestamp.now().year
    df['age'] = current_year - df['date_of_birth'].dt.year
    
    print("  → Standardizing categorical variables...")
    df['client_type'] = df['client_type'].str.strip().str.title()
    df['gender'] = df['gender'].str.strip().str.upper()
    df['country'] = df['country'].str.strip().str.title()
    df['region'] = df['region'].str.strip().str.title()
    df['acquisition_purpose'] = df['acquisition_purpose'].str.strip().str.title()
    df['loan_applied'] = df['loan_applied'].str.strip().str.title()
    df['referral_channel'] = df['referral_channel'].str.strip().str.title()
    
    print("  → Removing duplicates...")
    df = df.drop_duplicates(subset='client_id', keep='first')
    
    print("  → Handling missing values...")
    df = df.dropna(subset=['client_id', 'client_type', 'gender', 'age'])
    
    print(f"✅ Cleaned records: {len(df):,} (retention: {len(df)/original_count*100:.1f}%)")
    return df


# PROPERTIES CLEANING

def clean_properties_data(properties_df):
    """Clean properties dataset"""
    print_section("STEP 3: CLEANING PROPERTIES DATA")
    
    df = properties_df.copy()
    original_count = len(df)
    print(f"📊 Original records: {original_count:,}")
    
    print("  → Cleaning price data...")
    df['sale_price'] = df['sale_price'].astype(str).str.replace('[$,]', '', regex=True)
    df['sale_price'] = pd.to_numeric(df['sale_price'], errors='coerce')
    
    print("  → Cleaning numeric columns...")
    df['floor_area_sqft'] = pd.to_numeric(df['floor_area_sqft'], errors='coerce')
    df['unit_number'] = pd.to_numeric(df['unit_number'], errors='coerce').astype('Int64')
    df['tower_number'] = pd.to_numeric(df['tower_number'], errors='coerce').astype('Int64')
    
    print("  → Parsing transaction dates...")
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df['transaction_year'] = df['transaction_date'].dt.year
    
    print("  → Standardizing categorical variables...")
    df['unit_category'] = df['unit_category'].str.strip().str.title()
    df['listing_status'] = df['listing_status'].str.strip().str.title()
    
    print("  → Handling missing values...")
    df = df.dropna(subset=['listing_id', 'sale_price', 'floor_area_sqft'])
    
    print(f"✅ Cleaned records: {len(df):,} (retention: {len(df)/original_count*100:.1f}%)")
    return df


# MERGE & FEATURE ENGINEERING

def merge_and_engineer_features(clients_df, properties_df):
    """Merge datasets and create features with advanced engineering"""
    print_section("STEP 4: MERGING DATASETS & FEATURE ENGINEERING")
    
    sold = properties_df[properties_df['listing_status'] == 'Sold'].copy()
    print(f"📊 Sold properties: {len(sold):,}")
    
    merged = sold.merge(clients_df, left_on='client_ref', right_on='client_id', how='inner')
    print(f"✅ Merged records: {len(merged):,}")
    
    print("\n  → Creating aggregated features...")
    
    features = merged.groupby('client_id').agg({
        'age': 'first',
        'gender': 'first',
        'client_type': 'first',
        'country': 'first',
        'region': 'first',
        'acquisition_purpose': 'first',
        'loan_applied': 'first',
        'referral_channel': 'first',
        'satisfaction_score': 'first',
        'sale_price': ['mean', 'sum', 'count'],
        'floor_area_sqft': 'mean',
        'listing_id': 'count'
    }).reset_index()
    
    features.columns = [
        'client_id', 'age', 'gender', 'client_type', 'country', 'region',
        'acquisition_purpose', 'loan_applied', 'referral_channel',
        'satisfaction_score', 'avg_sale_price', 'total_investment',
        'property_count', 'avg_floor_area', 'total_transactions'
    ]
    
    print("  → Creating advanced features...")
    
    # Age groups
    features['age_group'] = pd.cut(
        features['age'],
        bins=[0, 30, 40, 50, 60, 100],
        labels=['<30', '30-40', '40-50', '50-60', '60+']
    )
    
    # Price per sqft
    features['avg_price_per_sqft'] = features['avg_sale_price'] / features['avg_floor_area']
    features['avg_price_per_sqft'] = features['avg_price_per_sqft'].fillna(features['avg_price_per_sqft'].median())
    
    # Property diversity index
    features['property_diversity'] = features['property_count'] / features['total_transactions']
    features['property_diversity'] = features['property_diversity'].fillna(1.0)
    
    # Investment per year
    features['investment_span_years'] = 1
    features['investment_per_year'] = features['total_investment'] / features['investment_span_years']
    
    # Price range
    price_stats = merged.groupby('client_id')['sale_price'].agg(['min', 'max']).reset_index()
    price_stats['price_range'] = price_stats['max'] - price_stats['min']
    features = features.merge(price_stats[['client_id', 'price_range']], on='client_id', how='left')
    features['price_range'] = features['price_range'].fillna(0)
    
    print(f"✅ Created features for {len(features)} clients")
    return features


# ENCODING & SCALING

def encode_and_scale(features_df):
    """Encode categorical and scale numerical features"""
    print_section("STEP 5: FEATURE ENCODING & SCALING")
    
    df = features_df.copy()
    
    # Label encoding for binary/categorical columns
    le_dict = {}
    binary_cols = ['gender', 'client_type', 'loan_applied', 'acquisition_purpose']
    
    for col in binary_cols:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        le_dict[col] = le
    
    # One-hot encoding for multi-category columns
    df = pd.get_dummies(df, columns=['country', 'region', 'referral_channel'], drop_first=True)
    
    # Select features for clustering
    feature_cols = ['age', 'satisfaction_score', 'avg_sale_price', 
                    'total_investment', 'property_count', 'avg_floor_area',
                    'gender_encoded', 'client_type_encoded', 
                    'loan_applied_encoded', 'acquisition_purpose_encoded']
    
    # Add one-hot encoded columns
    one_hot_cols = [col for col in df.columns if col.startswith(('country_', 'region_', 'referral_channel_'))]
    feature_cols.extend(one_hot_cols)
    
    print(f"✅ Selected {len(feature_cols)} features for clustering")
    
    # Scale features
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[feature_cols])
    
    return df, scaled, scaler, feature_cols

def prepare_features_for_clustering(features_df):
    """Alias for encode_and_scale - used by main.py"""
    return encode_and_scale(features_df)


# SAVE / LOAD HELPERS

def save_figure(fig, path, dpi=300):
    """Save figure to reports folder"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"✅ Saved figure: {path}")

def save_processed_data(df, path):
    """Save processed data to CSV"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Saved processed data: {path}")

def save_model(model, path):
    """Save model using pickle"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Saved model: {path}")

def load_model(path):
    """Load model using pickle"""
    with open(path, 'rb') as f:
        return pickle.load(f)