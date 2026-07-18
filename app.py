"""
🏢 Parcl Real Estate Buyer Segmentation Dashboard
Advanced Streamlit Application for Buyer Intelligence
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from datetime import datetime


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Parcl Buyer Intelligence | AI Segmentation",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CUSTOM CSS - Professional Design

st.markdown("""
<style>
    /* Main Header */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    /* Sub Header */
    .sub-header {
        font-size: 1.1rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Metric Cards */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .metric-container:hover {
        transform: translateY(-5px);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Cluster Cards */
    .cluster-card {
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        border-left: 5px solid;
        background: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    .cluster-c1 { border-left-color: #FF6B6B; }
    .cluster-c2 { border-left-color: #4ECDC4; }
    .cluster-c3 { border-left-color: #45B7D1; }
    .cluster-c4 { border-left-color: #96CEB4; }
    
    /* Section Headers */
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3b82f6;
        display: inline-block;
    }
    
    /* Sidebar */
    .css-1d391kg { background: #f8fafc; }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 2rem;
        border-top: 1px solid #e2e8f0;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* DataFrame */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# COLOR PALETTE

CLUSTER_COLORS = {
    'Global Investors': '#FF6B6B',
    'First-Time Buyers': '#4ECDC4',
    'Corporate Buyers': '#45B7D1',
    'Luxury Investors': '#96CEB4'
}

CLUSTER_ICONS = {
    'Global Investors': '🌍',
    'First-Time Buyers': '🏠',
    'Corporate Buyers': '🏢',
    'Luxury Investors': '💎'
}


# LOAD DATA FUNCTION

@st.cache_data(ttl=3600)
def load_data():
    """Load and cache clustered data"""
    try:
        df = pd.read_csv('models/clustered_data.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Please run `python main.py` first to generate clustered data!")
        return None


# SIDEBAR

with st.sidebar:
    # Logo area
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: #1e3a8a; font-weight: 800;">🏢 PARCL</h2>
        <p style="color: #64748b; font-size: 0.8rem;">AI Buyer Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Filters
    st.header("🔍 Smart Filters")
    
    df = load_data()
    
    if df is not None:
        # Country filter
        countries = ['All Countries'] + sorted(df['country'].unique().tolist())
        selected_country = st.selectbox("🌍 Country", countries, key='country')
        
        # Region filter
        regions = ['All Regions'] + sorted(df['region'].unique().tolist())
        selected_region = st.selectbox("📍 Region", regions, key='region')
        
        # Purpose filter
        purposes = ['All Purposes'] + sorted(df['acquisition_purpose'].unique().tolist())
        selected_purpose = st.selectbox("🎯 Purpose", purposes, key='purpose')
        
        # Type filter
        types = ['All Types'] + sorted(df['client_type'].unique().tolist())
        selected_type = st.selectbox("👤 Client Type", types, key='type')
        
        # Satisfaction range
        st.markdown("---")
        st.subheader("⭐ Satisfaction Range")
        sat_min, sat_max = st.slider(
            "Select Range",
            min_value=1.0,
            max_value=5.0,
            value=(1.0, 5.0),
            step=0.5
        )
        
        # Investment range
        st.markdown("---")
        st.subheader("💰 Investment Range")
        inv_min, inv_max = st.slider(
            "Select Range ($)",
            min_value=int(df['total_investment'].min()),
            max_value=int(df['total_investment'].max()),
            value=(int(df['total_investment'].min()), int(df['total_investment'].max())),
            step=10000
        )
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; font-size: 0.75rem; color: #94a3b8;">
        <p>Powered by AI/ML</p>
        <p>© 2026 Parcl Co. Limited</p>
    </div>
    """, unsafe_allow_html=True)


# MAIN CONTENT


# Header
st.markdown('<p class="main-header">🏢 Real Estate Buyer Segmentation</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Driven Buyer Intelligence & Investment Profiling Dashboard</p>', unsafe_allow_html=True)

# Load data
df = load_data()

if df is not None:
    # Apply filters
    filtered_df = df.copy()
    
    if selected_country != 'All Countries':
        filtered_df = filtered_df[filtered_df['country'] == selected_country]
    if selected_region != 'All Regions':
        filtered_df = filtered_df[filtered_df['region'] == selected_region]
    if selected_purpose != 'All Purposes':
        filtered_df = filtered_df[filtered_df['acquisition_purpose'] == selected_purpose]
    if selected_type != 'All Types':
        filtered_df = filtered_df[filtered_df['client_type'] == selected_type]
    
    filtered_df = filtered_df[
        (filtered_df['satisfaction_score'] >= sat_min) &
        (filtered_df['satisfaction_score'] <= sat_max) &
        (filtered_df['total_investment'] >= inv_min) &
        (filtered_df['total_investment'] <= inv_max)
    ]
    
    # Show filter status
    if len(filtered_df) != len(df):
        st.info(f"📊 Showing **{len(filtered_df)}** of **{len(df)}** clients ({len(filtered_df)/len(df)*100:.1f}%)")
    
    
    # KPI CARDS
    
    st.markdown("---")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <div class="metric-value">{len(filtered_df):,}</div>
            <div class="metric-label">Total Clients</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div class="metric-value">${filtered_df['total_investment'].mean()/1000:.1f}K</div>
            <div class="metric-label">Avg Investment</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <div class="metric-value">{filtered_df['satisfaction_score'].mean():.2f}</div>
            <div class="metric-label">Avg Satisfaction</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <div class="metric-value">{filtered_df['property_count'].mean():.1f}</div>
            <div class="metric-label">Avg Properties</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        loan_rate = (filtered_df['loan_applied'] == 'Yes').mean() * 100
        st.markdown(f"""
        <div class="metric-container" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
            <div class="metric-value">{loan_rate:.1f}%</div>
            <div class="metric-label">Loan Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    
    # TABS
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "📈 Behavior",
        "🗺️ Geography",
        "📋 Insights",
        "📥 Export"
    ])
    
    
    # TAB 1: OVERVIEW
    
    with tab1:
        st.markdown('<p class="section-title">Buyer Segmentation Overview</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Interactive 3D Pie Chart
            cluster_counts = filtered_df['cluster_name'].value_counts().reset_index()
            cluster_counts.columns = ['Cluster', 'Count']
            
            fig = px.pie(
                cluster_counts,
                values='Count',
                names='Cluster',
                title='Segment Distribution',
                color='Cluster',
                color_discrete_map=CLUSTER_COLORS,
                hole=0.55
            )
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                textfont_size=12,
                pull=[0.02, 0.02, 0.02, 0.02],
                marker=dict(line=dict(color='white', width=2))
            )
            fig.update_layout(
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=-0.1),
                title_font_size=16,
                title_x=0.5
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cluster Summary Cards
            st.subheader("Segment Summary")
            
            for cluster_name in filtered_df['cluster_name'].unique():
                cluster_data = filtered_df[filtered_df['cluster_name'] == cluster_name]
                icon = CLUSTER_ICONS.get(cluster_name, '🔹')
                color = CLUSTER_COLORS.get(cluster_name, '#666')
                
                st.markdown(f"""
                <div class="cluster-card" style="border-left-color: {color};">
                    <h4 style="color: {color}; margin: 0;">{icon} {cluster_name}</h4>
                    <p style="margin: 0.5rem 0 0 0; color: #64748b; font-size: 0.9rem;">
                        {len(cluster_data)} clients • ${cluster_data['total_investment'].mean()/1000:.0f}K avg
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        # Detailed comparison table
        st.markdown("---")
        st.subheader("📊 Segment Comparison")
        
        comparison = filtered_df.groupby('cluster_name').agg({
            'age': 'mean',
            'avg_sale_price': 'mean',
            'total_investment': 'mean',
            'property_count': 'mean',
            'satisfaction_score': 'mean',
            'client_id': 'count'
        }).round(2)
        
        comparison.columns = ['Avg Age', 'Avg Price ($)', 'Avg Investment ($)', 
                             'Avg Properties', 'Avg Satisfaction', 'Count']
        comparison = comparison.reset_index()
        
        # Style the dataframe
        st.dataframe(
            comparison.style.background_gradient(subset=['Avg Investment ($)'], cmap='YlOrRd')
                                   .background_gradient(subset=['Avg Satisfaction'], cmap='YlGn'),
            use_container_width=True
        )
    
    
    # TAB 2: BEHAVIOR
    
    with tab2:
        st.markdown('<p class="section-title">Investor Behavior Analysis</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot - Investment by cluster
            fig = px.box(
                filtered_df,
                x='cluster_name',
                y='total_investment',
                color='cluster_name',
                color_discrete_map=CLUSTER_COLORS,
                title='Investment Distribution by Segment',
                labels={'total_investment': 'Total Investment ($)', 'cluster_name': 'Segment'}
            )
            fig.update_layout(showlegend=False, title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Violin plot - Age distribution
            fig = px.violin(
                filtered_df,
                x='cluster_name',
                y='age',
                color='cluster_name',
                color_discrete_map=CLUSTER_COLORS,
                box=True,
                title='Age Distribution by Segment',
                labels={'age': 'Age (years)', 'cluster_name': 'Segment'}
            )
            fig.update_layout(showlegend=False, title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
        
        # Loan vs Investment Purpose
        col1, col2 = st.columns(2)
        
        with col1:
            loan_pivot = filtered_df.groupby(['cluster_name', 'loan_applied']).size().reset_index(name='count')
            fig = px.bar(
                loan_pivot,
                x='cluster_name',
                y='count',
                color='loan_applied',
                title='Loan Application by Segment',
                barmode='group',
                color_discrete_sequence=['#FF6B6B', '#4ECDC4']
            )
            fig.update_layout(title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            purpose_pivot = filtered_df.groupby(['cluster_name', 'acquisition_purpose']).size().reset_index(name='count')
            fig = px.bar(
                purpose_pivot,
                x='cluster_name',
                y='count',
                color='acquisition_purpose',
                title='Acquisition Purpose by Segment',
                barmode='group',
                color_discrete_sequence=['#45B7D1', '#96CEB4']
            )
            fig.update_layout(title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
        
        # Sunburst chart
        st.markdown("---")
        sunburst_data = filtered_df.groupby(['cluster_name', 'acquisition_purpose', 'loan_applied']).size().reset_index(name='count')
        fig = px.sunburst(
            sunburst_data,
            path=['cluster_name', 'acquisition_purpose', 'loan_applied'],
            values='count',
            title='Hierarchical Segment Analysis',
            color='cluster_name',
            color_discrete_map=CLUSTER_COLORS
        )
        fig.update_layout(title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)
    
    
    # TAB 3: GEOGRAPHY
    
    with tab3:
        st.markdown('<p class="section-title">Geographic Buyer Analysis</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Choropleth-style bar chart
            country_data = filtered_df.groupby(['country', 'cluster_name']).agg({
                'client_id': 'count',
                'total_investment': 'mean'
            }).reset_index()
            country_data.columns = ['Country', 'Segment', 'Clients', 'Avg Investment']
            
            fig = px.bar(
                country_data,
                x='Country',
                y='Clients',
                color='Segment',
                color_discrete_map=CLUSTER_COLORS,
                title='Buyer Segments by Country',
                barmode='group'
            )
            fig.update_layout(title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Region heatmap
            region_pivot = filtered_df.pivot_table(
                values='total_investment',
                index='region',
                columns='cluster_name',
                aggfunc='mean'
            ).fillna(0)
            
            fig = px.imshow(
                region_pivot,
                title='Average Investment Heatmap by Region',
                color_continuous_scale='RdYlGn',
                aspect='auto'
            )
            fig.update_layout(title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
        
        # Treemap
        st.markdown("---")
        geo_agg = filtered_df.groupby(['country', 'region']).agg({
            'client_id': 'count',
            'total_investment': 'mean',
            'satisfaction_score': 'mean'
        }).reset_index()
        
        fig = px.treemap(
            geo_agg,
            path=['country', 'region'],
            values='client_id',
            color='total_investment',
            title='Geographic Investment Distribution',
            color_continuous_scale='Viridis',
            hover_data=['satisfaction_score']
        )
        fig.update_layout(title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)
    
    
    # TAB 4: INSIGHTS
    
    with tab4:
        st.markdown('<p class="section-title">Deep Segment Insights</p>', unsafe_allow_html=True)
        
        # Select cluster for deep dive
        selected_cluster = st.selectbox(
            "🔍 Select Segment for Deep Analysis",
            options=filtered_df['cluster_name'].unique()
        )
        
        cluster_data = filtered_df[filtered_df['cluster_name'] == selected_cluster]
        icon = CLUSTER_ICONS.get(selected_cluster, '🔹')
        color = CLUSTER_COLORS.get(selected_cluster, '#666')
        
        # Header
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {color}22, {color}11); 
                    border-radius: 15px; padding: 2rem; margin: 1rem 0;
                    border: 2px solid {color};">
            <h2 style="color: {color}; margin: 0;">{icon} {selected_cluster}</h2>
            <p style="color: #64748b; margin: 0.5rem 0 0 0;">
                {len(cluster_data)} clients in current filter
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg Age", f"{cluster_data['age'].mean():.1f}")
        with col2:
            st.metric("Avg Investment", f"${cluster_data['total_investment'].mean():,.0f}")
        with col3:
            st.metric("Avg Satisfaction", f"{cluster_data['satisfaction_score'].mean():.2f}")
        with col4:
            st.metric("Properties/Client", f"{cluster_data['property_count'].mean():.1f}")
        
        # Detailed charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Age histogram
            fig = px.histogram(
                cluster_data,
                x='age',
                nbins=15,
                title=f'{selected_cluster} - Age Distribution',
                color_discrete_sequence=[color]
            )
            fig.update_layout(showlegend=False, title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Satisfaction histogram
            fig = px.histogram(
                cluster_data,
                x='satisfaction_score',
                nbins=5,
                title=f'{selected_cluster} - Satisfaction Distribution',
                color_discrete_sequence=[color]
            )
            fig.update_layout(showlegend=False, title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
        
        # Investment vs Area scatter
        fig = px.scatter(
            cluster_data,
            x='avg_floor_area',
            y='avg_sale_price',
            size='property_count',
            color='satisfaction_score',
            title=f'{selected_cluster} - Price vs Area Analysis',
            labels={
                'avg_floor_area': 'Floor Area (sqft)',
                'avg_sale_price': 'Sale Price ($)',
                'property_count': 'Properties'
            },
            color_continuous_scale='Plasma'
        )
        fig.update_layout(title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)
    
    
    # TAB 5: EXPORT
    
    with tab5:
        st.markdown('<p class="section-title">Data Export Center</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📥 Download Filtered Data")
            
            # CSV Export
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name=f'parcl_buyer_segmentation_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv',
                use_container_width=True
            )
            
            # JSON Export
            json = filtered_df.to_json(orient='records', indent=2)
            st.download_button(
                label="⬇️ Download JSON",
                data=json,
                file_name=f'parcl_buyer_segmentation_{datetime.now().strftime("%Y%m%d")}.json',
                mime='application/json',
                use_container_width=True
            )
        
        with col2:
            st.subheader("📊 Export Statistics")
            
            # Summary statistics
            stats = filtered_df.groupby('cluster_name').agg({
                'client_id': 'count',
                'total_investment': ['mean', 'sum'],
                'satisfaction_score': 'mean'
            }).round(2)
            
            stats_csv = stats.to_csv().encode('utf-8')
            st.download_button(
                label="⬇️ Download Summary Stats",
                data=stats_csv,
                file_name=f'parcl_summary_stats_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv',
                use_container_width=True
            )
        
        # Data preview
        st.markdown("---")
        st.subheader("🔍 Data Preview")
        st.dataframe(filtered_df.head(20), use_container_width=True)
    
    
    # FOOTER
    
    st.markdown("""
    <div class="footer">
        <p>🏢 <strong>Parcl Co. Limited</strong> | Unified Mentor | Machine Learning Internship 2026</p>
        <p>Developed by: Yunes Abdulghani Mohammed Ghaleb | AI & ML Engineer</p>
        <p style="font-size: 0.75rem; margin-top: 0.5rem;">
            This dashboard uses AI-powered clustering algorithms to reveal hidden buyer segments 
            and drive data-driven real estate investment decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Error state
    st.error("❌ Data not found!")
    st.info("""
    **To get started:**
    1. Ensure `clients.csv` and `properties.csv` are in the `data/` folder
    2. Run: `python main.py` to generate clustered data
    3. Refresh this page
    """)
    
    # Show expected structure
    st.code("""
    Real_Estate_Buyer_Segmentation_Project/
    ├── data/
    │   ├── clients.csv
    │   └── properties.csv
    ├── models/
    │   └── clustered_data.csv  ← Generated by main.py
    └── app.py
    """, language='text')