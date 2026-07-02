import streamlit as st
import pandas as pd
import pickle
import time

# --- Page Configuration & High-Tech UI Settings ---
st.set_page_config(
    page_title="Shopper Spectrum | Retail Analytics SaaS", 
    page_icon="🛒", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- Advanced Custom CSS (Glassmorphism & Gradients) ---
st.markdown("""
    <style>
    /* Hide default Streamlit clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Gradient Title */
    .gradient-text {
        background: -webkit-linear-gradient(45deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0px;
    }
    
    /* Stylish Metric Cards */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease-in-out;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
    }
    
    /* Footer Styling */
    .pro-footer {
        text-align: center;
        padding: 20px;
        font-size: 0.9rem;
        color: #6c757d;
        margin-top: 50px;
        border-top: 1px solid #dee2e6;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Startup Toast (Micro-interaction) ---
# This will show a small non-intrusive popup when the app loads
st.toast('System Initialized: Machine Learning Models Loaded', icon='🚀')

# --- Load the Saved Models and Data ---
@st.cache_resource
def load_data():
    kmeans_model = pickle.load(open('kmeans_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    item_similarity_df = pd.read_pickle('item_similarity_matrix.pkl')
    rfm = pd.read_csv('rfm_data_for_app.csv')
    return kmeans_model, scaler, item_similarity_df, rfm

try:
    kmeans_model, scaler, item_similarity_df, rfm = load_data()
except Exception as e:
    st.error(f"System Failure: Unable to load AI models. {e}")
    st.stop()

# --- Professional Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3141/3141117.png", width=80)
    st.markdown("###  Shopper Spectrum")
    st.caption("v2.0.1 | Enterprise Retail Dashboard")
    st.divider()
    
    st.markdown("#### ⚡ System Architecture")
    st.markdown("- **Engine:** Python 3.14")
    st.markdown("- **UI Framework:** Streamlit")
    st.markdown("- **ML Pipeline:** Scikit-Learn")
    st.markdown("- **Data Structure:** Pandas DataFrames")
    
    st.divider()
    with st.expander(" How it works (Documentation)"):
        st.write("""
        **1. Recommendation Engine:** Uses Cosine Similarity on a User-Item matrix to find products bought together.
        
        **2. Segmentation:** Uses K-Means Clustering on RFM (Recency, Frequency, Monetary) data to profile users.
        """)

# --- Main App Header ---
st.markdown('<p class="gradient-text">Shopper Spectrum Analytics</p>', unsafe_allow_html=True)
st.markdown("Leveraging Unsupervised Machine Learning for real-time customer intelligence and product discovery.")
st.divider()

# --- Modular Tabs ---
tab1, tab2 = st.tabs([" Discovery Engine", " RFM Profiling Matrix"])

# --- TAB 1: Recommendation Engine ---
with tab1:
    st.markdown("###  Algorithmic Product Recommender")
    st.caption("Select a base product to trigger the Item-Based Collaborative Filtering matrix.")
    
    product_list = item_similarity_df.columns.tolist()
    
    col_search, _ = st.columns([2, 1])
    with col_search:
        selected_product = st.selectbox("Search Inventory:", product_list, index=0)
    
    if st.button("Initialize Recommendation Protocol", type="primary", use_container_width=False):
        # Adding a spinner for UI polish
        with st.spinner('Scanning transactional matrix for topological similarities...'):
            time.sleep(1) # Artificial delay for pro-feel
            
            similarity_scores = item_similarity_df[selected_product].sort_values(ascending=False)
            top_recommendations = similarity_scores.iloc[1:6]
            
            st.success(f"Analysis Complete. Top correlates for {selected_product} found.")
            st.divider()
            
            # Rendering cards using native containers
            rec_cols = st.columns(5)
            for i, (product, score) in enumerate(top_recommendations.items()):
                clean_product = str(product).strip()
                with rec_cols[i]:
                    with st.container(border=True):
                        st.markdown(f"**Match #{i+1}**")
                        st.caption(f"Confidence: {score*100:.1f}%")
                        st.write(f" {clean_product}")

# --- TAB 2: Customer Segmentation ---
with tab2:
    st.markdown("###  AI Customer Segmentation Model")
    st.caption("Input RFM parameters to execute the K-Means prediction algorithm in real-time.")
    
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            recency = st.number_input(" Recency (Days)", min_value=0, value=30, help="Days since the last transaction.")
        with col2:
            frequency = st.number_input(" Frequency (Orders)", min_value=1, value=5, help="Total count of historical orders.")
        with col3:
            monetary = st.number_input(" Monetary Value (₹)", min_value=0.0, value=5000.0, step=500.0, help="Total lifetime value in INR.")
        
        predict_btn = st.button("Execute Segmentation Model", type="primary", use_container_width=True)

    if predict_btn:
        with st.spinner('Processing RFM vectors through K-Means clustering...'):
            time.sleep(0.8) # Artificial delay
            
            input_data = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
            scaled_input = scaler.transform(input_data)
            predicted_cluster = kmeans_model.predict(scaled_input)[0]
            
            cluster_names = {
                0: ('Regular Customer ', 'normal', 'Standard retention protocols apply. Focus on upselling.'),
                1: ('At-Risk Customer ', 'error', 'High churn probability. Deploy aggressive win-back campaigns immediately.'),
                2: ('VIP / Champion ', 'success', 'Top percentile shopper. Enroll in exclusive loyalty programs.'),
                3: ('Loyal Customer ', 'info', 'Consistent engagement detected. Optimize for cross-selling.')
            }
            
            segment_name, alert_type, strategy = cluster_names.get(predicted_cluster, ("Unknown", "normal", ""))
            
            st.divider()
            
            # Results Section
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.markdown("#### Segment Profile")
                if alert_type == 'success':
                    st.success(f"**{segment_name}**")
                elif alert_type == 'error':
                    st.error(f"**{segment_name}**")
                elif alert_type == 'info':
                    st.info(f"**{segment_name}**")
                else:
                    st.warning(f"**{segment_name}**")
            
            with res_col2:
                st.markdown("#### Recommended Strategy")
                st.info(f" **Action Item:** {strategy}")
                
            st.markdown("#### Raw Input Metrics")
            m1, m2, m3 = st.columns(3)
            m1.metric(label="Recency", value=f"{recency} Days")
            m2.metric(label="Frequency", value=f"{frequency} Orders")
            m3.metric(label="Monetary", value=f"₹ {monetary:,.2f}")

# --- Professional Developer Footer ---
st.markdown(
    '<div class="pro-footer">System Engineered for High-Performance Analytics | Developed as a comprehensive data science solution.</div>', 
    unsafe_allow_html=True
)