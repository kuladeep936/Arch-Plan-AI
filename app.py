import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import re
import io
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import base64

# Page configuration
st.set_page_config(
    page_title="ArchPlan AI - Intelligent Architecture Pricing",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Updated Building Types and Styles
BUILDING_TYPES = ["Residential", "Apartment", "Commercial"]
ARCHITECTURAL_STYLES = [
    "Single-Storey House",
    "Two-Storey House", 
    "Triple-Storey House",
    "Multi-Storey House",
    "Single-Storey Supermarket",
    "Single-Storey Departmental Store",
    "Single-Storey Cafe / Restaurant"
]

# Custom CSS for ArchPlan AI branding
st.markdown("""
<style>
    /* Main Brand Colors */
    :root {
        --primary: #2E86AB;
        --secondary: #A23B72;
        --accent: #F18F01;
        --dark: #2B2D42;
        --light: #F8F9FA;
        --success: #28a745;
    }
    
    .main-header {
        font-size: 4rem;
        background: linear-gradient(45deg, var(--primary), var(--secondary), var(--accent));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .section-header {
        font-size: 2.5rem;
        background: linear-gradient(45deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        font-weight: 700;
        text-align: center;
    }
    
    .tagline {
        text-align: center;
        color: var(--dark);
        font-size: 1.4rem;
        margin-bottom: 3rem;
        font-weight: 300;
        opacity: 0.9;
    }
    
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 4rem 2rem;
        border-radius: 25px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        padding: 3rem;
        border-radius: 25px;
        color: white;
        text-align: center;
        box-shadow: 0 20px 40px rgba(46, 134, 171, 0.3);
        margin: 2rem 0;
        border: none;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(180deg); }
    }
    
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin: 1rem 0;
        border: none;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        text-align: center;
    }
    
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 25px 50px rgba(0,0,0,0.15);
    }
    
    .arch-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin: 1rem 0;
        text-align: center;
        border: none;
        transition: all 0.3s ease;
    }
    
    .arch-card:hover {
        transform: scale(1.02);
        box-shadow: 0 15px 30px rgba(0,0,0,0.2);
    }
    
    .similar-plan-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 20px;
        margin: 1rem;
        text-align: center;
        border: 3px solid rgba(255,255,255,0.8);
        transition: all 0.3s ease;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .similar-plan-card:hover {
        transform: translateY(-8px) scale(1.03);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }
    
    .stButton>button {
        background: linear-gradient(45deg, var(--primary), var(--secondary));
        color: white;
        border: none;
        padding: 1rem 3rem;
        border-radius: 50px;
        font-weight: bold;
        font-size: 1.2rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(46, 134, 171, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(46, 134, 171, 0.6);
    }
    
    .plan-comparison {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 3rem;
        border-radius: 30px;
        margin: 3rem 0;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        border: 2px solid rgba(255,255,255,0.8);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.15);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .team-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem;
        border-radius: 25px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .team-card:hover {
        transform: translateY(-10px);
    }
    
    .tech-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin: 1.5rem 0;
        transition: all 0.3s ease;
    }
    
    .tech-card:hover {
        transform: scale(1.05);
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
    }
    
    /* Custom badges */
    .ai-badge {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.3rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .developer-badge {
        background: linear-gradient(45deg, #2E86AB, #A23B72);
        color: white;
        padding: 0.7rem 1.2rem;
        border-radius: 25px;
        font-size: 1rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
    
    .nav-button {
        background: linear-gradient(45deg, var(--primary), var(--secondary));
        color: white;
        padding: 1rem 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
        border: none;
        font-weight: 600;
    }
    
    .nav-button:hover {
        transform: translateX(10px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    
    .stats-container {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        margin: 2rem 0;
    }
    
    .stat-item {
        text-align: center;
        padding: 1.5rem;
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        margin: 0.5rem;
        flex: 1;
        min-width: 150px;
        backdrop-filter: blur(10px);
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .similar-plans-container {
        display: flex;
        flex-direction: row;
        justify-content: space-between;
        gap: 1rem;
        overflow-x: auto;
        padding: 1rem 0;
    }
    
    .similar-plan-item {
        flex: 1;
        min-width: 250px;
    }
</style>
""", unsafe_allow_html=True)

class ArchPlanPredictor:
    def __init__(self):
        self.model = None
        self.le_building = None
        self.le_style = None
        self.feature_columns = None
        self.df = None
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize and train the model directly in the app"""
        try:
            # Load and preprocess the dataset directly in the app
            data = """Name,Plot Area,Total Built-up Area,Width,Length,Building Type,Style,Estimated Cost
single house plan,1500 sq. ft.,1300 sq. ft.,30 ft.,50 ft.,Residential,Single-Storey House,20 - 22 Lakhs
Single house plan 1,2000 sq. ft.,1442 sq. ft.,40 ft.,50 ft.,Residential,Single-storey House,22 – 25 Lakhs
Single house plan 2,3000 sq. ft.,2330 sq. ft.,50 ft.,60 ft.,Residential,Single-Storey House,35 – 40 Lakhs
Two Floor plan,2000 sq. ft.,1800 sq. ft.,40 ft.,50 ft.,Residential,Two-Storey House,27 – 31 Lakhs
Two Floor plan 1,2800 sq. ft.,2500 sq. ft.,40 ft.,70 ft.,Residential,Two-Storey House,38 – 43 Lakhs
Two Floor plan 2,2000 sq. ft.,1052 sq. ft.,40 ft.,50 ft.,Residential,Two-Storey House,16 – 18 Lakhs
Three Floor plan,3600 sq. ft.,1700 sq. ft.,40 ft.,65 ft.,Residential,Triple-Storey House,26 – 29 Lakhs
Three Floor plan 1,3500 sq. ft.,2000 sq. ft.,50 ft.,75 ft.,Residential,Triple-Storey House,30 – 34 Lakhs
Three Floor plan 2,2190 sq. ft.,6570 sq. ft.,30 ft.,73 ft.,Residential,Triple-Storey House,99 Lakhs– 1.12 Crores
Apartment Plan1,4200 sq. ft.,7800 sq. ft.,50 ft.,80 ft.,Apartment,Multi-Storey House,1– 1.8 Crores
Apartment Plan2,2600 sq. ft.,7200 sq. ft.,55 ft.,47 ft.,Apartment,Multi-Storey House,1.4–1.8 Crores
Apartment Plan3,1900 sq. ft.,5200 sq. ft.,57 ft.,40 ft.,Apartment,Multi-Storey House,85 Lakhs–1.1 Crores
Commercial 1,5000 sq. ft.,7500 sq. ft.,70 ft.,110 ft.,Commercial,Single-Storey Supermarket,1.5 – 2.0 Crores
Commercial 2,6000 sq. ft.,9000 sq. ft.,80 ft.,115 ft.,Commercial,Single-Storey Departmental Store,2.0 – 2.6 Crores
Commercial 3,1800 sq. ft.,2400 sq. ft.,30 ft.,60 ft.,Commercial,Single-Storey Cafe / Restaurant,65 – 90 Lakhs"""

            # Use io.StringIO instead of pandas.compat.StringIO
            self.df = pd.read_csv(io.StringIO(data))
            
            # Preprocess the data
            self.df = self.clean_data(self.df)
            self.df['Price_Lakhs'] = self.df['Estimated Cost'].apply(self.parse_cost)
            
            # Calculate additional features
            self.df['Area_Ratio'] = self.df['Built_up_Area_num'] / self.df['Plot_Area_num']
            self.df['Perimeter'] = 2 * (self.df['Width_num'] + self.df['Length_num'])
            
            # Encode categorical variables
            self.le_building = LabelEncoder()
            self.le_style = LabelEncoder()
            
            self.df['Building_Type_encoded'] = self.le_building.fit_transform(self.df['Building Type'])
            self.df['Style_encoded'] = self.le_style.fit_transform(self.df['Style'])
            
            # Select features for modeling
            self.feature_columns = ['Plot_Area_num', 'Built_up_Area_num', 'Width_num', 'Length_num', 
                                  'Area_Ratio', 'Perimeter', 'Building_Type_encoded', 'Style_encoded']
            
            X = self.df[self.feature_columns]
            y = self.df['Price_Lakhs']
            
            # Train the model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
            
            self.model.fit(X, y)
            
            return True
            
        except Exception as e:
            st.error(f"❌ Model initialization failed: {e}")
            return False
    
    def clean_data(self, df):
        """Clean architectural data"""
        df.columns = df.columns.str.strip()
        
        # Extract numeric values from strings
        df['Plot_Area_num'] = df['Plot Area'].apply(lambda x: float(str(x).replace(' sq. ft.', '').strip()))
        df['Built_up_Area_num'] = df['Total Built-up Area'].apply(lambda x: float(str(x).replace(' sq. ft.', '').strip()))
        df['Width_num'] = df['Width'].apply(lambda x: float(str(x).replace(' ft.', '').strip()))
        df['Length_num'] = df['Length'].apply(lambda x: float(str(x).replace(' ft.', '').strip()))
        
        return df
    
    def parse_cost(self, cost_str):
        """Parse cost string for architectural plans"""
        cost_str = str(cost_str).strip().lower()
        if 'crores' in cost_str or 'crore' in cost_str:
            numbers = re.findall(r'[\d.]+', cost_str)
            if len(numbers) >= 2:
                return (float(numbers[0]) + float(numbers[1])) / 2 * 100
            elif numbers:
                return float(numbers[0]) * 100
        elif 'lakhs' in cost_str or 'lakh' in cost_str:
            numbers = re.findall(r'[\d.]+', cost_str)
            if len(numbers) >= 2:
                return (float(numbers[0]) + float(numbers[1])) / 2
            elif numbers:
                return float(numbers[0])
        numbers = re.findall(r'[\d.]+', cost_str)
        if numbers:
            return float(numbers[0])
        return 0.0
    
    def create_features(self, plot_area, built_up_area, width, length, style, building_type):
        """Create feature vector for prediction"""
        try:
            # Calculate derived features
            area_ratio = built_up_area / plot_area if plot_area > 0 else 0
            perimeter = 2 * (width + length)
            
            # Encode categorical variables
            building_encoded = self.le_building.transform([building_type])[0] if self.le_building else 0
            style_encoded = self.le_style.transform([style])[0] if self.le_style else 0
            
            # Create feature array in the exact same order as training
            features = np.array([[
                plot_area, built_up_area, width, length, 
                area_ratio, perimeter, building_encoded, style_encoded
            ]])
            
            return features
            
        except Exception as e:
            st.error(f"Feature creation error: {e}")
            return None
    
    def predict_cost(self, plot_area, built_up_area, width, length, style, building_type):
        """Predict construction cost using the model"""
        try:
            # Create features
            features = self.create_features(plot_area, built_up_area, width, length, style, building_type)
            if features is None:
                return None, []
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            
            # Ensure prediction is reasonable
            prediction = max(prediction, 5)  # Minimum 5 lakhs
            
            # Find similar plans
            similar_plans = self.find_similar_plans(prediction, style, building_type)
            
            return prediction, similar_plans
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return None, []
    
    def find_similar_plans(self, predicted_price, style, building_type, n_plans=4):
        """Find similar architectural plans based on price and features"""
        try:
            if self.df is None:
                return []
            
            # Filter by style and building type
            filtered_df = self.df[
                (self.df['Style'].str.contains(style, case=False, na=False)) |
                (self.df['Building Type'].str.contains(building_type, case=False, na=False))
            ]
            
            if len(filtered_df) == 0:
                filtered_df = self.df
            
            # Calculate price similarity
            price_differences = np.abs(filtered_df['Price_Lakhs'].values - predicted_price)
            closest_indices = np.argsort(price_differences)[:n_plans]
            
            similar_plans = []
            for idx in closest_indices:
                plan_data = filtered_df.iloc[idx]
                similarity_score = max(0, 100 - (price_differences[idx] / predicted_price * 100))
                
                plan_info = {
                    'name': plan_data['Name'],
                    'price': plan_data['Price_Lakhs'],
                    'style': plan_data['Style'],
                    'type': plan_data['Building Type'],
                    'plot_area': plan_data['Plot_Area_num'],
                    'built_up_area': plan_data['Built_up_Area_num'],
                    'width': plan_data['Width_num'],
                    'length': plan_data['Length_num'],
                    'similarity': similarity_score
                }
                similar_plans.append(plan_info)
            
            return similar_plans
            
        except Exception as e:
            st.error(f"Similar plans search failed: {e}")
            return []

def display_cost_breakdown(prediction, built_up_area):
    """Display architectural cost breakdown"""
    cost_lakhs = prediction
    cost_rupees = cost_lakhs * 100000
    
    # Architectural cost components
    components = {
        'Structure & Foundation': 25,
        'Architectural Finishes': 20,
        'Mechanical Systems': 15,
        'Electrical & Plumbing': 12,
        'Interior Design': 10,
        'Landscaping': 8,
        'Architect Fees': 5,
        'Permits & Contingency': 5
    }
    
    costs = {k: cost_lakhs * v/100 for k, v in components.items()}
    
    # Create beautiful pie chart
    fig = px.pie(
        values=list(costs.values()),
        names=list(costs.keys()),
        title="🏗️ Architectural Cost Distribution",
        color_discrete_sequence=px.colors.sequential.Viridis,
        hole=0.4
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

def display_similar_plans_horizontal(similar_plans, predicted_price):
    """Display similar architectural plans in horizontal layout"""
    if not similar_plans:
        st.info("🏗️ No similar architectural designs found in our database.")
        return
    
    st.markdown("---")
    st.markdown('<div class="plan-comparison">', unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">🏛️ Similar Architectural Designs</h2>', unsafe_allow_html=True)
    st.write(f"*Designs with comparable features to your ₹{predicted_price:.2f}L project*")
    
    # Create horizontal container using columns
    cols = st.columns(len(similar_plans))
    
    # Display plans horizontally
    for i, (col, plan) in enumerate(zip(cols, similar_plans)):
        with col:
            st.markdown(f"""
            <div class="similar-plan-card">
                <h4>🏠 {plan['name']}</h4>
                <span class="ai-badge">₹{plan['price']:.2f}L</span><br>
                <strong>Style:</strong> {plan['style']}<br>
                <strong>Type:</strong> {plan['type']}<br>
                <strong>Plot:</strong> {plan['plot_area']} sq ft<br>
                <strong>Built-up:</strong> {plan['built_up_area']} sq ft<br>
                <strong>Dimensions:</strong> {plan['width']}×{plan['length']} ft<br>
                <span class="ai-badge">{plan['similarity']:.1f}% match</span>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_prediction_page():
    """Display the prediction page"""
    st.markdown('<h1 class="main-header">Cost Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">AI-Powered Architectural Cost Estimation</p>', unsafe_allow_html=True)
    
    # Initialize ArchPlan AI predictor
    predictor = ArchPlanPredictor()
    
    if not predictor.model:
        st.error("Model initialization failed. Please check the data and try again.")
        return

    # Main Prediction Interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="arch-card">
            <h3>📐 Plan Details</h3>
            <p>Enter your architectural plan specifications for AI analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("architectural_form", border=False):
            # Architectural Style Section
            st.subheader("🎨 Architectural Style")
            style = st.selectbox(
                "Design Style",
                ARCHITECTURAL_STYLES,
                index=0,
                help="Select the architectural style of your project"
            )
            
            # Building Type Section
            st.subheader("🏗️ Building Type")
            building_type = st.selectbox(
                "Structure Type", 
                BUILDING_TYPES,
                index=0,
                help="Choose the type of building"
            )
            
            # Architectural Dimensions
            st.subheader("📏 Design Dimensions")
            dim_col1, dim_col2 = st.columns(2)
            with dim_col1:
                plot_area = st.number_input(
                    "Plot Area (sq ft)", 
                    min_value=500, 
                    max_value=50000, 
                    value=2000, 
                    step=100,
                    help="Total land area for construction"
                )
                width = st.number_input(
                    "Width (ft)", 
                    min_value=20, 
                    max_value=500, 
                    value=40, 
                    step=5,
                    help="Building width dimension"
                )
            with dim_col2:
                built_up_area = st.number_input(
                    "Built-up Area (sq ft)", 
                    min_value=500, 
                    max_value=30000, 
                    value=1500, 
                    step=100,
                    help="Total constructed area"
                )
                length = st.number_input(
                    "Length (ft)", 
                    min_value=20, 
                    max_value=500, 
                    value=50, 
                    step=5,
                    help="Building length dimension"
                )
            
            # Professional Submit
            submitted = st.form_submit_button(
                "🚀 Analyze with ArchPlan AI", 
                use_container_width=True
            )
    
    with col2:
        st.markdown("""
        <div class="arch-card">
            <h3>💡 AI Analysis</h3>
            <p>Real-time cost predictions and architectural insights</p>
        </div>
        """, unsafe_allow_html=True)
        
        if submitted:
            with st.spinner("🏛️ ArchPlan AI is analyzing your design..."):
                # Get architectural prediction
                prediction, similar_plans = predictor.predict_cost(
                    plot_area, built_up_area, width, length, 
                    style, building_type
                )
                
                if prediction is not None:
                    # Professional Prediction Card
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    
                    st.metric(
                        label="🏛️ Estimated Construction Cost", 
                        value=f"₹{prediction:,.2f} Lakhs",
                        delta="AI Calculated"
                    )
                    
                    # Professional conversions
                    cost_lakhs = prediction
                    cost_crores = cost_lakhs / 100
                    cost_rupees = cost_lakhs * 100000
                    cost_per_sqft = (cost_rupees / built_up_area) if built_up_area > 0 else 0
                    
                    st.write(f"**≈ {cost_crores:.2f} Crores**")
                    st.write(f"**≈ ₹{cost_rupees:,.0f}**")
                    st.write(f"**Construction Rate: ₹{cost_per_sqft:,.0f} per sq ft**")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Architectural Summary
                    st.subheader("📋 Design Summary")
                    summary_col1, summary_col2 = st.columns(2)
                    
                    with summary_col1:
                        st.markdown(f"""
                        <div class="arch-card">
                        <h4>🏗️ Construction Profile</h4>
                        • Plot Area: {plot_area:,} sq ft<br>
                        • Built-up Area: {built_up_area:,} sq ft<br>
                        • Footprint: {width}ft × {length}ft<br>
                        • Utilization: {(built_up_area/plot_area*100):.1f}%
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with summary_col2:
                        st.markdown(f"""
                        <div class="arch-card">
                        <h4>🎨 Design Profile</h4>
                        • Style: {style}<br>
                        • Type: {building_type}<br>
                        • Rate: ₹{cost_per_sqft:,.0f}/sq ft<br>
                        • Perimeter: {2*(width+length)} ft
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Professional Cost Breakdown
                    st.subheader("📊 Cost Analysis")
                    display_cost_breakdown(prediction, built_up_area)
                    
                    # Similar Architectural Designs - Horizontal Layout
                    display_similar_plans_horizontal(similar_plans, prediction)
                        
        else:
            # Interactive Welcome
            st.markdown("""
            <div style='text-align: center; padding: 3rem; background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); border-radius: 20px;'>
                <h3>🏛️ Ready for Analysis</h3>
                <p>Fill out the form to get your AI-powered architectural cost estimate</p>
                <p><em>Your comprehensive construction analysis awaits!</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Interactive Visualization
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode = "gauge+number",
                value = 0,
                title = {'text': "ArchPlan AI Ready", 'font': {'size': 24}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#2E86AB"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 33], 'color': 'lightgray'},
                        {'range': [33, 66], 'color': 'gray'},
                        {'range': [66, 100], 'color': 'darkgray'}],
                }
            ))
            fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

# [Include all the other functions: show_home_page, show_about_page, main exactly as before]

def show_home_page():
    """Display Home page with hero section and features"""
    st.markdown('<h1 class="main-header">ArchPlan AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">Revolutionizing Architectural Cost Estimation with Artificial Intelligence</p>', unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h2 style='font-size: 2.5rem; margin-bottom: 1rem;'>🏛️ Intelligent Architecture Pricing</h2>
        <p style='font-size: 1.3rem; margin-bottom: 2rem;'>
        Transform your architectural planning with AI-powered cost predictions, 
        design matching, and professional analytics.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Stats
    st.markdown("""
    <div class="stats-container">
        <div class="stat-item">
            <h3>🚀</h3>
            <h4>Instant Predictions</h4>
            <p>Get accurate cost estimates in seconds</p>
        </div>
        <div class="stat-item">
            <h3>🎯</h3>
            <h4>AI-Powered</h4>
            <p>Advanced machine learning models</p>
        </div>
        <div class="stat-item">
            <h3>🏗️</h3>
            <h4>Professional Grade</h4>
            <p>Architect-approved calculations</p>
        </div>
        <div class="stat-item">
            <h3>📊</h3>
            <h4>Detailed Analytics</h4>
            <p>Comprehensive cost breakdowns</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Grid
    st.markdown('<h2 class="section-header">✨ Key Features</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🤖 AI Cost Prediction</h3>
            <p>Advanced neural networks trained on architectural data provide accurate construction cost estimates.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>🎨 Design Intelligence</h3>
            <p>Smart analysis of architectural styles and building types for precise pricing.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🏛️ Similar Plan Matching</h3>
            <p>Find comparable architectural designs with visual references and pricing.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>📈 Market Insights</h3>
            <p>Real-time cost analytics and professional construction rate calculations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>💼 Professional Reports</h3>
            <p>Detailed cost breakdowns and architectural analysis for clients.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>🔧 Smart Multipliers</h3>
            <p>Intelligent adjustments based on building type and architectural style.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Call to Action
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center;'>
            <h3>Ready to Transform Your Architectural Planning?</h3>
            <p>Get started with AI-powered cost estimation today!</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Start Predicting Now", use_container_width=True):
            st.session_state.current_page = "Prediction"
            st.rerun()

def show_about_page():
    """Display About page with enhanced design"""
    st.markdown('<h1 class="main-header">About ArchPlan AI</h1>', unsafe_allow_html=True)
    
    # Mission Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h2>🎯 Our Mission</h2>
            <p style='font-size: 1.2rem;'>
            ArchPlan AI revolutionizes architectural cost estimation by combining 
            <strong>artificial intelligence</strong> with <strong>architectural expertise</strong>. 
            Our platform provides accurate, data-driven construction cost predictions that help 
            architects, builders, and homeowners make informed decisions.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h2>🏆 Vision</h2>
            <p style='font-size: 1.2rem;'>
            To become the world's most trusted AI-powered architectural 
            planning platform, making professional cost estimation 
            accessible to everyone.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Development Team Section
    st.markdown('<h2 class="section-header">👥 Development Team</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="team-card">
        <h2>🏛️ Developed with Excellence</h2>
        <p style='font-size: 1.2rem;'>ArchPlan AI represents the culmination of extensive research and development in 
        architectural AI and machine learning applications, created by talented developers passionate about 
        transforming the construction industry.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Developer Cards
    dev_col1, dev_col2 = st.columns(2)
    
    with dev_col1:
        st.markdown("""
        <div class="team-card">
            <h3>E Kuladeep</h3>
            <p><strong>Lead Developer & AI Engineer</strong></p>
            <p>Machine Learning Specialist with expertise in neural networks, 
            computer vision, and data science applications for architectural planning.</p>
            <span class="developer-badge">AI Architecture</span>
            <span class="developer-badge">Model Training</span>
            <span class="developer-badge">System Design</span>
            <span class="developer-badge">Data Science</span>
        </div>
        """, unsafe_allow_html=True)
    
    with dev_col2:
        st.markdown("""
        <div class="team-card">
            <h3>G Aquill Rao</h3>
            <p><strong>Full Stack Developer & UI/UX Specialist</strong></p>
            <p>Web development expert focused on creating intuitive user experiences 
            and seamless system integration for architectural applications.</p>
            <span class="developer-badge">Frontend Development</span>
            <span class="developer-badge">UI/UX Design</span>
            <span class="developer-badge">API Integration</span>
            <span class="developer-badge">System Architecture</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Technology Stack
    st.markdown('<h2 class="section-header">🛠️ Technology Stack</h2>', unsafe_allow_html=True)
    
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.markdown("""
        <div class="tech-card">
            <h4>🤖 AI & Machine Learning</h4>
            • TensorFlow & Keras<br>
            • PyTorch & Transformers<br>
            • CLIP Model (OpenAI)<br>
            • Scikit-learn<br>
            • Neural Networks<br>
            • Computer Vision
        </div>
        """, unsafe_allow_html=True)
    
    with tech_col2:
        st.markdown("""
        <div class="tech-card">
            <h4>🌐 Web Technologies</h4>
            • Streamlit Framework<br>
            • Plotly Visualization<br>
            • Pandas & NumPy<br>
            • Pillow (Image Processing)<br>
            • Joblib<br>
            • Beautiful Visual Design
        </div>
        """, unsafe_allow_html=True)
    
    with tech_col3:
        st.markdown("""
        <div class="tech-card">
            <h4>📊 Data & Analytics</h4>
            • Feature Engineering<br>
            • Similarity Search<br>
            • Cost Prediction Models<br>
            • Image Processing<br>
            • Data Visualization<br>
            • Professional Reporting
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features Highlight
    st.markdown('<h2 class="section-header">🚀 Platform Capabilities</h2>', unsafe_allow_html=True)
    
    feature_col1, feature_col2 = st.columns(2)
    
    with feature_col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Core Features</h3>
            • <strong>AI-Powered Cost Estimation</strong><br>
            • <strong>Architectural Style Analysis</strong><br>
            • <strong>Similar Design Matching</strong><br>
            • <strong>Real-time Price Predictions</strong><br>
            • <strong>Professional Cost Breakdowns</strong><br>
            • <strong>Visual Plan Comparisons</strong>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_col2:
        st.markdown("""
        <div class="feature-card">
            <h3>💡 Advanced Capabilities</h3>
            • <strong>CLIP Model Integration</strong><br>
            • <strong>Neural Network Predictions</strong><br>
            • <strong>Interactive Visualizations</strong><br>
            • <strong>Market Comparison Tools</strong><br>
            • <strong>Professional Reporting</strong><br>
            • <strong>Multi-format Output</strong>
        </div>
        """, unsafe_allow_html=True)
    
    # Future Roadmap
    st.markdown("---")
    st.markdown("""
    <div class="arch-card">
        <h2>🔮 Future Roadmap</h2>
        <p>• <strong>Mobile Application</strong> - iOS & Android</p>
        <p>• <strong>3D Model Integration</strong> - BIM support</p>
        <p>• <strong>AR/VR Visualization</strong> - Immersive planning</p>
        <p>• <strong>International Pricing</strong> - Global markets</p>
        <p>• <strong>Contractor Network</strong> - Professional connections</p>
        <p>• <strong>Material Cost Database</strong> - Real-time pricing</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Initialize session state for page navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 2rem 1rem;'>
            <h1 style='font-size: 2.5rem; margin-bottom: 0.5rem;'>🏛️</h1>
            <h2>ArchPlan AI</h2>
            <p><em>Intelligent Architecture Platform</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation Menu with enhanced design
        st.markdown("### 🧭 Navigation")
        
        pages = {
            "🏠 Home": "Home",
            "🔮 Prediction": "Prediction", 
            "ℹ️ About": "About"
        }
        
        for page_name, page_key in pages.items():
            if st.button(
                page_name,
                key=page_key,
                use_container_width=True,
                type="primary" if st.session_state.current_page == page_key else "secondary"
            ):
                st.session_state.current_page = page_key
                st.rerun()
        
        st.markdown("---")
        
        # Quick Links
        st.markdown("### ⚡ Quick Access")
        st.markdown("""
        <div class="metric-card">
            <p>🏗️ <strong>Building Types:</strong></p>
            <span class="ai-badge">Residential</span>
            <span class="ai-badge">Apartment</span>
            <span class="ai-badge">Commercial</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.info("""
        **💡 Professional Tool**
        Designed for architects, builders, and construction professionals
        """)
    
    # Page Router
    if st.session_state.current_page == "Home":
        show_home_page()
    elif st.session_state.current_page == "Prediction":
        show_prediction_page()
    else:
        show_about_page()

if __name__ == "__main__":
    main()