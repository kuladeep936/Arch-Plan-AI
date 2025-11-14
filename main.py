import pandas as pd
import numpy as np
import torch
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import re
from pathlib import Path
import joblib
import sys
import shutil
from sklearn.neighbors import NearestNeighbors

# NEW: Plotly for visualizations
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

print("=== Training Improved ArchPlan AI Model ===")

# Configuration
CSV_PATH = 'data/House Plans Metadata.csv'
IMAGE_DIR = 'data/House Plans/'
MODEL_SAVE_DIR = 'models/'
VIZ_DIR = os.path.join(MODEL_SAVE_DIR, "visualizations")

# Create directories
Path(MODEL_SAVE_DIR).mkdir(exist_ok=True)
Path(VIZ_DIR).mkdir(exist_ok=True)

def _safe_write(fig, name_no_ext: str):
    """
    Save plotly figure as HTML (always) and PNG if kaleido is available.
    """
    html_path = os.path.join(VIZ_DIR, f"{name_no_ext}.html")
    fig.write_html(html_path, include_plotlyjs="cdn")
    try:
        png_path = os.path.join(VIZ_DIR, f"{name_no_ext}.png")
        fig.write_image(png_path, scale=2)  # requires kaleido
    except Exception:
        pass  # PNG optional

def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    print("Loading and preprocessing data...")
    
    df = pd.read_csv(CSV_PATH, encoding='latin-1')
    
    # Clean data
    df.columns = df.columns.str.strip()
    df['Plot Area'] = df['Plot Area'].str.replace(' sq. ft.', '', regex=False).astype(float)
    df['Total Built-up Area'] = df['Total Built-up Area'].str.replace(' sq. ft.', '', regex=False).astype(float)
    df['Width'] = df['Width'].str.replace(' ft.', '', regex=False).astype(float)
    df['Length'] = df['Length'].str.replace(' ft.', '', regex=False).astype(float)
    
    # Parse costs
    def parse_cost(cost_str):
        cost_str = str(cost_str).strip().lower()
        if 'crores' in cost_str or 'crore' in cost_str:
            numbers = re.findall(r'[\d.]+', cost_str)
            return float(numbers[0]) * 100 if numbers else 0
        elif 'lakhs' in cost_str or 'lakh' in cost_str:
            numbers = re.findall(r'[\d.]+', cost_str)
            return float(numbers[0]) if numbers else 0
        numbers = re.findall(r'[\d.]+', cost_str)
        return float(numbers[0]) if numbers else 0
    
    df['Estimated Cost'] = df['Estimated Cost'].apply(parse_cost)
    df = df.dropna()
    
    print(f"Processed {len(df)} samples")
    return df

def create_synthetic_features(df):
    """Create enhanced features for better prediction"""
    print("Creating enhanced features...")
    
    # Basic features
    features = df[['Plot Area', 'Total Built-up Area', 'Width', 'Length']].copy()
    
    # Enhanced features
    features['Area_Ratio'] = df['Total Built-up Area'] / df['Plot Area']
    features['Perimeter'] = 2 * (df['Width'] + df['Length'])
    features['Aspect_Ratio'] = df['Length'] / df['Width']
    features['Density'] = df['Total Built-up Area'] / (df['Width'] * df['Length'])
    
    # Price per sq ft (target engineering)
    features['Price_per_sqft'] = (df['Estimated Cost'] * 100000) / df['Total Built-up Area']
    
    return features

def create_improved_model(input_dim):
    """Create an improved neural network model"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2
    
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(64, activation='relu'),
        Dropout(0.1),
        
        Dense(32, activation='relu'),
        
        Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate accuracy within thresholds
    percentage_errors = np.abs((y_true - y_pred) / np.maximum(y_true, 1e-9))
    threshold_10 = np.mean(percentage_errors <= 0.10) * 100
    threshold_20 = np.mean(percentage_errors <= 0.20) * 100
    threshold_30 = np.mean(percentage_errors <= 0.30) * 100
    
    return {
        'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2_Score': r2,
        'Accuracy_10%': threshold_10, 'Accuracy_20%': threshold_20, 'Accuracy_30%': threshold_30
    }

# NEW: Visualization helpers
def plot_training_curves(history):
    """Plot Loss and MAE over epochs."""
    hist_df = pd.DataFrame(history.history)
    hist_df['epoch'] = np.arange(1, len(hist_df) + 1)

    # Loss curve
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=hist_df['epoch'], y=hist_df['loss'], mode='lines', name='Train Loss'))
    if 'val_loss' in hist_df:
        fig_loss.add_trace(go.Scatter(x=hist_df['epoch'], y=hist_df['val_loss'], mode='lines', name='Val Loss'))
    fig_loss.update_layout(title="Training History: Loss", xaxis_title="Epoch", yaxis_title="MSE Loss")
    _safe_write(fig_loss, "01_training_loss")

    # MAE curve
    if 'mae' in hist_df.columns:
        fig_mae = go.Figure()
        fig_mae.add_trace(go.Scatter(x=hist_df['epoch'], y=hist_df['mae'], mode='lines', name='Train MAE'))
        if 'val_mae' in hist_df:
            fig_mae.add_trace(go.Scatter(x=hist_df['epoch'], y=hist_df['val_mae'], mode='lines', name='Val MAE'))
        fig_mae.update_layout(title="Training History: MAE", xaxis_title="Epoch", yaxis_title="MAE")
        _safe_write(fig_mae, "02_training_mae")

def plot_parity_and_residuals(y_true_lakhs, y_pred_lakhs, X_test_df):
    """Parity plot (y_true vs y_pred) and residuals analysis."""
    dfp = pd.DataFrame({
        'Actual (Lakhs)': y_true_lakhs,
        'Predicted (Lakhs)': y_pred_lakhs,
        'Built_up_area': X_test_df['Total Built-up Area'].values
    })
    dfp['Residual (Lakhs)'] = dfp['Actual (Lakhs)'] - dfp['Predicted (Lakhs)']

    # Parity plot
    fig_parity = px.scatter(dfp, x='Actual (Lakhs)', y='Predicted (Lakhs)', hover_data=['Built_up_area'])
    # 45-degree line
    minv = float(min(dfp['Actual (Lakhs)'].min(), dfp['Predicted (Lakhs)'].min()))
    maxv = float(max(dfp['Actual (Lakhs)'].max(), dfp['Predicted (Lakhs)'].max()))
    fig_parity.add_trace(go.Scatter(x=[minv, maxv], y=[minv, maxv], mode='lines', name='Ideal', line=dict(dash='dash')))
    fig_parity.update_layout(title="Predicted vs Actual (Parity Plot)")
    _safe_write(fig_parity, "03_parity_plot")

    # Residuals histogram
    fig_resid = px.histogram(dfp, x='Residual (Lakhs)', nbins=30, marginal='box', opacity=0.85)
    fig_resid.update_layout(title="Residuals Distribution (Lakhs)")
    _safe_write(fig_resid, "04_residuals_histogram")

    # Error vs Built-up Area
    fig_err_size = px.scatter(dfp, x='Built_up_area', y='Residual (Lakhs)', trendline='ols')
    fig_err_size.update_layout(title="Residuals vs Built-up Area")
    _safe_write(fig_err_size, "05_residuals_vs_builtup")

def plot_feature_correlations(features_df):
    """Correlation heatmap among features + target (Price_per_sqft)."""
    corr = features_df.corr(numeric_only=True)
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect='auto',
        title="Feature Correlation Heatmap (includes Price_per_sqft)"
    )
    _safe_write(fig_corr, "00_feature_correlation_heatmap")

def main():
    try:
        # Load data
        df = load_and_preprocess_data()
        
        if len(df) < 10:
            print("ERROR: Not enough data for training!")
            return
        
        # Create enhanced features
        features = create_synthetic_features(df)

        # NEW: Correlation heatmap BEFORE split (includes target)
        plot_feature_correlations(features)

        X = features.drop('Price_per_sqft', axis=1)  # Remove target from features
        y = features['Price_per_sqft']  # Predict price per sq ft
        
        # Split data before scaling so we can keep DataFrame indices and columns for later
        X_train_df, X_test_df, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features (fit on train, transform both)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_df)
        X_test = scaler.transform(X_test_df)

        # Create and train model
        model = create_improved_model(X_train.shape[1])

        print("Training improved model...")
        # OPTIONAL: EarlyStopping / ReduceLROnPlateau (good practice)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, verbose=1)

        history = model.fit(
            X_train, y_train,
            epochs=int(os.getenv('EPOCHS', '200')),
            batch_size=16,
            validation_split=0.2,
            verbose=1,
            callbacks=[early_stop, reduce_lr]
        )

        # Visualize training curves
        plot_training_curves(history)

        # Evaluate model
        y_pred = model.predict(X_test).flatten()

        # Convert back to total cost using the unscaled test DataFrame's Total Built-up Area
        # y_test is a pandas Series (aligned with X_test_df index)
        y_test_total = (y_test * X_test_df['Total Built-up Area']) / 100000
        y_pred_total = (y_pred * X_test_df['Total Built-up Area']) / 100000
        
        metrics = calculate_metrics(y_test_total, y_pred_total)
        
        print("\n" + "="*60)
        print("📊 IMPROVED MODEL PERFORMANCE")
        print("="*60)
        print(f"Mean Absolute Error: {metrics['MAE']:.2f} Lakhs")
        print(f"R² Score: {metrics['R2_Score']:.4f}")
        print(f"Accuracy within 10%: {metrics['Accuracy_10%']:.2f}%")
        print(f"Accuracy within 20%: {metrics['Accuracy_20%']:.2f}%")
        print(f"Accuracy within 30%: {metrics['Accuracy_30%']:.2f}%")
        print("="*60)
        
        # Save model and artifacts
        model.save('models/improved_house_model.h5')
        joblib.dump(scaler, 'models/improved_scaler.pkl')
        
        # Save feature columns for reference
        feature_columns = list(X.columns)
        joblib.dump(feature_columns, 'models/feature_columns.pkl')
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv('models/improved_metrics.csv', index=False)

        # NEW: Save predictions vs actuals for audit
        audit_df = pd.DataFrame({
            'Actual_Total_Cost_Lakhs': y_test_total.values,
            'Pred_Total_Cost_Lakhs': y_pred_total,
            'Built_up_area': X_test_df['Total Built-up Area'].values
        })
        audit_df.to_csv(os.path.join(MODEL_SAVE_DIR, "predictions_vs_actuals.csv"), index=False)

        # NEW: Visualizations for predictions/residuals
        plot_parity_and_residuals(y_test_total.values, y_pred_total, X_test_df)

        print("✅ Improved model training completed!")
        print("📁 Visualizations saved in:", VIZ_DIR)
        print("   Open the HTML files in your browser to view the charts.")
        print("Next: Run the updated app.py to use the improved model")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
