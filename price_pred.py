# save_corrected_model.py
import pandas as pd
import numpy as np
import re
import io
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

def extract_numeric_area(area_str):
    """Extract numeric value from area strings"""
    if pd.isna(area_str):
        return 0
    numbers = re.findall(r'\d+', str(area_str))
    return float(numbers[0]) if numbers else 0

def extract_numeric_dimension(dim_str):
    """Extract numeric value from dimension strings"""
    if pd.isna(dim_str):
        return 0
    numbers = re.findall(r'\d+', str(dim_str))
    return float(numbers[0]) if numbers else 0

def convert_cost_to_lakhs(cost_str):
    """Convert cost string to numeric value in lakhs"""
    if pd.isna(cost_str):
        return 0
    
    cost_str = str(cost_str).lower()
    
    # Handle crore values
    if 'crore' in cost_str or 'crores' in cost_str:
        numbers = re.findall(r'[\d.]+', cost_str)
        if len(numbers) >= 2:
            return (float(numbers[0]) + float(numbers[1])) / 2 * 100
        elif numbers:
            return float(numbers[0]) * 100
    else:
        # Lakhs values
        numbers = re.findall(r'[\d.]+', cost_str)
        if len(numbers) >= 2:
            return (float(numbers[0]) + float(numbers[1])) / 2
        elif numbers:
            return float(numbers[0])
    
    return 0

# Load and preprocess data
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

df = pd.read_csv(io.StringIO(data))

# Apply preprocessing
df['Plot_Area_num'] = df['Plot Area'].apply(extract_numeric_area)
df['Built_up_Area_num'] = df['Total Built-up Area'].apply(extract_numeric_area)
df['Width_num'] = df['Width'].apply(extract_numeric_dimension)
df['Length_num'] = df['Length'].apply(extract_numeric_dimension)
df['Price_Lakhs'] = df['Estimated Cost'].apply(convert_cost_to_lakhs)

# Calculate additional features
df['Area_Ratio'] = df['Built_up_Area_num'] / df['Plot_Area_num']
df['Perimeter'] = 2 * (df['Width_num'] + df['Length_num'])
df['Aspect_Ratio'] = df['Length_num'] / df['Width_num']
df['Footprint_Area'] = df['Width_num'] * df['Length_num']

# Encode categorical variables
le_building = LabelEncoder()
le_style = LabelEncoder()

df['Building_Type_encoded'] = le_building.fit_transform(df['Building Type'])
df['Style_encoded'] = le_style.fit_transform(df['Style'])

# Select features for modeling - simpler set for better predictions
features = ['Plot_Area_num', 'Built_up_Area_num', 'Width_num', 'Length_num', 
           'Area_Ratio', 'Perimeter', 'Building_Type_encoded', 'Style_encoded']

X = df[features]
y = df['Price_Lakhs']

print("Feature ranges:")
for feature in features:
    print(f"{feature}: {X[feature].min():.1f} to {X[feature].max():.1f}")

print(f"\nTarget range: {y.min():.1f} to {y.max():.1f} Lakhs")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.2f} Lakhs")

# Test with sample data
print("\nSample predictions:")
sample_data = [
    [2000, 1500, 40, 50, 0.75, 180, 0, 0],  # Residential Single-Storey
    [3000, 2500, 50, 60, 0.83, 220, 0, 1],   # Residential Two-Storey
    [5000, 4000, 60, 80, 0.80, 280, 2, 4]    # Commercial Supermarket
]

for i, sample in enumerate(sample_data):
    pred = model.predict([sample])[0]
    print(f"Sample {i+1}: {pred:.2f} Lakhs")

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the model and preprocessing objects
joblib.dump(model, 'models/house_price_model.pkl')
joblib.dump(le_building, 'models/building_encoder.pkl')
joblib.dump(le_style, 'models/style_encoder.pkl')
joblib.dump(features, 'models/feature_columns.pkl')

# Save the dataset for similarity search
df.to_csv('models/dataset.csv', index=False)

print(f"\nModel saved successfully!")
print(f"Model can predict prices from {y_pred.min():.2f} to {y_pred.max():.2f} Lakhs")