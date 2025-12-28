import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. Setup
os.makedirs("results", exist_ok=True)

# 2. Load Data
print("Loading Data...")
df = pd.read_csv("mg_spinel_synth_data.csv")

# 3. Preprocessing
X = df.drop(columns=["voltage", "dopant_label"]) 
y = df["voltage"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 4. Train Model
print("Training Random Forest...")
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# 5. SHAP Analysis
print("Calculating SHAP values...")
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# 6. Visualization (Custom Colors)
# add colors with matplotlib

def plot_custom_shap_final(feature_name, shap_vals, X_data, interaction_col, title):
    # 1. data setup
    feat_idx = list(X_data.columns).index(feature_name)
    x_val = X_data[feature_name].values        # X축 값
    y_val = shap_vals[:, feat_idx]             # Y축 값 (SHAP)
    color_val = X_data[interaction_col].values # 색상 기준 (0,1,2,3)
    
    # 2. set color (Mn=gray, Zn=green, Al=blue, Co=red)
    colors = ['#95a5a6', '#2ecc71', '#3498db', '#e74c3c'] 
    labels = ['Pure Mn (0)', 'Zn (1) Stability', 'Al (2) Balance', 'Co (3) Voltage']
    
    plt.figure(figsize=(10, 6))
    
    
    for i in range(4):
        mask = (color_val == i) 
        
        plt.scatter(
            x_val[mask], 
            y_val[mask],
            s=60,                
            alpha=0.8,           
            color=colors[i],     
            label=labels[i],     
            edgecolor='white',   
            linewidth=0.5
        )

    # 4. add colors
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.xlabel(f"{feature_name} Value", fontsize=12)
    plt.ylabel(f"SHAP Value (Impact on Voltage)", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(title="Dopant Type", fontsize=10, loc='best')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # 5. save results
    save_name = f"results/colored_{feature_name}.png"
    plt.savefig(save_name, dpi=300)
    plt.close()
    print(f"Saved Graph: {save_name}")

# run
print("Generating FINAL Colored Plots...")

# (1) dopang graph
plot_custom_shap_final(
    feature_name="dopant_amount", 
    shap_vals=shap_values, 
    X_data=X_test, 
    interaction_col="dopant_code", 
    title="Final: Doping Amount Impact (by Element)"
)

# (2) JT_distortion graph
plot_custom_shap_final(
    feature_name="JT_distortion", 
    shap_vals=shap_values, 
    X_data=X_test, 
    interaction_col="dopant_code", 
    title="Final: Jahn-Teller Distortion Impact"
)

print("Done, Check 'results' folder")