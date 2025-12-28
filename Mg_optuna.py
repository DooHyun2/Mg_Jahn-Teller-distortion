import optuna, pandas as pd, json, os
from sklearn.ensemble import RandomForestRegressor

# 1 Setup & Train Surrogate Models
os.makedirs("results", exist_ok=True) # Making results folder
df = pd.read_csv("mg_spinel_synth_data.csv")

# X: Features, y: Targets (Voltage & Distortion)
X = df.drop(columns=["voltage", "JT_distortion", "dopant_label"])
print("Training surrogate models (Voltage & Distortion)...")

# Train two separate brains: one for Performance (V), one for Safety (D)
rf_vol = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42).fit(X, df["voltage"])
rf_dist = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42).fit(X, df["JT_distortion"])

# 2 Objective Function
def objective(trial):
    # Search Space
    d_code = trial.suggest_categorical("dopant_code", [0, 1, 2, 3]) # 0:Mn, 1:Zn, 2:Al, 3:Co
    d_amt = trial.suggest_float("dopant_amount", 0.0, 0.5)
    temp = trial.suggest_float("sinter_temp", 700, 1000)

    # Predict
    in_df = pd.DataFrame([[d_code, d_amt, temp]], columns=X.columns)
    pred_d = rf_dist.predict(in_df)[0]
    pred_v = rf_vol.predict(in_df)[0]

    # Constraint: Penalize if distortion is too high (Safety first)
    if pred_d > 0.6:
        return -1.0

    return pred_v

# 3 Run Optimization
print("Optimizing...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=500)

# 4 Process & Save Results
best = study.best_params
code_map = {0: 'Pure Mn', 1: 'Zn (Stability)', 2: 'Al (Balance)', 3: 'Co (Voltage)'}

# Add metadata for clarity
best["material"] = code_map[best["dopant_code"]]
best["predicted_voltage"] = study.best_value
best["predicted_distortion"] = rf_dist.predict(
    pd.DataFrame([[best["dopant_code"], best["dopant_amount"], best["sinter_temp"]]], columns=X.columns)
)[0]

# Save to JSON & TXT
with open("results/bo_best_params.json", "w") as f: json.dump(best, f, indent=4)
with open("results/bo_best_value.txt", "w") as f: f.write(str(study.best_value))

print(f"Done. Saved to results/bo_best_params.json")
print(f" Best: {best['material']} | Vol: {best['predicted_voltage']:.4f}V | Dist: {best['predicted_distortion']:.4f}")
