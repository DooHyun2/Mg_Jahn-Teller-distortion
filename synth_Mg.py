import numpy as np
import pandas as pd

# 1. Setup
rng = np.random.default_rng(42)
n = 1500

# 2. Input Features
# 0:Pure(Mn), 1:Zn, 2:Al, 3:Co
dopant_code = rng.integers(0, 4, n)

# Doping amount (0.01 ~ 0.50). Pure(0) has 0 amount.
raw_amt = rng.uniform(0.01, 0.5, n)
dopant_amount = np.where(dopant_code == 0, 0.0, raw_amt)

sinter_temp = rng.uniform(700, 1000, n)  # Temp (C)

# 3. Physics Logic: Jahn-Teller Distortion (jt Distortion)
# Zn: Suppresses Jahn-Teller distortion effectively.
# Al: Light and strong
# Co: High voltage
suppression_factor = np.select(
    [dopant_code == 1, dopant_code == 2, dopant_code == 3],
    [0.5, 0.3, 0.2],  # Weight: Zn > Al > Co
    default=0.0
)

# Higher doping amount = Lower distortion
base_dist = rng.normal(0.8, 0.1, n)
jt_distortion = base_dist - (suppression_factor * dopant_amount * 1.5)
jt_distortion = np.clip(jt_distortion, 0.05, 1.0)

# 4. Target: Voltage Prediction
# Co: Enables high voltage.
# Al: Moderate voltage boost.
vol_boost = np.select(
    [dopant_code == 3, dopant_code == 2],
    [0.8, 0.3],       # Boost: Co > Al
    default=0.0
)

noise = rng.normal(0, 0.05, n)

# Logic: High distortion reduces voltage (Penalty from Jahn-Teller).
voltage = (
    2.9
    + (vol_boost * dopant_amount)
    - (jt_distortion * 0.4
    + (sinter_temp - 700) * 5e-4
    + noise ))
voltage = np.clip(voltage, 2.0, 4.8)

# 5. Save Data
# Mapping for visualization
dopant_labels = np.array(['Pure_Mn', 'Zn_doped', 'Al_doped', 'Co_doped'])[dopant_code]

df = pd.DataFrame({
    "dopant_code": dopant_code,
    "dopant_label": dopant_labels,
    "dopant_amount": dopant_amount,
    "sinter_temp": sinter_temp,
    "JT_distortion": jt_distortion,
    "voltage": voltage
})

df.to_csv("mg_spinel_synth_data.csv", index=False)
print("Saved: mg_spinel_synth_data.csv", df.shape)
