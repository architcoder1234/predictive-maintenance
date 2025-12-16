# app.py


# D02-S01 Predictive Maintenance with Explainable AI 

import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error





# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Predictive Maintenance AI", layout="wide")
st.title("🔧 Predictive Maintenance using Explainable AI")
st.write("AI system to predict machine failure, remaining life, and explain the reasons.")







# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    
    
    
    return pd.read_csv("predictive_maintenance_data500.csv")

#   ^
#   |   i use my generated data file here according to my new name 




df = load_data()





#      
df = df.reset_index(drop=True)

#  ^
#  |






features = ["temperature", "vibration", "pressure", "rpm"]
X = df[features]
y_rul = df["RUL"]









# -----------------------------
# TRAIN MODELS (CACHED)
# -----------------------------
@st.cache_resource
def train_models(X, y_rul):
    
    
    
    # Anomaly Detection Model
    anomaly_model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    anomaly_model.fit(X)




    # RUL Prediction Model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_rul, test_size=0.2, random_state=42
    )
    
    rul_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rul_model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, rul_model.predict(X_test))
    return anomaly_model, rul_model, mae, X_test

anomaly_model, rul_model, mae, X_test = train_models(X, y_rul)

st.sidebar.success(f"Model MAE (RUL): {mae:.2f} days")





# -----------------------------
# MACHINE SELECTION
# -----------------------------
st.sidebar.header("🔍 Select Machine Reading")






#             

row_id = st.sidebar.slider(
    "Select time index",
    min_value=0,
    max_value=len(df) - 1,
    value=min(10, len(df) - 1)
)

sample = X.iloc[row_id:row_id+1]


#    ^
#    |        


#@@@@
st.write("Total rows in data:", len(df))
st.write("Selected row:", row_id)
#@@@


st.subheader("📌 Selected Sensor Values")
st.write(sample)







# -----------------------------
# AI PREDICTIONS
# -----------------------------
anomaly_pred = anomaly_model.predict(sample)
rul_pred = rul_model.predict(sample)[0]

st.subheader("🤖 AI Predictions")

if anomaly_pred[0] == -1:
    st.error("⚠️ Anomaly Detected! Abnormal machine behaviour.")
else:
    st.success("✅ Machine operating normally.")

st.metric("⏳ Predicted Remaining Useful Life (Days)", int(rul_pred))







# -----------------------------
# MACHINE HEALTH STATUS
# -----------------------------
st.subheader("🩺 Machine Health Status")
health = max(0, min(100, int(rul_pred)))
st.progress(health / 100)

if health < 30:
    st.warning("High risk! Immediate maintenance recommended.")
elif health < 60:
    st.info("Moderate risk. Monitor closely.")
else:
    st.success("Healthy machine.")









# -----------------------------
# EXPLAINABLE AI (SHAP)
# -----------------------------
st.subheader("🔍 Explainable AI – Why this prediction?")
explainer = shap.TreeExplainer(rul_model)
shap_values = explainer.shap_values(sample)

fig, ax = plt.subplots()
shap.bar_plot(shap_values[0], feature_names=features)
st.pyplot(fig)

st.caption("SHAP values show how each sensor contributes to reduced machine life.")










# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("**D02-S01 | Predictive Maintenance with Explainable AI**")
