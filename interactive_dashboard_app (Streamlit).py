
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from math import log

st.set_page_config(layout="wide")

# === Load Data ===
@st.cache_data
def load_data():
    return pd.read_excel("data.xlsx")

df = load_data()

# === User Inputs ===
st.sidebar.title("Controls")
experiment_options = df['Experiment'].dropna().unique()
selected_experiment = st.sidebar.selectbox("Choose Experiment", experiment_options)

model_type = st.sidebar.selectbox("Model Type", ["Main Effects", "Main + 2FI (Mock)"])

x_axis = st.sidebar.selectbox("X Variable", ["Glucose_%", "Methionine_mM", "Ethanol_%", "pH", "YNB_x"])
y_axis = st.sidebar.selectbox("Y Variable", ["pH", "Methionine_mM", "Ethanol_%", "Glucose_%", "YNB_x"])

response = st.sidebar.selectbox("Response", ["Calculated OD600 to Nanodrop", "Residual glucose (mg/l)"])

# === Subset and clean ===
df_sub = df[df['Experiment'] == selected_experiment].copy()
df_sub = df_sub[[x_axis, y_axis, response]].dropna()

if df_sub.shape[0] < 10:
    st.warning("Not enough data to build a surface for selected parameters.")
    st.stop()

# === Model Fit ===
X = df_sub[[x_axis, y_axis]]
y = df_sub[response]

model = LinearRegression().fit(X, y)

x_range = np.linspace(X[x_axis].min(), X[x_axis].max(), 30)
y_range = np.linspace(X[y_axis].min(), X[y_axis].max(), 30)
x_grid, y_grid = np.meshgrid(x_range, y_range)

z_pred = model.intercept_ + model.coef_[0] * x_grid + model.coef_[1] * y_grid

# === Plotly Surface ===
fig = go.Figure(data=[go.Surface(
    x=x_grid,
    y=y_grid,
    z=z_pred,
    colorscale="Viridis",
    showscale=True
)])

fig.update_layout(
    title=f"Predicted {response} Surface for {selected_experiment}",
    scene=dict(
        xaxis_title=x_axis,
        yaxis_title=y_axis,
        zaxis_title=response
    ),
    height=800
)

# === Display ===
st.plotly_chart(fig, use_container_width=True)

# === Raw Table ===
with st.expander("Show Raw Data"):
    st.dataframe(df_sub)

# === Coefficients ===
with st.expander("Model Coefficients"):
    st.write("Intercept:", model.intercept_)
    st.write("Coefficients:", dict(zip([x_axis, y_axis], model.coef_)))
