# test_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np

st.title("🔹 Simple Streamlit Test Dashboard")

# Sidebar slider
st.sidebar.header("Controls")
num_points = st.sidebar.slider("Number of points", min_value=10, max_value=100, value=50)

# Generate random data
data = pd.DataFrame({
    "x": range(num_points),
    "y": np.random.randn(num_points).cumsum()
})

# Display chart
st.line_chart(data.set_index("x"))

# Display raw data
st.subheader("Data Table")
st.dataframe(data)
