import pandas as pd
import streamlit as st

df = pd.read_csv("badminton_cleaned.csv")

st.title("Badminton Player Performance Analysis")
st.write("Analysis of professional badminton match data")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Dataset Statistics")
st.write(df.describe(include="all"))
