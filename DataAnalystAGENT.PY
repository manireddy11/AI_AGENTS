# app.py
# Streamlit + Ollama Mistral Agent for Data Visualization + Insight Generation

import streamlit as st
import pandas as pd
import plotly.express as px
from ollama import Client

# -----------------------------
# 1. Initialize Ollama client
# -----------------------------
@st.cache_resource(show_spinner=True)
def get_ollama_client():
    """
    Connects to the locally running Ollama server.
    Make sure 'ollama serve' is running before starting Streamlit.
    """
    return Client()

st.set_page_config(page_title="Data Insights Agent", layout="wide")
st.title("📊 Data Insights Agent (Upload → Visualize → AI Insights)")

client = get_ollama_client()

# -----------------------------
# 2. Upload Dataset
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV or Excel dataset", type=["csv", "xlsx"])
if uploaded_file is not None:
    # Read CSV or Excel
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    categorical_columns = df.select_dtypes(exclude='number').columns.tolist()

    # -----------------------------
    # 3. Visualizations
    # -----------------------------
    st.subheader("Interactive Visualizations")

    # Numeric columns trends
    if numeric_columns:
        st.markdown("*Line charts for numeric trends:*")
        for col in numeric_columns:
            fig = px.line(df, y=col, title=f"{col} Trend")
            st.plotly_chart(fig, use_container_width=True)

    # Bar charts for categorical vs numeric
    if numeric_columns and categorical_columns:
        st.markdown("*Bar charts for categorical comparisons:*")
        cat_col = st.selectbox("Select categorical column for bar charts", categorical_columns)
        num_col = st.selectbox("Select numeric column for bar charts", numeric_columns)
        fig = px.bar(df, x=cat_col, y=num_col, title=f"{num_col} by {cat_col}")
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    if len(numeric_columns) > 1:
        st.markdown("*Correlation Heatmap:*")
        corr = df[numeric_columns].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # 4. AI-driven Insights using Ollama Mistral
    # -----------------------------
    st.subheader("AI-driven Insights")
    st.markdown("Generating insights from dataset... (may take a few seconds)")

    # Prepare prompt for Ollama
    preview_csv = df.head(20).to_csv(index=False)
    prompt = f"""
You are a professional data analyst. Analyze the following dataset (first 20 rows shown) and provide:

1. Key trends and patterns
2. Top and bottom performing columns/categories
3. Correlations or interesting relationships
4. Any anomalies

Dataset preview:
{preview_csv}
"""

    # Generate insights via Ollama Mistral
    try:
        result = client.generate(
            model="mistral",
            prompt=prompt
        )

        # Some versions return {"response": "..."} others {"text": "..."}
        insights_text = (
            result.get("response")
            or result.get("text")
            or str(result)
        ).strip()

    except Exception as e:
        insights_text = f"Error generating insights: {e}"

    st.text_area("AI-generated Insights", value=insights_text, height=300)

    # -----------------------------
    # 5. Optional: Download insights
    # -----------------------------
    st.download_button(
        label="Download AI Insights",
        data=insights_text,
        file_name="ai_insights.txt",
        mime="text/plain"
    )

else:
    st.info("Please upload a CSV or Excel dataset to get started.")
