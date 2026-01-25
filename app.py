import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Shopper Spectrum")
st.subheader("Customer Segmentation & Product Recommendation System")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("cleaned_data.csv")
    return data

final_data = load_data()

# -----------------------------
# PRODUCT RECOMMENDATION SETUP
# -----------------------------
@st.cache_data
def build_recommendation_system(data):
    customer_product_matrix = data.pivot_table(
        index='CustomerID',
        columns='StockCode',
        values='Quantity',
        aggfunc='sum'
    ).fillna(0)

    similarity = cosine_similarity(customer_product_matrix.T)

    similarity_df = pd.DataFrame(
        similarity,
        index=customer_product_matrix.columns,
        columns=customer_product_matrix.columns
    )

    product_lookup = (
        data[['StockCode', 'Description']]
        .drop_duplicates()
        .set_index('StockCode')
    )

    return similarity_df, product_lookup

product_similarity_df, product_lookup = build_recommendation_system(final_data)

def recommend_products(product_name, top_n=5):
    matches = product_lookup[
        product_lookup['Description'].str.contains(product_name, case=False, na=False)
    ]

    if matches.empty:
        return []

    stock_code = matches.index[0]
    scores = product_similarity_df[stock_code].sort_values(ascending=False).iloc[1:]

    recommendations = product_lookup.loc[scores.index]['Description']
    recommendations = recommendations.drop_duplicates().head(top_n)

    return recommendations.tolist()

# -----------------------------
# LOAD CLUSTERING MODELS
# -----------------------------
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("rfm_scaler.pkl")

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Select Module",
    ["Product Recommendation", "Customer Segmentation"]
)

# ======================================================
# MODULE 1 – PRODUCT RECOMMENDATION
# ======================================================
if option == "Product Recommendation":

    st.header("🎯 Product Recommendation")

    product_input = st.text_input(
        "Enter Product Name (keyword)",
        placeholder="e.g. HEART, GIN, METAL SIGN"
    )

    if st.button("Get Recommendations"):
        recommendations = recommend_products(product_input)

        if len(recommendations) == 0:
            st.warning("No matching product found.")
        else:
            st.success("Recommended Products:")
            for i, product in enumerate(recommendations, 1):
                st.write(f"{i}. {product}")

# ======================================================
# MODULE 2 – CUSTOMER SEGMENTATION
# ======================================================
if option == "Customer Segmentation":

    st.header("👥 Customer Segmentation")

    recency = st.number_input("Recency (days since last purchase)", min_value=0)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0)

    if st.button("Predict Segment"):

        # Log transform
        r = np.log1p(recency)
        f = np.log1p(frequency)
        m = np.log1p(monetary)

        scaled_input = scaler.transform([[r, f, m]])
        cluster = kmeans.predict(scaled_input)[0]

        # Correct cluster-to-segment mapping
        cluster_map = {
            0: "Occasional Customer",
            1: "High-Value Customer",
            2: "Regular Customer",
            3: "At-Risk Customer"
        }
        segment = cluster_map[cluster]
        st.success(f"Predicted Segment: **{segment}**")




