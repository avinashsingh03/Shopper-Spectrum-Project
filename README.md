# 🛒 Shopper Spectrum: Customer Segmentation & Product Recommendation System

An end-to-end **Data Science & Machine Learning project** that analyzes e-commerce transaction data to segment customers using **RFM analysis + KMeans clustering** and recommend products using **Item-based Collaborative Filtering**. The project concludes with an interactive **Streamlit web application** for real-time predictions.

Link: https://shopper-spectrum-project.streamlit.app/
---

## Project Overview

The global e-commerce industry generates massive volumes of transactional data every day. Proper analysis of this data enables businesses to understand customer behavior, design targeted marketing strategies, improve retention, and deliver personalized product recommendations.

This project focuses on:

* Understanding customer purchasing behavior
* Segmenting customers based on **Recency, Frequency, and Monetary (RFM)** metrics
* Building a **product recommendation system** based on customer co-purchase patterns
* Deploying both systems through a **Streamlit web app**

---

## Business Objectives

* Identify **High-Value, Regular, Occasional, and At-Risk customers**
* Enable **targeted marketing campaigns**
* Improve **customer retention strategies**
* Provide **personalized product recommendations**
* Support **inventory planning** using product demand insights

---

## Problem Type

* **Unsupervised Machine Learning** – Customer Segmentation (Clustering)
* **Collaborative Filtering** – Product Recommendation System

---

## Dataset Description

The dataset contains transactional data from an online retail store.

| Column      | Description                  |
| ----------- | ---------------------------- |
| InvoiceNo   | Transaction number           |
| StockCode   | Unique product code (SKU)    |
| Description | Product name                 |
| Quantity    | Number of units purchased    |
| InvoiceDate | Date and time of transaction |
| UnitPrice   | Price per unit               |
| CustomerID  | Unique customer identifier   |
| Country     | Customer country             |

After cleaning, the final dataset contains **392,692 transactions** with no missing values.

---

## Project Workflow

### 1️. Data Cleaning & Preprocessing

* Removed missing CustomerIDs
* Removed cancelled invoices
* Filtered out negative or zero quantities and prices
* Converted dates to proper datetime format
* Created a `TotalPrice` feature

### 2️. Exploratory Data Analysis (EDA)

EDA was conducted using the **UBM rule**:

* **Univariate Analysis** – distributions, bar charts, histograms
* **Bivariate Analysis** – scatter plots, box plots
* **Multivariate Analysis** – heatmaps, pair plots

**Key Insights:**

* UK contributes ~88% of total revenue
* Sales peak between October and November
* A small set of products generates a large share of revenue
* Customer behavior is highly skewed

### 3️. RFM Feature Engineering

For each customer:

* **Recency** = Days since last purchase
* **Frequency** = Number of transactions
* **Monetary** = Total spend

A customer-level RFM dataset with **4,338 unique customers** was created.

### 4️. Customer Segmentation (Clustering)

* Applied log transformation and standard scaling
* Used **KMeans clustering**
* Optimal clusters selected using **Elbow Method & Silhouette Score**
* Chosen number of clusters: **4** (for business interpretability)

#### Customer Segments

| Segment    | Description                        |
| ---------- | ---------------------------------- |
| High-Value | Recent, frequent, high spenders    |
| Regular    | Moderate frequency and spending    |
| Occasional | Infrequent, low spending customers |
| At-Risk    | Long inactive customers            |

> Note: Cluster labels were mapped to business segments based on **RFM mean interpretation**, not raw cluster IDs.

---

## Product Recommendation System

* **Approach:** Item-based Collaborative Filtering
* **Similarity Metric:** Cosine Similarity
* **Input:** Product name (keyword)
* **Output:** Top 5 similar products

The system recommends products based on **customer co-purchase behavior**, not textual similarity. Hence, some recommended items may look similar but represent distinct SKUs.

---

## Tech Stack

* **Programming:** Python
* **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
* **Machine Learning:** KMeans, Cosine Similarity
* **Deployment:** Streamlit
* **Version Control:** Git, GitHub

---

## Key Learnings

* Practical RFM-based customer segmentation
* Importance of business-driven cluster interpretation
* Collaborative filtering using real purchase data
* Building and deploying ML apps with Streamlit

---

## Future Improvements

* NLP-based product name normalization
* Hybrid recommendation system
* User-based collaborative filtering
* Advanced customer lifetime value (CLV) modeling

---

## 👤 Author
```
**Avinash Singh**
B.Tech CSE (AI) | Data Science & Machine Learning Enthusiast
```
