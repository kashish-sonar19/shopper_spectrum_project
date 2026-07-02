#  Shopper Spectrum: Retail Analytics SaaS

![Python](https://img.shields.io/badge/Python-3.14-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg)
![Status](https://img.shields.io/badge/Status-Live-success.svg)

**Shopper Spectrum** is an end-to-end Data Analytics and Unsupervised Machine Learning dashboard designed to empower e-commerce businesses with actionable customer intelligence and product discovery features. 

 **https://shopperspectrumproject19.streamlit.app/**

---

##  Core Features

### 1.  Discovery Engine (Product Recommendations)
*   **Algorithm:** Item-Based Collaborative Filtering using Cosine Similarity.
*   **Functionality:** Generates real-time "Frequently Bought Together" recommendations based on historical customer purchase patterns.
*   **Business Impact:** Increases Average Order Value (AOV) by optimizing cross-selling strategies.

### 2.  RFM Profiling Matrix (Customer Segmentation)
*   **Algorithm:** K-Means Clustering applied to RFM (Recency, Frequency, Monetary) vectors.
*   **Functionality:** Classifies users into actionable segments:
    *    **VIP/Champion:** High value, frequent shoppers.
    *    **Loyal Customer:** Consistent engagement.
    *    **Regular Customer:** Standard retention protocols apply.
    *    **At-Risk Customer:** High churn probability requiring win-back campaigns.
*   **Business Impact:** Enables highly targeted marketing and personalized retention campaigns.

---

##  System Architecture & Tech Stack

*   **Frontend UI:** Streamlit (Custom CSS, Responsive Grid Layouts)
*   **Machine Learning:** Scikit-Learn (K-Means, StandardScaler, Cosine Similarity)
*   **Data Processing:** Pandas, NumPy
*   **Deployment:** Streamlit Community Cloud (with Git LFS for model tracking)

---
