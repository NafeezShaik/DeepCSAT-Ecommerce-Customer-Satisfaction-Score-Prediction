# 🛒 DeepCSAT - E-Commerce Customer Satisfaction Score Prediction (Complete ML Suite)

<img width="296" height="170" alt="E-Commerce CSAT Prediction" src="https://github.com/user-attachments/assets/e4aecc53-7d6a-46ba-8f9b-cd5f8db5f1dd" />

## 📌 Project Summary

### 🔍 Overview

This project implements a **comprehensive Customer Satisfaction (CSAT) score prediction system** using both **Traditional Machine Learning** and **Deep Learning** approaches. 

In the e-commerce industry, understanding customer satisfaction through interactions and feedback is crucial for:
* Improving service quality
* Boosting customer retention  
* Driving business growth
* Proactive service recovery

By leveraging **7 different ML models**, this project provides accurate CSAT score predictions that enable **real-time insights** for business action.

---

### 📖 Project Background

Customer satisfaction is a key metric driving **loyalty, repeat purchases, and referrals**. Traditionally measured via surveys, which:
* Take time to collect
* Capture only a portion of customer experience  
* Provide delayed insights

With this ML system, companies can now **predict satisfaction scores dynamically**, helping identify weak points and optimize service delivery **instantly**.

---

## 📂 Dataset Overview

**Source:** E-Commerce Customer Support Data (Shopzilla - 1 month)
* **Records:** 85,907 support interactions
* **Features:** 20 columns
* **Target:** CSAT Score (1, 2, 3, 4, 5)

**Key Features:**
* **Unique ID** – identifier for each record
* **Channel Name** – service channel (Outcall, Inbound, Chat)
* **Category / Sub-category** – issue types (12 categories, 57 sub-categories)
* **Customer Remarks** – feedback text
* **Order Details** – Order ID, date/time
* **Timestamps** – Issue reported, responded, survey completion
* **Customer Info** – City (1,782 cities)
* **Product Data** – Category (9), Item Price
* **Agent Info** – Name (1,371 agents), Supervisor (40), Manager (6)
* **Agent Attributes** – Tenure Bucket (5 levels), Shift (5 timings)
* **Response Metrics** – Handling time, response time
* **CSAT Score** – Target variable (1-5 scale)

---

## 🎯 Project Goal

**Primary Objective:** Predict CSAT scores using customer interaction data to enable:
* **Proactive service improvement**
* **Real-time satisfaction monitoring**  
* **Actionable insights** for management
* **Data-driven resource allocation**

**Success Criteria:** 70-80% prediction accuracy

---

## 🛠️ Tech Stack

* **Programming Language:** Python 3.8+ 🐍
* **Traditional ML:** scikit-learn, XGBoost, LightGBM
* **Deep Learning:** TensorFlow / Keras
* **Data Analysis:** Pandas, NumPy
* **Statistical Analysis:** SciPy, statsmodels
* **Visualization:** Matplotlib, Seaborn, Plotly
* **Handling Imbalance:** imbalanced-learn (SMOTE)
* **Frontend:** Streamlit
* **Model Persistence:** Joblib, H5 Format
* **Version Control:** Git & GitHub

---

## 🧠 Models Implemented

### Traditional Machine Learning (5 Models):
1. **Logistic Regression** (Baseline) – ~67% accuracy
2. **Decision Tree** – ~70% accuracy
3. **Random Forest** – ~74% accuracy
4. **XGBoost** – ~76% accuracy ⭐
5. **LightGBM** – ~75% accuracy

### Deep Learning (2 Models):
6. **Baseline ANN** – ~73% accuracy
7. **Optimized ANN** (GridSearch + CV) – ~76% accuracy ⭐

**Best Models:** XGBoost & Optimized ANN achieve **~76% accuracy**

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 67% | 0.66 | 0.67 | 0.66 |
| Decision Tree | 70% | 0.69 | 0.70 | 0.69 |
| Random Forest | 74% | 0.73 | 0.74 | 0.73 |
| **XGBoost** | **76%** | **0.75** | **0.76** | **0.75** |
| LightGBM | 75% | 0.74 | 0.75 | 0.74 |
| Baseline ANN | 73% | 0.72 | 0.73 | 0.72 |
| **Optimized ANN** | **76%** | **0.75** | **0.76** | **0.75** |

### Per-Class Performance (Optimized ANN):
* CSAT=1 → Precision 0.86, Recall 0.89
* CSAT=2 → Precision 0.81, Recall 0.77
* CSAT=3 → Precision 0.64, Recall 0.61
* CSAT=4 → Precision 0.65, Recall 0.93
* CSAT=5 → Precision 0.78, Recall 0.50

✅ **Insight:** Models perform well overall, especially in distinguishing satisfied customers (scores 1, 2, 4). Neutral class (3) is harder due to overlapping patterns.

---

## 📈 Key Features of Analysis

### 1. Comprehensive EDA (15+ Visualizations)
* CSAT score distribution (pie chart)
* Channel vs satisfaction analysis
* Agent performance metrics (shift timing, tenure)
* Response time impact on CSAT
* Top performers (managers, agents, supervisors, cities, products)
* Correlation heatmaps and pair plots

### 2. Advanced Feature Engineering
* **Text Processing:** TF-IDF vectorization of Customer Remarks
* **Categorical Encoding:**
  - One-Hot: Low cardinality features (channel, shift)
  - Label Encoding: High cardinality (agent names, supervisors)
  - Frequency Encoding: Customer_City (1,782 values)
* **Time Features:** Response time, day of week, hour extraction
* **Feature Selection:** VIF analysis, correlation-based removal

### 3. Statistical Rigor
* **Hypothesis Testing:** 3 Chi-Square tests for categorical relationships
* **Outlier Treatment:** IQR method + Mean±3σ for skewed features
* **Multicollinearity Check:** VIF < 5 enforcement
* **Class Imbalance:** SMOTE oversampling

### 4. Model Optimization
* GridSearchCV for hyperparameter tuning
* 5-Fold Stratified Cross-Validation
* Comprehensive evaluation metrics
* Feature importance analysis

---

## 🌐 Streamlit Deployment

The Streamlit app provides **two prediction modes**:

### 1. Manual Input (Sidebar)
* Enter customer interaction details manually
* Get instant CSAT prediction
* View confidence scores

### 2. CSV Batch Prediction
* Upload dataset with multiple records
* Process all predictions at once
* Download results as CSV

📸 **Screenshots Available:** See `Streamlit/` folder for interface previews

---

## 📦 How to Run

### 1️⃣ Clone/Download Repository

```bash
# Clone if using Git
git clone https://github.com/yourusername/deepcsat-prediction.git
cd deepcsat-prediction

# Or extract ZIP file
unzip DeepCSAT_Enhanced_Project.zip
cd DeepCSAT_Enhanced_Project
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Jupyter Notebook

```bash
# Open notebook
jupyter notebook Notebook/Ecommerce_CSAT_Prediction_Model_Final_Deploy_Ready.ipynb

# Run all cells to:
# - Load and preprocess data
# - Train all 7 models  
# - Generate visualizations
# - Save model artifacts
```

### 4️⃣ Launch Streamlit App

```bash
# Navigate to Streamlit folder
cd Streamlit

# Run app
streamlit run app.py

# Or use alternative app
streamlit run app1.py
```

App will open at `http://localhost:8501`

---

## 📁 Project Structure

```
DeepCSAT_Enhanced_Project/
│
├── Notebook/
│   └── Ecommerce_CSAT_Prediction_Model_Final_Deploy_Ready.ipynb  (268 cells)
│
├── Dataset/
│   └── eCommerce_Customer_support_data.csv
│
├── Streamlit/
│   ├── app.py                 # Main Streamlit app
│   ├── app1.py                # Alternative interface
│   ├── csat_model.h5          # Saved neural network
│   ├── features.pkl           # Feature list
│   ├── scaler.pkl             # StandardScaler
│   └── requirements.txt       # Streamlit dependencies
│
├── My_Streamlit/
│   ├── features.pkl           # Alternative model artifacts
│   ├── scaler (1).pkl
│   └── target_encoder.pkl
│
├── README.md                  # This file
└── requirements.txt           # Project dependencies
```

---

## 💡 Key Insights & Business Recommendations

### Factors Driving CSAT:
1. **Response Time** – Faster response = higher satisfaction (strong negative correlation)
2. **Communication Channel** – Chat has highest satisfaction, outcall lowest
3. **Agent Experience** – Higher tenure correlates with better scores
4. **Shift Timing** – Morning shift shows consistently better performance
5. **Product Category** – Electronics has lowest CSAT, needs attention

### Actionable Recommendations:
* **Invest in chat infrastructure** – Highest satisfaction channel
* **Reduce response times** – Critical driver of satisfaction  
* **Agent training programs** – Focus on new hires (low tenure)
* **Optimize shift scheduling** – More resources in morning shift
* **Electronics support enhancement** – Dedicated training/resources
* **Proactive monitoring** – Use model to flag at-risk customers in real-time

---

## 🔮 Future Enhancements

1. **Advanced NLP:**
   - Use BERT/GPT for customer remarks analysis
   - Sentiment analysis integration
   - Topic modeling for issue clustering

2. **Real-Time Deployment:**
   - REST API with FastAPI/Flask
   - Cloud deployment (AWS SageMaker, GCP AI Platform)
   - Dockerized containerization

3. **Model Improvements:**
   - Ensemble stacking of top models
   - AutoML for hyperparameter optimization
   - Time-series analysis for trend prediction

4. **Explainability:**
   - SHAP values for model interpretability
   - LIME for local explanations
   - Feature contribution dashboards

5. **Continuous Learning:**
   - A/B testing framework
   - Automated model retraining pipeline
   - Drift detection and monitoring

---

## 📊 Notebook Highlights

**Total Cells:** 268 cells (Enhanced from 240)

**Sections:**
1. Project Header with Guidelines
2. Know Your Data (comprehensive)
3. Understanding Variables
4. Data Preprocessing (99.7% missing data handled)
5. Exploratory Data Analysis (15+ charts)
6. Hypothesis Testing (3 statistical tests)
7. Outlier Detection & Treatment
8. Feature Engineering (TF-IDF, VIF, SMOTE)
9. **Traditional ML Models (5 algorithms)** ⭐ NEW
10. **Deep Learning Models (2 ANNs)** ⭐
11. **Final Model Comparison** ⭐ NEW
12. Conclusion & Recommendations

**Code Quality:**
* ✅ Well-commented (125 code cells)
* ✅ Comprehensive documentation (143 markdown cells)
* ✅ Production-ready
* ✅ Fully reproducible

---

## 📝 Deliverables Checklist

✅ Complete Jupyter Notebook (268 cells)
✅ All code well-commented
✅ 15+ visualizations with insights
✅ 7 ML models (5 traditional + 2 deep learning)
✅ Hypothesis testing (3 tests)
✅ Outlier treatment
✅ Advanced feature engineering
✅ Model comparison analysis
✅ Production-ready Streamlit app (2 versions)
✅ All model artifacts saved (.pkl, .h5)
✅ Comprehensive README documentation
✅ Requirements.txt with dependencies

---

## 👤 Author

**[Your Name]**  
Data Science & Machine Learning Enthusiast  
[LinkedIn] | [GitHub] | [Email]

---

## 📝 License

This project is for educational purposes.

**Dataset:** E-Commerce Customer Support Data (Shopzilla - 1 month sample)

---

## 🙏 Acknowledgments

* Original deep learning implementation insights
* scikit-learn, XGBoost, LightGBM communities
* TensorFlow/Keras team
* Streamlit for easy deployment

---

**⭐ If you found this project helpful, please star the repository!**

---

**Last Updated:** March 2026  
**Version:** 2.0 (Enhanced with Traditional ML Models)
