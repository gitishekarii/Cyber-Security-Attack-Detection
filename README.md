# Cyber Security Attack Type Detection

An end-to-end machine learning pipeline that classifies network traffic into cyber-attack types with **92.5% accuracy**.  
Built as part of an applied MSc in Data Science & AI at DSTI.

---

## 🚀 Features

### Data Preprocessing & Cleaning
- Handle missing values, normalize protocols, standardize formats

### Feature Engineering & Selection
- Extract and select top-10 predictors via statistical tests and `SelectKBest`

### Model Training & Evaluation
- Compare Logistic Regression, Random Forest & XGBoost  
- Evaluate with accuracy, precision & recall

### Deployment
- Flask web app for live predictions (manual input or CSV upload)  
- Interactive Power BI dashboard for real-time monitoring

### Performance
- Tuned XGBoost model: **92.5% accuracy**  
- Automated ETL scripts: **30% reduction** in processing time

---

## 📁 Repository Structure

```bash
cyber-attack-detection/
│
├── data/  
│   ├── raw/                  # Original network logs (CSV)  
│   └── processed/            # Cleaned & feature-engineered data  
│
├── notebooks/                # Jupyter notebooks for EDA & modeling  
│   ├── 01_eda.ipynb  
│   └── 02_modeling.ipynb  
│
├── src/  
│   ├── data_preprocessing.py # Data cleaning & ETL  
│   ├── feature_engineering.py# Feature creation & selection  
│   ├── train_models.py       # Model training & evaluation  
│   └── app.py                # Flask application  
│
├── dashboards/               # Power BI files  
│   └── Attack_Dashboard.pbix  
│
├── requirements.txt          # Python dependencies  
├── README.md                 # Project overview & instructions  
└── .gitignore                # Files/folders to ignore  
