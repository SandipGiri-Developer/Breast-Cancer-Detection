# 🧠 Breast Cancer Risk Assessment Tool

This project is a **machine learning-powered web application** for predicting the likelihood of breast cancer based on key tumor features. Built using `Streamlit`, it offers an intuitive interface for healthcare researchers, students, and professionals to assess cancer risk quickly and accurately.

---

## 🚀 Features

- 🔍 Predicts malignancy probability using trained ML model  
- 📊 Accepts 10 tumor-related input features (mean & worst values)  
- 🧠 Trained on the **Wisconsin Breast Cancer Dataset**  
- 📈 Returns prediction along with confidence probability  
- ⚠️ Includes medical disclaimer for responsible usage  

---

## 🧪 Tech Stack

- Python  
- Streamlit  
- scikit-learn  
- NumPy & Pandas  
- Joblib (for model persistence)

---

## 📂 File Structure

| File | Description |
|------|-------------|
| `app.py` | Streamlit app for user interface and prediction |
| `model.joblib` | Trained machine learning model |
| `data.csv` | Source dataset (Wisconsin Breast Cancer) |
| `application.ipynb` | Model training and analysis notebook |
| `req.txt` | List of Python dependencies |

---

## 🧠 Model Details

- **Model Type**: Logistic Regression  
- **Regularization**: L1 (Lasso)  
- **Why L1?**: To improve interpretability and perform feature selection  
- **Features Used**: Selected top 10 based on domain relevance and correlation  

---

## 📦 Installation

```bash
pip install -r req.txt
streamlit run app.py
