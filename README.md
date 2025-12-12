# SMS & Email Spam Detector

This project implements a machine learning–based Spam Detection System for SMS and Email messages.  
It classifies messages as Spam or Not Spam using NLP techniques and supervised learning models, and is deployed as an interactive Streamlit web application.


---

## Project Workflow

The project is divided into four major stages:

1. Data Cleaning  
   - Removal of unnecessary columns  
   - Text normalization (lowercasing, punctuation removal, stopword removal)  
   - Tokenization and stemming  

2. Exploratory Data Analysis (EDA)  
   - Spam vs Non-Spam distribution  
   - Message length analysis  
   - Frequent word visualization  

3. Model Building  
   - Feature extraction using TF-IDF Vectorization  
   - Training multiple machine learning classifiers  

4. Model Selection and Deployment  
   - Model comparison based on Accuracy and Precision  
   - Best-performing model deployed using Streamlit  

---

## Machine Learning Models Used

- Naive Bayes  
- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest  
- Decision Tree  
- K-Nearest Neighbors (KNN)  

Evaluation Metrics:
- Accuracy  
- Precision  

The final model was selected based on high precision, which is critical for spam detection tasks.

---

## Streamlit Web Application

The trained model is deployed as a Streamlit web application, allowing users to input custom messages and receive instant predictions.

Spam Prediction  
The application correctly identifies spam messages in real time.

Non-Spam Prediction  
The application accurately classifies genuine messages as non-spam.

---

## Technologies and Libraries Used

- Python  
- Pandas  
- NumPy  
- NLTK  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Streamlit  

---

## Key Highlights

- End-to-end machine learning pipeline  
- Strong NLP preprocessing workflow  
- Multiple model comparison and evaluation  
- Interactive user interface using Streamlit  
- Easily extendable to Email spam detection  

---

## Future Enhancements

- Add deep learning models such as LSTM or BERT  
- Improve UI with advanced Streamlit components  
- Deploy on cloud platforms like AWS, GCP, or Azure  
- Support multi-language spam detection  

---
