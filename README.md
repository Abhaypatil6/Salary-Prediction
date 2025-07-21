
# Employee Salary Prediction Web Application

This project is a machine learning–based salary predictor that estimates employee salaries based on key attributes such as age, gender, education level, job title, and experience. It features an interactive Streamlit web app that provides real-time predictions, dynamic visualizations, and a stylish user interface.



## Features

🚀 Project Features

- 🔍 Predicts employee salary using a trained **Random Forest Regressor**
- ✅ Inputs: Age, Gender, Education Level, Job Title, Years of Experience
- 📊 Shows the predicted salary in **Indian Rupees (INR)**
- 📈 Compares user prediction on a live histogram with dataset statistics
- 🖼️ Custom images and dynamic visuals based on prediction range
- 🌐 Built with **Streamlit** for an easy-to-use web application
# Employee Salary Prediction Web Application

employee-salary-prediction/

├── app.py # Streamlit web application

├── train_model.py # Model training code

├── data/

│ └── employee_data.csv # Input dataset

├── model/

│ ├── salary_model.pkl # Trained machine learning model

│ └── encoder.pkl # OneHotEncoder for categorical inputs

├── images/

│ ├── aaa.jpeg # Image for middle-level salary

│ ├── bbb.jpeg # Image for early-career roles

│ └── ccc.jpeg # Header banner / high-salary image

├── .streamlit/

│ └── config.toml # Optional theme settings

├── README.md

└── requirements.txt

## Algorithm Used


---

## 🧠 Algorithm Used

**Random Forest Regressor**  
- Handles both numerical and categorical data
- Reduces overfitting using ensemble decision trees
- Provides strong prediction power and robustness

---

## ⚙️ Tech Stack

| Task                   | Tool/Library             |
|------------------------|--------------------------|
| Language               | Python 3.8+              |
| Machine Learning       | scikit-learn             |
| Data Handling          | pandas, numpy            |
| Web Framework          | Streamlit                |
| Data Visualization     | matplotlib, plotly       |
| Model Persistence      | pickle                   |
| Version Control        | Git & GitHub             |

---

## 📥 Installation & Setup

> ✅ Make sure you have Python 3.8+ installed.

1. Clone the repository:
https://github.com/Abhaypatil6/Salary-Prediction.git


2. Install the required libraries:
pip install -r requirements.txt

text

3. Train the model:
python train_model.py

text

4. Run the web app:
streamlit run app.py
## Future Enhancements

🎯 Future Enhancements

- Add login/user authentication with prediction history
- Integrate with cloud services for deployment
- Add more realistic features: location, industry, company size
- Real-time salary data updates
- Model explainability (e.g., SHAP values)

## Authors

Abhay Patil

https://github.com/Abhaypatil6

