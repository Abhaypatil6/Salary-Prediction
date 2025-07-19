import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# --- Custom Style: Modern Gradient & Buttons ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 50%, #667eea 100%);
        min-height: 100vh;
    }
    .main {
        background-color: rgba(255,255,255,0.88);
        border-radius: 18px;
        padding: 2rem;
        margin-top: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 2px 20px 0 rgba(102,126,234,0.08);
    }
    .stButton button {
        background-color: #5433ff;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        font-size: 1rem;
    }
    h1, h2 {
        color: #22223b;
    }
    </style>
""", unsafe_allow_html=True)

# --- Banner Image (now the 3rd image comes in the first place) ---
st.image('ccc.jpeg', caption="App Banner", use_container_width=True)

st.markdown("""
<div style="text-align:center; padding:18px; background: linear-gradient(45deg, #38f9d7, #43e97b); border-radius: 10px;">
  <h1 style="color:#fff; font-family:sans-serif;">💼 Salary Predictor</h1>
  <p style="color:#eee; font-size:18px;">Discover what you're worth based on your skills and experience.</p>
</div>
""", unsafe_allow_html=True)

st.header("Predict Your Salary with Style and Insight")

# --- Load Model, Encoder, Data ---
with open('model/salary_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

data = pd.read_csv('data/employee_data.csv')

# --- Dynamic User Choices from Data ---
genders = sorted(data['Gender'].dropna().unique())
educations = sorted(data['Education Level'].dropna().unique())
job_titles = sorted(data['Job Title'].dropna().unique())

with st.container():
    age = st.number_input("Age", min_value=18, max_value=65, step=1)
    gender = st.selectbox("Gender", genders)
    education = st.selectbox("Education Level", educations)
    job = st.selectbox("Job Title", job_titles)
    exp = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, step=0.5)

    if st.button("🌟 Predict Salary"):
        input_df = pd.DataFrame(
            [[age, gender, education, job, exp]],
            columns=["Age", "Gender", "Education Level", "Job Title", "Years of Experience"]
        )
        input_encoded = encoder.transform(input_df[["Gender", "Education Level", "Job Title"]])
        input_final = np.concatenate(
            [input_df[["Age", "Years of Experience"]].values, input_encoded], axis=1
        )
        salary_pred = model.predict(input_final)
        st.success(f"🎊 Predicted Salary: ${salary_pred[0]:,.2f}")

        # --- Dynamic result image based on Salary Range ---
        if salary_pred[0] > 100000:
            st.image("rrr.jpeg", caption="You're in the top bracket!", width=250, use_container_width=True)
        # elif salary_pred[0] < 50000:
        #     st.image("bbb.jpeg", caption="Early Career Level", width=250, use_container_width=True)
        else:
            st.image('aaa.jpeg', caption="Growing Strong!", width=250, use_container_width=True)

        st.markdown("""
        <span style='color:#43e97b; font-size:18px;'>Congratulations on taking your career to the next step!</span>
        """, unsafe_allow_html=True)

        st.subheader("🔎 Where Does Your Salary Stand?")
        fig, ax = plt.subplots()
        ax.hist(data['Salary'].dropna(), bins=30, color='#667eea', edgecolor='black', alpha=0.8)
        ax.axvline(salary_pred[0], color='limegreen', linewidth=2, linestyle='dashed', label='Your Prediction')
        ax.set_xlabel("Salary")
        ax.set_ylabel("Number of Employees")
        ax.legend()
        st.pyplot(fig)

st.markdown("---")
st.subheader("📊 Salary Statistics From Your Dataset")

fig1, ax1 = plt.subplots()
ax1.hist(data['Salary'].dropna(), bins=30, color='#38f9d7', edgecolor='black', alpha=0.85)
ax1.set_title("Overall Salary Distribution")
ax1.set_xlabel("Salary")
ax1.set_ylabel("Number of Employees")
st.pyplot(fig1)

st.subheader("💡 Average Salary by Job Title")
avg_salary = data.groupby('Job Title')['Salary'].mean().sort_values()
st.bar_chart(avg_salary)

st.subheader("📈 Salary vs. Years of Experience")
fig2 = px.scatter(
    data.dropna(subset=["Salary", "Years of Experience"]),
    x="Years of Experience", y="Salary", color="Education Level",
    title="Experience vs. Salary"
)
st.plotly_chart(fig2)

# --- Sidebar Logo/Branding (ensure this is a distinctive image) ---
st.sidebar.image('bbb.jpeg', caption='Your Company', width=120, use_container_width=True)
st.sidebar.markdown("#### Made with ❤️ using Streamlit ")
