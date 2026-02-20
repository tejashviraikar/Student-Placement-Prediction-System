import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("Job_Placement_Data_Enhanced.csv")

# -----------------------------
# Select Required Columns
# -----------------------------
features = [
    'gender',
    'ssc_percentage',
    'hsc_percentage',
    'hsc_subject',
    'undergrad_degree',
    'degree_percentage',
    'work_experience',
    'years_experience',
    'internship_completed',
    'interview_score',
    'company_tier'
]

target = 'status'

df = df[features + [target]]

# -----------------------------
# Model Preparation
# -----------------------------
X = df[features]
y = df[target]

categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

# Model Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# -----------------------------
# Title
# -----------------------------
st.title("🎓 Student Placement Prediction Dashboard")

st.write("Enter student details below:")

# -----------------------------
# SERIAL INPUTS (CENTERED)
# -----------------------------

name = st.text_input("1️⃣ Student Name")

gender = st.selectbox("2️⃣ Gender", df['gender'].unique())

ssc_percentage = st.number_input("3️⃣ SSC Percentage", 0.0, 100.0, 70.0)

hsc_percentage = st.number_input("4️⃣ HSC Percentage", 0.0, 100.0, 60.0)

hsc_subject = st.selectbox("5️⃣ HSC Stream", df['hsc_subject'].unique())

undergrad_degree = st.selectbox(
    "6️⃣ Undergraduate Branch",
    ["Computer", "ENTC", "Electrical"]
)

degree_percentage = st.number_input("7️⃣ Degree Percentage", 0.0, 100.0, 60.0)

work_experience = st.selectbox("8️⃣ Work Experience", df['work_experience'].unique())

years_experience = st.number_input("9️⃣ Years of Experience", 0.0, 10.0, 0.0)

internship_completed = st.selectbox("🔟 Internship Completed", df['internship_completed'].unique())

interview_score = st.number_input("11️⃣ Interview Score", 0.0, 100.0, 60.0)

company_tier = st.selectbox("12️⃣ Company Tier", df['company_tier'].unique())

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Placement Status"):

    input_data = pd.DataFrame([{
        'gender': gender,
        'ssc_percentage': ssc_percentage,
        'hsc_percentage': hsc_percentage,
        'hsc_subject': hsc_subject,
        'undergrad_degree': undergrad_degree,
        'degree_percentage': degree_percentage,
        'work_experience': work_experience,
        'years_experience': years_experience,
        'internship_completed': internship_completed,
        'interview_score': interview_score,
        'company_tier': company_tier
    }])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    st.subheader(f"Prediction Result for {name}")

    if prediction.lower() == "placed":
        st.success(f"{name} is likely to be PLACED ✅")
    else:
        st.error(f"{name} is likely to be NOT PLACED ❌")

    st.write("### Prediction Confidence")

    # Safe probability extraction
    class_labels = model.classes_

    for label, prob in zip(class_labels, probabilities):
        st.write(f"{label} Probability: {round(prob * 100, 2)}%")

# -----------------------------
# Insights Section (WORKING)
# -----------------------------
st.markdown("---")
st.subheader("📊 Dataset Insights")

st.write(f"### Model Accuracy: {round(accuracy*100,2)}%")

# 1️⃣ Placement Distribution
fig1, ax1 = plt.subplots()
df['status'].value_counts().plot(kind='bar', ax=ax1)
ax1.set_title("Placement Distribution")
ax1.set_xlabel("Status")
ax1.set_ylabel("Count")
st.pyplot(fig1)

# 2️⃣ Average Interview Score by Placement
fig2, ax2 = plt.subplots()
df.groupby("status")["interview_score"].mean().plot(kind='bar', ax=ax2)
ax2.set_title("Average Interview Score by Placement")
ax2.set_ylabel("Average Score")
st.pyplot(fig2)

# 3️⃣ Average Degree Percentage by Placement
fig3, ax3 = plt.subplots()
df.groupby("status")["degree_percentage"].mean().plot(kind='bar', ax=ax3)
ax3.set_title("Average Degree Percentage by Placement")
ax3.set_ylabel("Average Percentage")
st.pyplot(fig3)
