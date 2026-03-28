#import the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import os

pio.renderers.default = "browser"


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#Load Dataset
data = {
    'Student_ID': ['STU_201', 'STU_202', 'STU_203', 'STU_204', 'STU_205', 'STU_206', 'STU_207', 'STU_208', 'STU_209', 'STU_210', 'STU_211', 'STU_212', 'STU_213', 'STU_214', 'STU_215', 'STU_216', 'STU_217', 'STU_218', 'STU_219', 'STU_220'],
    'Attendance_Percentage': [0.92, 0.85, 0.65, 0.78, 0.95, 0.5, 0.88, 0.72, 0.9, 0.6, 0.98, 0.82, 0.45, 0.7, 0.89, 0.62, 0.77, 0.94, 0.55, 0.81],
    'Study_Hours_Per_Week': [15, 12, 4, 10, 20, 2, 14, 8, 18, 5, 22, 11, 3, 9, 16, 6, 13, 19, 4, 10],
    'Placement_Study_Hours': [10, 8, 15, 5, 12, 20, 6, 10, 14, 18, 10, 7, 25, 4, 12, 15, 9, 11, 22, 8],
    'Assignment_Score': [88, 76, 45, 82, 94, 30, 70, 62, 85, 50, 96, 74, 25, 65, 80, 55, 78, 92, 40, 72],
    'Internal_Marks': [22, 18, 10, 19, 24, 8, 20, 15, 21, 12, 25, 17, 7, 14, 21, 11, 19, 23, 9, 16],
    'Final_Exam_Score': [84, 72, 32, 68, 91, 25, 75, 58, 82, 40, 95, 70, 20, 55, 79, 42, 74, 88, 35, 65],
    'Result': ['Pass', 'Pass', 'Fail', 'Pass', 'Pass', 'Fail', 'Pass', 'Pass', 'Pass', 'Fail', 'Pass', 'Pass', 'Fail', 'Pass', 'Pass', 'Fail', 'Pass', 'Pass', 'Fail', 'Pass']
}

df = pd.DataFrame(data)

#Basic Cleaning of the Data
df.fillna(df.mean(numeric_only=True), inplace=True)
df['Result'] = df['Result'].map({'Pass': 1, 'Fail': 0})

#Exploratory Data Analysis (EDA)
sns.heatmap(
    df.select_dtypes(include='number').corr().abs(),
    annot=True,
    cmap='Blues'
)
plt.title("Absolute Correlation Heatmap")
plt.show()


#Data Visualization
plt.hist(df['Final_Exam_Score'], bins=10)
plt.xlabel("Final Exam Score")
plt.ylabel("Number of Students")
plt.title("Distribution of Final Exam Scores")
plt.show()

#Seaborn Visulization
sns.boxplot(x='Result', y='Attendance_Percentage', data=df)
plt.title("Attendance vs Result")
plt.show()

#Intersactive Visualization with Plotly
fig1 = px.box(
    df,
    x='Result',
    y='Study_Hours_Per_Week',
    color='Result',
    title='Study Hours Comparison Between Pass and Fail Students'
)
fig1.show()


fig2 = px.box(
    df,
    x='Result',
    y='Placement_Study_Hours',
    color='Result',
    title='Placement Study Hours Comparison Between Pass and Fail Students'
    
)
fig2.show()


#Machine Learning Model
X = df[['Attendance_Percentage',
        'Study_Hours_Per_Week',
        'Assignment_Score',
        'Internal_Marks']]
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("ML Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

#Deep Learning Model
ann = Sequential()
ann.add(Dense(8, activation='relu', input_shape=(4,)))
ann.add(Dense(1, activation='sigmoid'))

ann.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])

ann.fit(X_train, y_train, epochs=20, validation_split=0.2)
ann.evaluate(X_test, y_test)





