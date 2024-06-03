import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# שלב 1: טעינת הנתונים מקובץ CSV
df = pd.read_csv('D:\Tichnut B\AI\iris.csv')  # ודא שהנתיב כאן הוא הנתיב הנכון למיקום שבו שמרת את הקובץ

# שלב 2: חקירת הנתונים
print(df.head())

# שלב 3: הכנת הנתונים - הפרדת המאפיינים מהתוויות
X = df.drop(columns=['species'])
y = df['species']

# שלב 4: חלוקת הנתונים למערכי אימון ובדיקה
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# שלב 5: בניית מודל 1 - Decision Tree
model1 = DecisionTreeClassifier()

# שלב 6: אימון מודל 1
model1.fit(X_train, y_train)

# שלב 7: הערכת מודל 1
y_pred1 = model1.predict(X_test)
print("Evaluation of Decision Tree Classifier (Model 1):")
print(f"Accuracy: {accuracy_score(y_test, y_pred1)}")
print(classification_report(y_test, y_pred1))

# שלב 8: בניית מודל 2 - Logistic Regression
model2 = LogisticRegression(max_iter=200)

# שלב 9: אימון מודל 2
model2.fit(X_train, y_train)

# שלב 10: הערכת מודל 2
y_pred2 = model2.predict(X_test)
print("\nEvaluation of Logistic Regression (Model 2):")
print(f"Accuracy: {accuracy_score(y_test, y_pred2)}")
print(classification_report(y_test, y_pred2))

# שלב 11: שימוש בשני המודלים לחיזוי נתונים חדשים
new_data = [[5.1, 3.5, 1.4, 0.2]]  # לדוגמה, נתוני פרח חדש

# חיזוי באמצעות מודל 1
prediction1 = model1.predict(new_data)
print(f"\nThe predicted class for the new data using Decision Tree is: {prediction1[0]}")

# חיזוי באמצעות מודל 2
prediction2 = model2.predict(new_data)
print(f"The predicted class for the new data using Logistic Regression is: {prediction2[0]}")
