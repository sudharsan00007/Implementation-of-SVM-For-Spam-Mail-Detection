# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the Program.
2.Import the necessary packages.
3.Read the given csv file and display the few contents of the data.
4.Assign the features for x and y respectively.
5.Split the x and y sets into train and test sets.
6.Convert the Alphabetical data to numeric using CountVectorizer.
7.Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
8.Find the accuracy of the model.
9.End the Program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: sudharsan s
RegisterNumber:  24009664
*/
```
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


file_path = r"C:\Users\sudharshan\Downloads\spam.csv"
data = pd.read_csv(file_path, encoding='latin-1')

print(data.head())

data = data.rename(columns={'v1': 'label', 'v2': 'text'})
data = data[['label', 'text']]  
data['label'] = data['label'].map({'ham': 0, 'spam': 1})  

X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = SVC(kernel='linear', C=1.0, random_state=42)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

sample_email = ["Congratulations! You've won a lottery. Click here to claim your prize."]
sample_tfidf = vectorizer.transform(sample_email)
prediction = model.predict(sample_tfidf)
print(f"Prediction for sample email: {'Spam' if prediction[0] == 1 else 'Ham'}")
```
## Output:

![Screenshot 2024-12-20 095330](https://github.com/user-attachments/assets/f45fcc88-b611-4577-a771-43d457d60dbe)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
