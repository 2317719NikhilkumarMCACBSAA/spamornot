# 📦 1. Import Libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

spam_keywords = ['free', 'win', 'money', 'click', 'urgent', 'offer', 'credit', 'prize', 'claim', 'guaranteed']

data = {
    'message': [
        "You have won a free prize! Click now to claim",
        "Meeting at 3PM with the team",
        "Urgent offer just for you, win money now",
        "Please find attached your invoice",
        "Claim your guaranteed reward today",
        "Let's catch up this weekend"
    ],
    'label': [1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)


model = Pipeline([
    ('vectorizer', CountVectorizer(vocabulary=spam_keywords)), 
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


joblib.dump(model, 'spam_detector_model.pkl')
