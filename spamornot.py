import joblib

model = joblib.load('spam_detector_model.pkl')
email = input("Paste the email content: ")
prediction = model.predict([email])[0]
if prediction == 1:
    print("🚨 This is a SPAM email.")
else:
    print("✅ This is a valid email.")
