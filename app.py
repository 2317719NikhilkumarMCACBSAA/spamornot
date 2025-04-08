from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('spam_detector_model.pkl')

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email = request.form['email']
        prediction = model.predict([email])[0]
        result = "🚨 This is a SPAM email." if prediction == 1 else "✅ This is a normal email."
        return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
