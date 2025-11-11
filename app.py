from flask import Flask, request, jsonify, render_template
import joblib
import traceback

app = Flask(__name__)

# Load your trained model pipeline
# (Make sure spam_detector_pipeline.joblib is in the same folder as this file)
try:
    model = joblib.load("spam_detector_pipeline.joblib")
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_spam", methods=["POST"])
def predict_spam():
    try:
        data = request.get_json()
        email_text = data.get("message", "")

        if not email_text.strip():
            return jsonify({"error": "No email content provided"}), 400

        # Predict using the pipeline (model already includes vectorizer)
        prediction = model.predict([email_text])[0]

        # Optional: get confidence score
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba([email_text])[0]
            confidence = round(max(prob) * 100, 2)
        else:
            confidence = None

        result = "Spam" if prediction == 1 else "Not Spam"

        return jsonify({
            "result": result,
            "confidence": confidence
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)



if __name__ == "__main__":
    app.run(debug=True)

