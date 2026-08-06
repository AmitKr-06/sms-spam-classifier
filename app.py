from flask import Flask, render_template, request

from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/classify", methods=["GET", "POST"])
def classify():
    result = None
    confidence = None
    message = ""

    if request.method == "POST":
        message = request.form.get("message", "").strip()

        if message:
            pipeline = PredictPipeline()
            preds, proba = pipeline.predict(message)
            result = "SPAM" if preds[0] == 1 else "HAM"
            confidence = round(float(max(proba[0])) * 100, 2)

    return render_template("index.html", result=result, confidence=confidence, message=message)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)