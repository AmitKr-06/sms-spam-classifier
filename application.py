from flask import Flask, render_template, request

from src.pipeline.predict_pipeline import PredictPipeline

application = Flask(__name__)


@application.route("/")
def home():
    return render_template("home.html")


@application.route("/classify", methods=["GET", "POST"])
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
    application.run(debug=True, use_reloader=False)