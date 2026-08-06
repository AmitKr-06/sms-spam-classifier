from src.pipeline.predict_pipeline import PredictPipeline

if __name__ == "__main__":
    pipeline = PredictPipeline()
    preds, proba = pipeline.predict("Congratulations! You've won a free prize, call now!")
    label = "SPAM" if preds[0] == 1 else "HAM"
    print(f"Prediction: {label}, Probabilities: {proba}")