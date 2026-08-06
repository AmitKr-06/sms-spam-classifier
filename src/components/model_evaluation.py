import os
import sys
import json
from dataclasses import dataclass

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


@dataclass
class ModelEvaluationConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")
    metrics_file_path: str = os.path.join("artifacts", "metrics.json")


class ModelEvaluation:
    """
    Loads the trained model, scores it on the held-out test set, and saves
    a metrics.json report. This is the file you asked was still blank.
    """

    def __init__(self):
        self.eval_config = ModelEvaluationConfig()

    def initiate_model_evaluation(self, X_test, y_test):
        try:
            logging.info("Loading trained model for evaluation")
            model = load_object(self.eval_config.model_path)

            preds = model.predict(X_test)

            metrics = {
                "accuracy": float(accuracy_score(y_test, preds)),
                "precision": float(precision_score(y_test, preds)),
                "recall": float(recall_score(y_test, preds)),
                "f1_score": float(f1_score(y_test, preds)),
            }

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
            else:
                logging.info("Model has no predict_proba; skipping ROC-AUC")

            cm = confusion_matrix(y_test, preds)
            metrics["confusion_matrix"] = cm.tolist()

            report_str = classification_report(y_test, preds)
            logging.info(f"Classification report:\n{report_str}")
            print(report_str)

            os.makedirs(os.path.dirname(self.eval_config.metrics_file_path), exist_ok=True)
            with open(self.eval_config.metrics_file_path, "w") as f:
                json.dump(metrics, f, indent=4)

            logging.info(f"Metrics saved to {self.eval_config.metrics_file_path}")
            logging.info(f"Metrics: {metrics}")

            return metrics
        except Exception as e:
            raise CustomException(e, sys)