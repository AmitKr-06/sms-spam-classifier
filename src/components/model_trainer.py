import os
import sys
from dataclasses import dataclass

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_config


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    """
    Responsible ONLY for: train candidate models -> pick the best by a quick
    F1 check -> save it. Detailed evaluation (precision/recall/ROC-AUC/
    confusion matrix/report) now lives in ModelEvaluation.
    """

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.params = load_config().get("model", {})

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Training candidate models")
            models = {
                "Naive Bayes": MultinomialNB(),
                "Logistic Regression": LogisticRegression(
                    max_iter=self.params.get("logistic", {}).get("max_iter", 1000)
                ),
                "SVM": SVC(probability=self.params.get("svm", {}).get("probability", True)),
            }

            model_report = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                f1 = f1_score(y_test, preds)
                model_report[name] = f1
                logging.info(f"{name} -> Spam F1: {f1:.4f}")

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with acceptable F1 score", sys)

            logging.info(f"Best model: {best_model_name} with F1: {best_model_score:.4f}")
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            return best_model_name, best_model_score
        except Exception as e:
            raise CustomException(e, sys)