import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
import joblib
from utils import load_config

# DagsHub setup
dagshub.init(repo_owner='lakshmiprasadlp', repo_name='MLOPS', mlflow=True)

# Load config
config = load_config()

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config['data']['test_size'], random_state=config['data']['random_state']
)

# Start MLflow tracking
with mlflow.start_run():

    # Model
    model = LogisticRegression(**config['model']['params'])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Log parameters and metrics
    mlflow.log_params(config['model']['params'])

    report = classification_report(y_test, y_pred, output_dict=True)
    mlflow.log_metrics({
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1-score": report["weighted avg"]["f1-score"]
    })

    # Log model
    joblib.dump(model, "model.pkl")
    mlflow.log_artifact("model.pkl")
