import joblib
import pandas as pd
import mlflow
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import numpy as np
import pathlib
import sys
import pickle


def load_model(path: str) -> pd.DataFrame:
    model = joblib.load(path)
    with open(path, "rb") as f:
        loaded_model = pickle.load(f)
    return loaded_model


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def log_metrics(model, X_test, y_test) -> None:
    with mlflow.start_run():
        y_pre = model.predict(X_test,y_test)
        r2 = r2_score(y_test,y_pre)
        mae = mean_absolute_error(y_test,y_pre)
        mse = mean_squared_error(y_test,y_pre)
        rmse = np.sqrt(mse)

        mlflow.log_metric('r2',r2)
        mlflow.log_metric('mae',mae)
        mlflow.log_metric('mse',mse)
        mlflow.log_metric('rmse',rmse)


def main():
    mlflow.set_experiment("salary_test")
    
    main_path = pathlib.Path(__file__).parent.parent.parent
    model_file = sys.argv[1]
    model_path = main_path / model_file

    model = load_model(path=model_path)
    data_file = sys.argv[2]
    data_path = main_path / data_file

    test = load_data(path=data_path)
    X_test = test.drop(columns='remainder__Salaries')
    y_test = test['remainder__Salaries']
    log_metrics(X_test=X_test,y_test=y_test)


if __name__ == '__main__':
    main()

