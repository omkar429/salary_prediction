import joblib
import pandas as pd
import mlflow
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import numpy as np
import pathlib
import sys
import pickle
import yaml
from catboost import CatBoostRegressor


def load_model(path: str) -> pd.DataFrame:
    model = joblib.load(path)
    # model = CatBoostRegressor()
    # model.load_model(path,format="cbm")
    
    return model


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def log_metrics(model, X_test, y_test, path) -> None:
    with mlflow.start_run():
        y_pre = model.predict(X_test)
        r2 = r2_score(y_test,y_pre)
        mae = mean_absolute_error(y_test,y_pre)
        mse = mean_squared_error(y_test,y_pre)
        rmse = np.sqrt(mse)

        mlflow.log_param('web_scrap',path['web_scrap']['page_number'])
        mlflow.log_param('test_size',path['database']['test_size'])
        mlflow.log_param('random_state', path['database']['random_state'])
        mlflow.log_param('n_estimators',path['make_model']['n_estimators'])
        mlflow.log_param('depth',path['make_model']['depth'])
        mlflow.log_param('learning_rate',path['make_model']['learning_rate'])
        mlflow.log_param('l2_leaf_reg',path['make_model']['l2_leaf_reg'])
        mlflow.log_param('border_count',path['make_model']['border_count'])


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

    pairam_path = yaml.safe_load(open('params.yaml'))
    log_metrics(X_test=X_test,y_test=y_test,model=model, path=pairam_path)


if __name__ == '__main__':
    main()

