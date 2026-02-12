import catboost
import pandas as pd
from sklearn.model_selection import GridSearchCV
import pathlib
import joblib
import sys
import yaml

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def make_pairam(iterations,depth,learning_rate,l2_leaf_reg,border_count):
    pairamiter_grid = {
        'iterations':iterations,
        'depth': depth,
        'learning_rate':learning_rate,
        'l2_leaf_reg': l2_leaf_reg,
        'border_count': border_count

    }
    return pairamiter_grid


def models(pairamiter_grid, X_train, y_train):
    grid = GridSearchCV(
        estimator=catboost.CatBoostRegressor(),
        param_grid=pairamiter_grid,
        n_jobs=-1,
        verbose=0,
        cv=4
    )
    grid.fit(X_train,y_train)
    model = grid.best_estimator_
    return model

def save_model(path,model):
    pathlib.Path(path).mkdir(exist_ok=True,parents=True)
    joblib.dump(model, path / 'test.pkl')

def main():
    main_path = pathlib.Path(__file__).parent.parent.parent

    data_file = sys.argv[1]
    data_path = main_path / data_file

    train = load_data(path=data_path)

    pairams = yaml.safe_load(open('params.yaml'))['make_model']
    pairam_grid = make_pairam(iterations=pairams['iterations'], depth=pairams['depth'],learning_rate=pairams['learning_rate'],l2_leaf_reg=pairams['l2_leaf_reg'],border_count=pairams['border_count'])
    
    X_train = train.drop(columns='remainder__Salaries')
    y_train = train['remainder__Salaries']

    model = models(pairamiter_grid=pairam_grid,X_train=X_train,y_train=y_train)

    model_file = sys.argv[2]
    model_path = main_path / model_file

    save_model(path=model_path,model=model)


