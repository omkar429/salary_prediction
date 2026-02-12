import catboost
import pandas as pd
from sklearn.linear_model import LinearRegression
import pathlib
import joblib
import sys
import yaml
import pickle

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# def make_pairam(iterations,depth,learning_rate,l2_leaf_reg,border_count):
#     pairamiter_grid = {
#         'iterations':iterations,
#         'depth': depth,
#         'learning_rate':learning_rate,
#         'l2_leaf_reg': l2_leaf_reg,
#         'border_count': border_count
        

#     }
#     print(pairamiter_grid)
#     return pairamiter_grid


def modelsa(X_train, y_train):
    # grid = RandomizedSearchCV(
    #     estimator=catboost.CatBoostRegressor(random_state=42, verbose=0),
    #     param_distributions=pairamiter_grid,
    #     n_jobs=-1,
    #     verbose=0,
    #     cv=4
    # )
    # grid.fit(X_train,y_train)
    # model = grid.best_estimator_
    # return model
    lr = LinearRegression()
    lr.fit(X_train,y_train)
    return lr

def save_model(path, model):
    path.mkdir(parents=True, exist_ok=True) 
    path = pathlib.Path(path)
    joblib.dump(model,path / 'model_test.joblib')
        
    # path.mkdir(parents=True, exist_ok=True) 
    # model.save_model(path / "my_model.cbm")
    

def main():
    main_path = pathlib.Path(__file__).parent.parent.parent

    data_file = sys.argv[1]
    data_path = main_path / data_file

    train = load_data(path=data_path)

    pairams = yaml.safe_load(open('params.yaml'))['make_model']
    # pairam_grid = make_pairam(iterations=pairams['iterations'], depth=pairams['depth'],learning_rate=pairams['learning_rate'],l2_leaf_reg=pairams['l2_leaf_reg'],border_count=pairams['border_count'])
    
    X_train = train.drop(columns='remainder__Salaries')
    y_train = train['remainder__Salaries']

    model = modelsa(X_train=X_train,y_train=y_train)

    model_file = sys.argv[2]
    model_path = main_path / model_file

    save_model(path=model_path,model=model)


if __name__ == '__main__':
    main()


