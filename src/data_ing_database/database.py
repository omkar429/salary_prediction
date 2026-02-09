import pandas as pd
from sklearn.model_selection import train_test_split
import pathlib
import sys
import yaml
import mlflow
import dagshub
import os


# load data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def drop(data: pd.DataFrame) -> pd.DataFrame:
    data.dropna(inplace=True)
    return data


# drop Duplicated data
def drop_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    df = data.drop_duplicates()
    return df

# corect the data
def data_corect(df: pd.DataFrame) -> pd.DataFrame:
    df['Reviews'] = df['allas'].str.split('Reviews').str[0]
    df['Salaries'] = df['allas'].str.split(' ').str[1].str.split('Reviews').str[1]
    df['Interviews'] = df['allas'].str.split(' ').str[2].str.split('Salaries').str[1]
    df['jobs'] = df['allas'].str.split(' ').str[3].str.split('Interviews').str[1]
    df['Benefits'] = df['allas'].str.split(' ').str[4].str.split('Jobs').str[1]
    df['photos'] = df['allas'].str.split(' ').str[5].str.split('Benefits').str[1]
    return df

# drop unesery columns 
def drop_columns(data: pd.DataFrame) -> pd.DataFrame:
    df = data.drop(columns='allas')
    return df

# train test split
def train_test(df: pd.DataFrame, test_size, random_state) -> pd.DataFrame:
    train, test = train_test_split(df,test_size=test_size, random_state=random_state)
    return train, test

# save data
def save_data(path: str, train: pd.DataFrame, test: pd.DataFrame) -> None:
    pathlib.Path(path).mkdir(parents=True,exist_ok=True)
    train.to_csv(path / 'train.csv',index=False)
    test.to_csv(path / 'test.csv',index=False)

def main():

    curr_path = pathlib.Path(__file__)
    main_path = curr_path.parent.parent.parent
    data_file_path = sys.argv[1]
    data_path = main_path / data_file_path
    save_data_flie = sys.argv[2]
    save_data_path = main_path / save_data_flie
    params = yaml.safe_load(open('params.yaml'))['database']
    test_size = params['test_size']
    random_state = params['random_state']
       
        
    df = load_data(data_path)
    df = drop(data=df)
    df = drop_duplicates(df)
    df = data_corect(df)
    df = drop_columns(df)
    train, test = train_test(df,test_size=test_size, random_state=random_state)
    save_data(train=train, test=test, path=save_data_path)


if __name__ == '__main__':
    main()
