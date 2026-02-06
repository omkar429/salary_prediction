import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pathlib
import sys

# load data
def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


# Change company_reviews ing to catagorical
def change_into_cat(df):
    a = []
    for i in df['company_reviews']:
        if i > 4:
            a.append('excellent')
        elif i < 4 and i > 3:
            a.append('good')
        else:
            a.append('Poor') 
    df['company_reviews'] = a

    return df


# remove chairecter
def remove_car(df: str) -> pd.DataFrame:
    df['Reviews'] = df['Reviews'].str.replace('k','')
    df['Reviews'] = df['Reviews'].str.replace('L','')
    df['Reviews'] = df['Reviews'].str.replace('--','')
    df['Reviews'] = df['Reviews'].str.replace(' ','0')
    df['Reviews'] = df['Reviews'].astype(float)
    return df


# change astyp
def change_astype(df: pd.DataFrame) -> pd.DataFrame:
    df['Reviews'] = df['Reviews'].astype(float)
    return df


# remove letter in salary columns and change the datatype into float
def replace_salary_car(df: pd.DataFrame) -> pd.DataFrame:
    b = []

    for i in df['Salaries']:
        if 'k' in i:
            n = i.replace('k','')
            n = float(n)
            n = n * 1000
            b.append(n)
        elif 'L' in i:
            n = i.replace('L','')
            n = float(n)
            n = n * 100000
            b.append(n)
        elif '--' in i:
            n = i.replace('--','0')
            n = float(n)
            n = n * 100000
            b.append(n)
        else:
            b.append(i)
    df['Salaries'] = b
    df['Salaries'] = df.drop(index=952)['Salaries'].astype(float)
    return df
    


# remove into interviews k cairecter 
def replace_Interviews_car(df: pd.DataFrame) -> pd.DataFrame:
    b = []

    for i in df['Interviews']:
        if 'k' in i:
            n = i.replace('k','')
            n = float(n)
            n = n * 1000
            b.append(n)
        elif '--' in i:
            n = i.replace('--','0')
            n = float(n)
            b.append(n)
        else:
            b.append(i)
    df['Interviews'] = b
    df['Interviews'] = df['Interviews'].astype(float)

    return df
    

def replace_jobs_car(df: pd.DataFrame) -> pd.DataFrame:
    b = []

    for i in df['jobs']:
        if 'k' in i:
            n = i.replace('k','')
            n = float(n)
            n = n * 1000
            b.append(n)
        elif '--' in i:
            n = i.replace('--','0')
            n = float(n)
            b.append(n)
        else:
            b.append(i)
    df['jobs'] = b
    df['jobs'] = df['jobs'].astype(float)

    return df
    

def replace_Benefits_car(df: pd.DataFrame) -> pd.DataFrame:
    b = []

    for i in df['Benefits']:
        if 'k' in i:
            n = i.replace('k','')
            n = float(n)
            n = n * 1000
            b.append(n)
        elif '--' in i:
            n = i.replace('--','0')
            n = float(n)
            b.append(n)
        else:
            b.append(i)
    df['Benefits'] = b
    df['Benefits'] = df['Benefits'].astype(float)

    return df


def replace_photos_car(df: pd.DataFrame) -> pd.DataFrame:
    b = []

    for i in df['photos']:
        if 'k' in i:
            n = i.replace('k','')
            n = float(n)
            n = n * 1000
            b.append(n)
        elif '--' in i:
            n = i.replace('--','0')
            n = float(n)
            b.append(n)
        else:
            b.append(i)
    df['photos'] = b
    df['photos'] = df['photos'].astype(int)

    return df


# drop NaN values
def drop_nan(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(inplace=True)
    return df


def columns_transormers(df: pd.DataFrame) -> pd.DataFrame:
    col = ColumnTransformer(transformers=[

        ('Oneh',OneHotEncoder(drop='first',sparse_output=False),[0]),
    ], remainder='passthrough')
    new_df = col.fit_transform(df)
    new_df = pd.DataFrame(new_df,columns=col.get_feature_names_out())
    return new_df


# save data
def save_data(train: pd.DataFrame, test: pd.DataFrame, path: str) -> None:
    pathlib.Path(path).mkdir(exist_ok=True, parents=True)
    train.to_csv(path / 'preprocessing_train.csv',index=False)
    test.to_csv(path / 'preprocessing_test.csv',index=False)


def main():
    main_path = pathlib.Path(__file__).parent.parent.parent
    train_data_file = sys.argv[1]
    test_data_file = sys.argv[2]

    train_path = main_path / train_data_file
    test_path = main_path / test_data_file

    train = load_data(path=train_path)
    test = load_data(path=test_path)

    train = change_into_cat(df=train)
    test = change_into_cat(df=test)

    train = remove_car(df=train)
    test = remove_car(df=test)

    train = change_astype(df=train)
    test = change_astype(df=test)

    train = replace_salary_car(df=train)
    test = replace_salary_car(df=test)

    train = replace_Interviews_car(df=train)
    test = replace_Interviews_car(df=test)

    train = replace_jobs_car(df=train)
    test = replace_jobs_car(df=test)

    train = replace_Benefits_car(df=train)
    test = replace_Benefits_car(df=test)

    train = replace_photos_car(df=train)
    test = replace_photos_car(df=test)

    train = drop_nan(df=train)
    test = drop_nan(df=test)

    train = columns_transormers(df=train)
    test = columns_transormers(df=test)

    save_data_file = sys.argv[3]
    save_data_path = main_path / save_data_file

    save_data(train=train, test=test, path=save_data_path)


if __name__ == '__main__':
    main()
