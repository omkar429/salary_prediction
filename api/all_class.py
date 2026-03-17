# make data Frame in to predict model data class
import pathlib
import pandas as pd
import joblib

def change_OnehotEncoder(data):
    data1 = data.model_dump(exclude=['company_reviews_encoded','company_reviews'])
    data2 = data.model_dump(include=['company_reviews_encoded'])

    data1['Oneh__company_reviews_excellent'] = data2['company_reviews_encoded'][0][0]
    data1['Oneh__company_reviews_good'] = data2['company_reviews_encoded'][0][1]

    df = pd.DataFrame(data1, index=[0])
    col1 = df.pop('Oneh__company_reviews_excellent')
    col2 = df.pop('Oneh__company_reviews_good')
    df.insert(0, 'Oneh__company_reviews_excellent', col1)
    df.insert(1, 'Oneh__company_reviews_good', col2)
    df = df.values
    return df


def load_model(path: str):
    model = joblib.load(path)
    return model



def model_path():
    home_path = pathlib.Path(__file__).parent.parent
    model_path = home_path / 'models/cat/model_test.joblib'
    return model_path