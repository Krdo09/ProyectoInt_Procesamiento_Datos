from datasets import load_dataset
import numpy as np
import pandas as pd

dataset = load_dataset("mstz/heart_failure")
data = dataset["train"]

# Parte I
def prom_age(dat: dataset):
    arr_age = np.array(dat['age'])
    print('The average age in dataset is: {:.2f}'
          .format(arr_age.mean()))


# Parte II
def to_df(dat: dataset) -> pd.DataFrame:
    df = pd.DataFrame(data)
    return df

def separation(df: pd.DataFrame):
    is_dead = df[df['is_dead'] == 1]
    is_not_dead = df[df['is_dead'] == 0]
    return is_dead, is_not_dead

def average_age(df: pd.DataFrame) -> pd.DataFrame:
    mean_age = df['age'].mean()
    return mean_age


# Parte III
def data_check(df: pd.DataFrame):
    columns_names = ['age', 'has_anaemia', 'creatinine_phosphokinase_concentration_in_blood',
              'has_diabetes', 'heart_ejection_fraction', 'has_high_blood_pressure',
              'platelets_concentration_in_blood', 'serum_creatinine_concentration_in_blood',
              'serum_sodium_concentration_in_blood', 'is_male', 'is_smoker', 'days_in_study',
              'is_dead']
    for column in columns_names:
        data_type = str(df[column].dtype)
        if data_type == 'int68':
            df[column] = pd.to_numeric(df[column], downcast='integer')

        if data_type == 'float64':
            df[column] = pd.to_numeric(df[column], downcast='float')

        if data_type == 'bool':
            df[column] = df[column].astype(bool)

def smoker(df: pd.DataFrame):
    smokers = df.groupby(['is_male', 'is_smoker']).size().reset_index(name='count')
    smokers = smokers[smokers['is_smoker'] == True]
    smokers.columns = ['is_male', 'is_smoker', 'number_smokers']
    return smokers

