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
