from code import *
import plotly

def main():
    df = pd.read_csv('data_clean.csv')
    model_creation(df)


if __name__ == '__main__':
    main()
