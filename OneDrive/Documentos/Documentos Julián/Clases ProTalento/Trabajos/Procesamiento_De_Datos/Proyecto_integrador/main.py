from code import *
import plotly

def main():
    df = pd.read_csv('data_clean.csv')
    random_forest(df)



if __name__ == '__main__':
    main()
