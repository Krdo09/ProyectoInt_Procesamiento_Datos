from code import *
import plotly

def main():
    df = pd.read_csv('data_clean.csv')
    tree_model(df)


if __name__ == '__main__':
    main()
