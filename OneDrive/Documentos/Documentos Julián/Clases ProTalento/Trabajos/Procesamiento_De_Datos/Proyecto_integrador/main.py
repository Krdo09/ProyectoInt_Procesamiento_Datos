from code import *
import plotly

def main():
    df = pd.read_csv('data_clean.csv')
    scatter_graphic(df)


if __name__ == '__main__':
    main()
