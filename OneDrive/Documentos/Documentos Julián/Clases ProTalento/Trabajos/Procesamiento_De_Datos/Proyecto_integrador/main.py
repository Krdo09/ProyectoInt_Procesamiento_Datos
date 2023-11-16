from code import *
import plotly

def main():
    df = pd.read_csv('data_clean.csv')
    age_category(df)

if __name__ == '__main__':
    main()
