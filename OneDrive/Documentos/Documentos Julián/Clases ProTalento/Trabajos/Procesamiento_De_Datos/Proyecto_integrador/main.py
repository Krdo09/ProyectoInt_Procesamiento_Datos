from code import *

def main():
    df = pd.read_csv('data.csv')
    df_clean = data_processing(df)


if __name__ == '__main__':
    main()
