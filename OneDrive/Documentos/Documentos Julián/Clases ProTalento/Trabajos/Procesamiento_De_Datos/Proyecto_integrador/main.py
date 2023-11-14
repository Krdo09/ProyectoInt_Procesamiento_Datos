from code import *

def main():
    df = pd.read_csv('data_clean.csv')
    pies_graphic(df)


if __name__ == '__main__':
    main()
