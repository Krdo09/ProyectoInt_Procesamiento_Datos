from code import *

def main():
    df = pd.read_csv('data_clean.csv')
    hist_group(df)


if __name__ == '__main__':
    main()
