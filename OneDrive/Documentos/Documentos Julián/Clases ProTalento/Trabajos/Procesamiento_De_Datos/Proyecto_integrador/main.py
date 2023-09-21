from code import *

def main():
    df_data = to_df(data)
    df1, df2 = separation(df_data)
    print(average_age(df1))
    print(average_age(df2))


if __name__ == '__main__':
    main()
