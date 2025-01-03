import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':

    df = pd.read_csv('../../data/SP 500 Stock Prices 2014-2017.csv')
    print(df.groupby('symbol').size().sort_values())
    print(df[~df['high'].isnull()].groupby('symbol').size().sort_values())

    print(df['symbol'].nunique())
    print(df)
    filtered_df = df[df['symbol'].isin(['AMZN', 'GOOGL'])]

    filtered_df['date'] = pd.to_datetime(filtered_df['date'])

    pivot_df = filtered_df.pivot(index='date', columns='symbol', values='high')

    plt.figure(figsize=(10, 6))
    for column in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[column], label=column)

    plt.title('Daily Max Value for AMZN and GOOGL')
    plt.xlabel('Date')
    plt.ylabel('Daily Max Value')
    plt.legend(title='Stock Symbol')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../../artifacts/snp_3_3_1/snp_max_value_plot.png')
    plt.close()