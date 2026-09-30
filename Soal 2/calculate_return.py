import numpy as np
import pandas

if __name__ == "__main__":
    # Mengolah csv stock_train.csv menjadi pandas.df
    df = pandas.read_csv("stock_train.csv")

    # Calculates Return vector
    # df['Close'].diff(1) = df['Close'][i] - df['CLose'][i-1]
    # df['Close'].shift(1) = df['Close'][i-1]
    df['Return'] = (df['Close'].diff(1) / df['Close'].shift(1))

    ##### --------- #####
    # Building A        #
    ##### --------- #####

    return_col = df['Return']
    lag1 = return_col.shift(1) # R[i-1]
    lag2 = return_col.shift(2) # R[i-2]

    # Makes sure we build A with valid elements
    valid = lag1.notna() & lag2.notna()
    l1 = lag1[valid]
    l2 = lag2[valid]

    # Boolean mask for flag
    pos = l1 >= 0
    neg = ~pos

    # Builds overdetermined matrix A
    matrix_A = np.column_stack([
        pos, pos * l1, pos * l2, # 1, R[i-1], R[i-2]
        neg, neg * l1, neg * l2
    ]).astype(float)

    ## print(matrix_A)

    ##### --------- #####
    # Building B       #
    ##### --------- #####

    vector_B = df[valid]['Return'].to_numpy()

    ## print(vector_B)
