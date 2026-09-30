import pandas as pd

    # Mengolah csv stock_train.csv and store the closing value at a variable
df = pd.read_csv("stock_train.csv")
print(df.head())
    # Menghitung deret return




def calculate_return(prev_close:float, curr_close:float) -> float:
    ## Menghitung return harian diberikan close t dan close t-1
    # prev_close = P(t-1)
    # curr_close = P(t)
    # Rt = (P(t) - P(t-1))/P(t-1)
    return (curr_close - prev_close) / prev_close
