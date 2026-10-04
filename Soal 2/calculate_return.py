import csv

if __name__ == "__main__":
    # Mengolah csv stock_train.csv menjadi list of dict
    with open("stock_train.csv", newline="") as f:
        reader = csv.DictReader(f)
        close = [float(row["Close"]) for row in reader]

    # Calculates Return vector
    # return_[i] = (close[i] - close[i-1]) / close[i-1]
    return_ = [None] * len(close)
    for i in range(1, len(close)):
        return_[i] = (close[i] - close[i - 1]) / close[i - 1]

    ##### --------- #####
    # Building A        #
    ##### --------- #####

    matrix_A = []
    vector_B = []

    for i in range(len(close)):
        l1 = return_[i - 1] if i - 1 >= 0 else None  # R[i-1]
        l2 = return_[i - 2] if i - 2 >= 0 else None  # R[i-2]

        # Makes sure we build A with valid elements
        if l1 is None or l2 is None:
            continue

        pos = l1 >= 0
        neg = not pos

        # Builds overdetermined matrix A
        matrix_A.append([
            1.0 if pos else 0.0, l1 if pos else 0.0, l2 if pos else 0.0,  # 1, R[i-1], R[i-2]
            1.0 if neg else 0.0, l1 if neg else 0.0, l2 if neg else 0.0,
        ])

        ##### --------- #####
        # Building B       #
        ##### --------- #####

        vector_B.append(return_[i])
