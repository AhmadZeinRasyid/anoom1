import csv
import math


##### --------- #####
# Linear algebra helpers (no numpy)   #
##### --------- #####

def transpose(M):
    return [[M[i][j] for i in range(len(M))] for j in range(len(M[0]))]


def matmul(A, B):
    rows_A, cols_A = len(A), len(A[0])
    cols_B = len(B[0])
    result = [[0.0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for k in range(cols_A):
            aik = A[i][k]
            if aik == 0.0:
                continue
            for j in range(cols_B):
                result[i][j] += aik * B[k][j]
    return result


def mat_vec(M, v):
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def vec_norm(v):
    return math.sqrt(dot(v, v))


def vec_scale(v, s):
    return [x * s for x in v]


def solve_linear_system(M, b):
    # Gauss-Jordan elimination with partial pivoting
    n = len(M)
    aug = [row[:] + [b[i]] for i, row in enumerate(M)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-15:
            raise ValueError("Singular matrix, cannot solve")
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot
        for r in range(n):
            if r != col:
                factor = aug[r][col]
                if factor != 0.0:
                    for j in range(col, n + 1):
                        aug[r][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


def power_iteration(M, tol=1e-12, max_iter=5000):
    # Largest eigenvalue of symmetric matrix M via power iteration
    n = len(M)
    v = vec_scale([1.0] * n, 1.0 / math.sqrt(n))
    eigenvalue = 0.0
    for _ in range(max_iter):
        w = mat_vec(M, v)
        norm_w = vec_norm(w)
        if norm_w == 0.0:
            return 0.0
        w = vec_scale(w, 1.0 / norm_w)
        new_eigenvalue = dot(w, mat_vec(M, w))
        v = w
        if abs(new_eigenvalue - eigenvalue) < tol:
            eigenvalue = new_eigenvalue
            break
        eigenvalue = new_eigenvalue
    return eigenvalue


def inverse_power_iteration(M, tol=1e-12, max_iter=5000):
    # Smallest eigenvalue of symmetric matrix M via inverse power iteration
    n = len(M)
    v = vec_scale([1.0] * n, 1.0 / math.sqrt(n))
    eigenvalue = 0.0
    for _ in range(max_iter):
        w = solve_linear_system(M, v)
        norm_w = vec_norm(w)
        if norm_w == 0.0:
            break
        w = vec_scale(w, 1.0 / norm_w)
        new_eigenvalue = dot(w, mat_vec(M, w))
        v = w
        if abs(new_eigenvalue - eigenvalue) < tol:
            eigenvalue = new_eigenvalue
            break
        eigenvalue = new_eigenvalue
    return eigenvalue


def condition_number_symmetric(M):
    # kappa_2(M) = lambda_max / lambda_min, for symmetric positive-definite M
    lambda_max = power_iteration(M)
    lambda_min = inverse_power_iteration(M)
    if lambda_min <= 0.0:
        raise ValueError("M is singular or not positive definite, condition number is infinite")
    return lambda_max / lambda_min


def condition_number(A):
    # kappa_2(A) for a (possibly rectangular) A: sqrt(lambda_max / lambda_min) of A^T A
    AtA = matmul(transpose(A), A)
    return math.sqrt(condition_number_symmetric(AtA))


##### --------- #####
# Normal equations         #
##### --------- #####

def build_normal_equations(A, b):
    # Builds (A^T A) x = A^T b, the normal equations for the least squares problem A x = b
    At = transpose(A)
    AtA = matmul(At, A)
    Atb = mat_vec(At, b)
    return AtA, Atb


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

    ##### --------- #####
    # Solving via normal equations #
    ##### --------- #####

    AtA, Atb = build_normal_equations(matrix_A, vector_B)
    coeffs = solve_linear_system(AtA, Atb)

    print("coefficients (c_pos, b1_pos, b2_pos, c_neg, b1_neg, b2_neg):")
    print(coeffs)

    ##### --------- #####
    # Conditioning comparison #
    ##### --------- #####

    kappa_A = condition_number(matrix_A)
    kappa_AtA = condition_number_symmetric(AtA)

    print()
    print(f"kappa_2(A)      = {kappa_A:.6f}")
    print(f"kappa_2(A^T A)  = {kappa_AtA:.6f}")
    print(f"kappa_2(A)^2    = {kappa_A ** 2:.6f}  (should match kappa_2(A^T A))")
