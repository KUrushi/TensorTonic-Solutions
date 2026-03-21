def get_median(data):
    n = len(data)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return float(data[mid])
    else:
        return (data[mid - 1] + data[mid]) / 2.0

def robust_scaling(values):
    n = len(values)
    if n == 1:
        return [0.0]

    # 1. ソート
    sorted_vals = sorted(values)

    # 2. 中央値の計算
    median_val = get_median(sorted_vals)

    # 3. データの分割（中央値を除外）
    lower_half = sorted_vals[: n // 2]
    upper_half = sorted_vals[n // 2 + (n % 2) :]

    # 4. Q1, Q3の計算
    q1 = get_median(lower_half)
    q3 = get_median(upper_half)

    # 5. スケーリング
    iqr = q3 - q1
    if iqr == 0:
        return [float(x - median_val) for x in values]
    else:
        return [float(x - median_val) / iqr for x in values]