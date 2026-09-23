def scalar_expression_partials(a: float, b: float, c: float, h: float) -> tuple[float, float, float, float]:
    """
    Returns the expression value and numerical partials for a, b, and c.
    """
    a = np.float64(a)
    b = np.float64(b)
    c = np.float64(c)
    f_ = lambda a,b,c: a*b + c
    f = f_(a,b,c)
    f_ah = f_(a+h, b, c)
    f_bh = f_(a, b+h, c)
    f_ch = f_(a, b, c+h)
    return (float(f),float((f_ah-f)/h), float((f_bh - f)/h), float((f_ch-f)/h))