import numpy as np
def fx(c:np.typing.NDArray, x:np.typing.DNArray) -> np.typing.NDArray:
    return np.sum(c * np.power(x, np.arange(0, len(c))))
    
    

def finite_difference_derivative(coefficients: list, x: float, h: float) -> tuple[float, float, float]:
    """
    Returns the value at x, the value at x plus h, and the estimated slope.
    """
    coefficients = np.asarray(coefficients, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    fx_value = float(fx(coefficients, x))
    fx_h_value = float(fx(coefficients, x+h))
    return (fx_value, fx_h_value, ((fx_h_value - fx_value) / h))
    
