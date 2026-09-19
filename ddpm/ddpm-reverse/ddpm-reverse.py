import numpy as np


def get_alpha_bar(betas: list[float]) -> list[float]:
  """Returns the cumulative alpha-bar values rounded to six decimals."""
  outputs = []
  value = 1.0
  for beta in betas:
    value *= 1 - beta
    outputs.append(round(value, 6))
  return outputs


def reverse_step(
    x_t: list, t: int, epsilon_pred: list, betas: list[float], z: list = None
) -> list:
  """Returns x at timestep t - 1, rounded to four decimals."""
  x_t = np.asarray(x_t)
  beta = betas[t]

  alpha = 1 - beta
  alpha_bar = get_alpha_bar(betas)[t]
  epsilon_pred = np.asarray(epsilon_pred)

  mu_t = 1 / np.sqrt(alpha) * (x_t - beta / np.sqrt(1 - alpha_bar) * epsilon_pred)

  # t > 1（途中ステップ）かつ z が渡されている場合にノイズを加算
  if t > 1 and z is not None:
    z = np.asarray(z)
    mu_t += np.sqrt(beta) * z

  # 小数点第4位で丸めて list 型で返す
  return np.round(mu_t, 4).tolist()
import numpy as np


def reverse_step(
    x_t: list, t: int, epsilon_pred: list, betas: list[float], z: list = None
) -> list:
  """Performs one DDPM reverse-diffusion step with 1-based index `t`."""
  # float64 精度の NumPy 配列へ変換
  x_t_arr = np.asarray(x_t, dtype=np.float64)
  eps_arr = np.asarray(epsilon_pred, dtype=np.float64)
  betas_arr = np.asarray(betas, dtype=np.float64)

  # 1-based インデックスの処理
  beta_t = betas_arr[t - 1]
  alpha_t = 1.0 - beta_t
  alpha_bar_t = np.prod(1.0 - betas_arr[:t])

  # \mu_t の計算
  mu_t = (1.0 / np.sqrt(alpha_t)) * (
      x_t_arr - (beta_t / np.sqrt(1.0 - alpha_bar_t)) * eps_arr
  )

  # t > 1 の場合はノイズを加算、t = 1 の場合はノイズを無視
  if t > 1 and z is not None:
    z_arr = np.asarray(z, dtype=np.float64)
    x_prev = mu_t + np.sqrt(beta_t) * z_arr
  else:
    x_prev = mu_t

  # 小数点第4位に丸めてネストしたリスト形式で返却
  return np.round(x_prev, 4).tolist()