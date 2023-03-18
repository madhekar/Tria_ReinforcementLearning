
import numpy as np

def KL(a, b):
  a = np.asarray(a, dtype=np.float64)
  b = np.asarray(b, dtype=np.float64)

  return np.sum(np.where(a != 0, a * np.log(a / b), 0))


values1 = [1.346112,1.337432,1.246655]
values2 = [1.033836,1.082015,1.117323]

print('KL Divergence value:', KL(values1, values2))
