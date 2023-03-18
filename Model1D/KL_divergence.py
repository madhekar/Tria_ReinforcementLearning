
import numpy as np

def KL(a, b):
  a = np.asarray(a, dtype=np.float64)
  b = np.asarray(b, dtype=np.float64)

  return np.sum(np.where(a != 0, a * np.log(a / b), 0))


def KL_optimized(a, b):
     """ Epsilon is used here to avoid conditional code for checking that neither a nor b is equal to 0. """
     epsilon = 0.00001

     a = [_a+epsilon for _a in a]
     b = [_b+epsilon for _b in b]

     # You may want to instead make copies to avoid changing the np arrays.
     a = np.asarray([_a + epsilon for _a in a], dtype=np.float64)
     b = np.asarray([_b + epsilon for _b in b], dtype=np.float64)

     divergence = np.sum(a*np.log(a/b))

     return divergence
