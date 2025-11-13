import py_ldpc4qkd as ldpc

print("hello world")
print(ldpc.__version__)

import numpy as np

print(np.__version__)


def print_available_codes():
    for i in range(1_000_000):
        try:
            code = ldpc.get_rate_adaptive_code(i)
            print(f"Code {i} maps {code.getNCols()} -> {code.get_n_rows_mother_matrix()}")
        except RuntimeError as e:  # `get_rate_adaptive_code` throws `RuntimeError` if no code availabel for ID.
            break
    print("\n")


if __name__ == "__main__":
    print_available_codes()
