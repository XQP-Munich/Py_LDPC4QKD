import warnings
import math
import numpy as np
import py_ldpc4qkd as ldpc

np.random.seed(42)

def hash_vector(vec):
    """
     * This is only used for the tests to verify agreement between vectors.
     * Note: due to the bitsize conversions this hash function has no guarantees about any properties.
     * Adapted from https://stackoverflow.com/a/27216842
    """
    assert len(vec.shape) == 1, "only accepts 1d vectors"
    seed = np.uint32(vec.shape[0])
    for i in vec:
        seed ^= np.uint32(i) + np.uint32(0x9e3779b9) + (seed << np.uint32(6)) + (seed >> np.uint32(2))
    return seed


def print_available_codes():
    print("Printing list of available LDPC mother matrices:")
    for i in range(1_000_000):
        try:
            code = ldpc.get_rate_adaptive_code(i)
            print(f"Code {i} maps {code.getNCols()} -> {code.get_n_rows_mother_matrix()}")
        except RuntimeError as e:  # `get_rate_adaptive_code` throws `RuntimeError` if no code availabel for ID.
            break
    print("\n")


def test_small():
    """
    How to use LDPC code with all manual settings
    """
    code = ldpc.get_code_small()

    key = np.array([1, 1, 1, 1, 0, 0, 0], dtype=np.uint8)
    print(f"{key=}")
    noisy_key = np.array([1, 1, 1, 1, 0, 0, 1], dtype=np.uint8)
    print(f"{noisy_key=}")

    syndrome = code.encode_no_ra(key)
    print(f"{syndrome=}")

    qber = 1 / 7
    vlog = np.log((1 - qber) / qber)
    llrs = np.array([vlog * (1. - 2. * noisy_bit) for noisy_bit in noisy_key], dtype=np.double)

    out = np.zeros(len(key), dtype=np.uint8)
    is_decoding_success: bool = code.decode_infer_rate(llrs, syndrome, out, 50, 100.)

    assert is_decoding_success, "Decoder did not converge!"
    assert np.all(out == key), "Decoder converged to wrong codeword!"
    print(f"{out=}")
    print("SUCCESS!\n\n")


def test_small_default():
    """
    How to use LDPC code with default settings
    """
    code = ldpc.get_code_small()

    key = np.array([1, 1, 1, 1, 0, 0, 0], dtype=np.uint8)
    print(f"{key=}")
    noisy_key = np.array([1, 1, 1, 1, 0, 0, 1], dtype=np.uint8)
    print(f"{noisy_key=}")

    syndrome = code.encode_no_ra(key)
    print(f"{syndrome=}")

    qber = 1 / 7
    out = np.zeros(len(key), dtype=np.uint8)
    is_decoding_success: bool = code.decode_default(noisy_key, syndrome, out, qber)

    assert is_decoding_success, "Decoder did not converge!"
    assert np.all(out == key), "Decoder converged to wrong codeword!"
    print(f"{out=}")
    print("SUCCESS!\n\n")


def test_encode_with_ra():
    code = ldpc.get_rate_adaptive_code(1)

    print(f"Code setting without ra: {code.getNCols()} -> {code.get_n_rows_after_rate_adaption()}")

    key = binary_symmetric_channel(np.zeros(code.getNCols(), dtype=np.uint8), 0.5)

    requested_syndrome_size = math.floor(0.95 * code.get_n_rows_mother_matrix())
    syndrome = code.encode_with_ra(key, requested_syndrome_size)
    assert len(syndrome) == requested_syndrome_size, "Syndrome does not match requested size"

    qber = 0.01
    noisy_key = binary_symmetric_channel(key, qber)

    corrected_noisy_key = np.zeros(len(key), dtype=np.uint8)
    is_decoding_success: bool = code.decode_default(noisy_key, syndrome, corrected_noisy_key, qber)
    assert is_decoding_success, "Decoder did not converge!"
    assert np.all(corrected_noisy_key == key), "Decoder converged to wrong codeword!"


def get_test_key(size):
    pattern = [0, 0, 0, 0, 0, 0, 0, 0,
               1, 1, 1, 1, 1, 1, 1, 1,
               0, 0, 0, 0, 0, 0, 0, 0,
               1, 1, 1, 1, 1, 1, 1, 1,
               0, 0, 0, 0, 0, 0, 0, 0,
               1, 1, 1, 1, 1, 1, 1, 1,
               0, 0, 0, 0, 0, 0, 0, 0,
               1, 1, 1, 1, 1, 1, 1, 1,
               0, 0, 0, 0, 0, 0, 0, 0,
               1, 1, 1, 1, 1, 1, 1, 1,
               0, 0, 0, 0, 0, 0, 0, 0,
               1, 1, 1, 1, 1, 1, 1, 1,
               0, 0, 0, 0, 0, 0, 0, 0,
               1, 1, 1, 1, 1, 1, 1, 1,
               0, 0, 0, 0, 0, 0, 0, 0,
               1, 1, 1, 1, 1, 1, 1, 1]
    return np.concatenate((pattern, np.zeros(shape=(size - len(pattern)), dtype=np.uint8))).astype(np.uint8)


def binary_symmetric_channel(input_bits, error_probability):
    random_values = np.random.rand(len(input_bits))
    error_mask = random_values < error_probability

    output_bits = np.logical_xor(input_bits, error_mask)

    return output_bits.astype(np.uint8)


def test_big():
    """
    Very very crude frame error rate simulation
    """
    code = ldpc.get_rate_adaptive_code(1)

    n_trials = 100
    n_failures = 0

    print(f"{code.getNCols()} -> {code.get_n_rows_after_rate_adaption()}")

    key = binary_symmetric_channel(np.zeros(code.getNCols(), dtype=np.uint8), 0.5)
    # print(f"{hash_vector(key)=}")

    syndrome = code.encode_no_ra(key)
    # assert hash_vector(syndrome) == 2814594723, \
    #     "value (from C++ code) for small matrix 6144 -> 2048 and input hash 1233938212"

    true_target_qber = 0.05
    for sim_idx in range(n_trials):
        noisy_key = binary_symmetric_channel(key, true_target_qber)
        # print(f"{hash_vector(noisy_key)=}")
        # print(f"errors: {len(noisy_key) - np.count_nonzero(noisy_key == key)} "
        #       f"({(len(noisy_key) - np.count_nonzero(noisy_key == key)) / len(noisy_key):.2%})")
        if sim_idx % (3_000_000 // code.getNCols()) == 1:
            print(f"\rSimulating frame error rate at BSC({true_target_qber:.2%}) ({sim_idx} / {n_trials}, "
                  f"current FER ~ {n_failures / sim_idx:.2E})", end="")
        estim_qber = 0.05
        corrected_noisy_key = np.zeros(len(key), dtype=np.uint8)
        is_decoding_success: bool = code.decode_default(noisy_key, syndrome, corrected_noisy_key, estim_qber)
        if not np.all(key == corrected_noisy_key):
            n_failures += 1
            if is_decoding_success:
                print("Reported decoding success despite incorrect result! (this should be very rare!)")
        elif not is_decoding_success:
            print("Reported decoding failure despite CORRECT RESULT. (This has to be a bug!)")
    print(f"\rSimulation done. Did {n_trials=} and {n_failures=} on BSC({true_target_qber:.2%}). "
          f"FER ~ {n_failures / n_trials:.2E}")


if __name__ == "__main__":
    print_available_codes()

    test_small()
    test_small_default()
    test_encode_with_ra()
    test_big()
