import warnings
import math

import numpy as np
import py_ldpc4qkd as ldpc

np.random.seed(42)


def get_code_small():
    """
    We use this matrix as an example:
       H =  [1 0 1 0 1 0 1
             0 1 1 0 0 1 1
             0 0 0 1 1 1 1]

    To use it, we must convert H to compressed sparse column (CSC) storage.
    """
    colptr = [0, 1, 2, 4, 5, 7, 9, 12]
    row_idx = [0, 1, 0, 1, 2, 0, 2, 1, 2, 0, 1, 2]
    return ldpc.RateAdaptiveCode(colptr, row_idx)


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


def test_small():
    """
    How to use LDPC code with all manual settings
    """
    code = get_code_small()

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
    code = get_code_small()

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


def print_available_codes():
    for i in range(1_000_000):
        try:
            code = ldpc.get_rate_adaptive_code(i)
            print(f"Code {i} maps {code.getNCols()} -> {code.get_n_rows_mother_matrix()}")
        except RuntimeError as e:  # `get_rate_adaptive_code` throws `RuntimeError` if no code availabel for ID.
            print("\n")
            return i


def test_encode_with_ra():
    code = ldpc.get_rate_adaptive_code(1)

    print(f"Code setting without ra: {code.getNCols()} -> {code.get_n_rows_after_rate_adaption()}")

    key = binary_symmetric_channel(np.zeros(code.getNCols(), dtype=np.uint8), 0.5)

    requested_syndrome_size = math.floor(0.9 * code.get_n_rows_mother_matrix())
    syndrome = code.encode_with_ra(key, requested_syndrome_size)
    assert len(syndrome) == requested_syndrome_size, "Syndrome does not match requested size"

    ch_param = 0.03
    noisy_key = binary_symmetric_channel(key, ch_param)

    corrected_noisy_key = np.zeros(len(key), dtype=np.uint8)
    is_decoding_success: bool = code.decode_default(noisy_key, syndrome, corrected_noisy_key, ch_param)
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


def h2inv(target_entropy, lo=1e-6, hi=0.5):
    """Bisection inverse of ldpc.binary_entropy, for p in (0, 0.5)."""
    for _ in range(40):
        mid = (lo + hi) / 2
        if ldpc.binary_entropy(mid) < target_entropy:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def test_print_version():
    print(ldpc.__version__)


def test_big(code_id, n_trials=20, safety_margin=1.3):
    """
    Very very crude frame error rate simulation.
    Targets a QBER derived from this code's own (mother) rate via h2inv (with a safety margin).
    """
    code = ldpc.get_rate_adaptive_code(code_id)

    n_failures = 0

    print(f"{code.getNCols()} -> {code.get_n_rows_after_rate_adaption()}")

    rate = code.get_n_rows_mother_matrix() / code.getNCols()
    true_target_qber = h2inv(rate / safety_margin)

    key = binary_symmetric_channel(np.zeros(code.getNCols(), dtype=np.uint8), 0.5)
    syndrome = code.encode_no_ra(key)

    for _ in range(n_trials):
        noisy_key = binary_symmetric_channel(key, true_target_qber)
        corrected_noisy_key = np.zeros(len(key), dtype=np.uint8)
        is_decoding_success: bool = code.decode_default(noisy_key, syndrome, corrected_noisy_key, true_target_qber)
        if not np.all(key == corrected_noisy_key):
            n_failures += 1
            if is_decoding_success:
                print("Reported decoding success despite incorrect result! (this should be very rare!)")
        elif not is_decoding_success:
            print("Reported decoding failure despite CORRECT RESULT. (This has to be a bug!)")
    print(f"Simulation done. Did {n_trials=} and {n_failures=} on BSC({true_target_qber:.2%}). "
          f"FER ~ {n_failures / n_trials:.2E}")


def test_with_block_splitting(ch_param=0.049):
    ecc_code_spec = ldpc.ECCodeSpec.select_suitable(ch_param_estimate=ch_param)

    code = ecc_code_spec.get_corresponding_code()

    print(f"Code setting without ra: {code.getNCols()} -> {code.get_n_rows_after_rate_adaption()}")

    block_size = 2 * code.getNCols() + 11
    key = binary_symmetric_channel(np.zeros(block_size, dtype=np.uint8), 0.5)

    syndrome = ldpc.compute_syndrome_all_blocks(key, ecc_code_spec)

    noisy_key = binary_symmetric_channel(key, ch_param)
    lrate = len(syndrome) / len(noisy_key)
    f = lrate / ldpc.binary_entropy(ch_param)
    print(f"Correcting {len(noisy_key)} bits using full syndrome {len(syndrome)}, {lrate=:.4f}. {ch_param=}. {f=:.3f}")
    corrected_noisy_key = ldpc.decode_all_blocks(noisy_key, syndrome, ecc_code_spec, ch_param)
    assert np.all(corrected_noisy_key == key), "Decoder converged to wrong codeword!"


def test_819k_default_fer(qbers=None, n_trials=100):
    """
    Estimates FER (like test_big: single block, no block splitting) for whichever 819k code
    ECCodeSpec.select_suitable picks by default at each QBER, using code.decode_default -- the
    same decode path (and iteration budget) real callers get, not a hand-tuned one.
    """
    if qbers is None:
        qbers = [q / 1000 for q in range(5, 41, 5)]  # 0.005, 0.010, ..., 0.040

    for qber in qbers:
        ecc_code_spec = ldpc.ECCodeSpec.select_suitable(ch_param_estimate=qber)
        code = ecc_code_spec.get_corresponding_code()
        n_cols = code.getNCols()
        rate = code.get_n_rows_after_rate_adaption() / n_cols

        key = binary_symmetric_channel(np.zeros(n_cols, dtype=np.uint8), 0.5)
        syndrome = code.encode_at_current_rate(key)

        n_failures = 0
        for _ in range(n_trials):
            noisy_key = binary_symmetric_channel(key, qber)
            corrected = np.zeros(n_cols, dtype=np.uint8)
            success = code.decode_default(noisy_key, syndrome, corrected, qber)
            if not success or not np.all(corrected == key):
                n_failures += 1

        print(f"QBER={qber:.3%} ecc_id={ecc_code_spec.ecc_id} rate={rate:.4f} "
              f"n_trials={n_trials} n_failures={n_failures} FER~{n_failures / n_trials:.2e}")


if __name__ == "__main__":
    test_print_version()

    n_codes = print_available_codes()

    test_small()
    test_small_default()
    test_encode_with_ra()

    [test_with_block_splitting(q / 1000) for q in range(5, 90, 10)]

    # test_big at least once per distinct N (matrix column count)
    # of the 819k-block codes (ids 6-14) all share N=819200, so testing each one adds no new
    # column-size coverage, just redundant runtime.
    first_id_for_n = {}
    for i in range(n_codes):
        n_cols = ldpc.get_rate_adaptive_code(i).getNCols()
        first_id_for_n.setdefault(n_cols, i)
    for i in first_id_for_n.values():
        test_big(i)
