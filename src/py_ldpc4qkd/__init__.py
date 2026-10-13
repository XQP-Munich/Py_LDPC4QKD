from __future__ import annotations

from ._core import *

from dataclasses import dataclass, asdict, field
from importlib.metadata import version, PackageNotFoundError
from packaging.version import Version, InvalidVersion

import math
import warnings
import numpy as np
import numpy.typing as npt

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"


def binary_entropy(p: float) -> float:
    """
    Shannon binary entropy function
    """
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1")
    elif p == 0 or p == 1:
        return 0
    else:
        return -p * math.log(p, 2) - (1 - p) * math.log(1 - p, 2)


DEFAULT_INPUT_BLOCK_SIZE = 819200


@dataclass
class ECCodeSpec:
    """
    Specifies a concrete LDPC code.
    The LDPC matrix may be derived from a mother matrix by using rate adaption.
    """
    ecc_id: int
    ldpc_block_size: int
    syndrome_bits_per_block: int
    ecc_type: str
    # Version of this library that produced the spec.
    ldpc4qkd_version: str = field(default_factory=lambda: __version__)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ECCodeSpec":
        stored_version = data.get("ldpc4qkd_version")
        if stored_version is None:
            warnings.warn(
                f"Error correction request: no package version specified. Skipping comparison.",
                stacklevel=2,
            )
        else:
            try:
                is_newer = Version(stored_version) > Version(__version__)
            except InvalidVersion:
                warnings.warn(
                    f"Could not compare ECCodeSpec version ({stored_version!r}) against the running "
                    f"version ({__version__!r}); skipping compatibility check.",
                    stacklevel=2,
                )
            else:
                if is_newer:
                    warnings.warn(
                        f"Loading ECCodeSpec created by a newer version of py_ldpc4qkd ({stored_version}) "
                        f"than the one currently running ({__version__}). Loaded data may be incompatible.",
                        stacklevel=2,
                    )
        return cls(**data)

    @classmethod
    def select_suitable(cls, ch_param_estimate, input_block_size=DEFAULT_INPUT_BLOCK_SIZE) -> ECCodeSpec:
        """
        Selects an LDPC code and rate adaption based on requirements.
        Actual selection is done in C++.
        This could be improved further by taking into account more information,
            such as estimate uncertainty, or requirements about FER.
        :param ch_param_estimate: estimated parameter of binary symmetric channel
        :param input_block_size: size of input block. `None` uses the default (DEFAULT_INPUT_BLOCK_SIZE).
        :return: chosen ECCodeSpec
        """
        if input_block_size is None:
            input_block_size = DEFAULT_INPUT_BLOCK_SIZE

        choice = select_suitable_code(ch_param_estimate, input_block_size)
        if choice is None:
            raise NotImplementedError(
                f"Found no suitable code for given parameters. {ch_param_estimate=}, {input_block_size=}.")

        return cls(
            ecc_id=choice.code_id,
            ldpc_block_size=choice.ldpc_block_size,
            syndrome_bits_per_block=choice.syndrome_bits_per_block,
            ecc_type=choice.ecc_type,
        )

    def get_corresponding_code(self) -> RateAdaptiveCode:
        code: RateAdaptiveCode = get_rate_adaptive_code(self.ecc_id)
        assert code.getNCols() == self.ldpc_block_size, \
            f"Unexpected block size of associated id. {code.getNCols()=} != {self.ldpc_block_size=}"
        mother_matrix_syndrome_length = code.get_n_rows_mother_matrix()
        if mother_matrix_syndrome_length < self.syndrome_bits_per_block:
            raise NotImplementedError(f"Requested rate adaption increasing syndrome size "
                                      f"from {mother_matrix_syndrome_length=} to {self.syndrome_bits_per_block=}.")
        if mother_matrix_syndrome_length == self.syndrome_bits_per_block:
            return code
        else:
            # Need to do rate adaption!
            n_rate_adaption_steps = mother_matrix_syndrome_length - self.syndrome_bits_per_block
            code.set_rate(n_rate_adaption_steps)
            return code


def compute_syndrome_all_blocks(
        full_key: npt.NDArray[np.uint8],
        ecc_code_spec: ECCodeSpec) -> npt.NDArray[np.uint8]:
    """
    Split the key into blocks of the size that the code expects.
    Compute the syndrome of each block. Append leftover key bits at the end.
    Therefore, try to ensure that `len(full_key) // ecc_code_spec.ldpc_block_size` is small.

    :param full_key: 1-D array of bits
    :param ecc_code_spec: specification of an LDPC code
    :return: concatenated syndromes (1-D array of bits)
    """
    code = ecc_code_spec.get_corresponding_code()
    # Split the key into error-correction-blocks. Add left-overs at the end of the syndrome.
    single_ecc_block_size = code.getNCols()
    n_ecc_blocks = len(full_key) // single_ecc_block_size
    single_syndrome_block_size = code.get_n_rows_after_rate_adaption()

    leftover_key_size = len(full_key) % single_ecc_block_size
    full_syndrome_size = n_ecc_blocks * single_syndrome_block_size + leftover_key_size
    full_syndrome = np.zeros(n_ecc_blocks * single_syndrome_block_size + leftover_key_size, dtype=np.uint8)

    for i in range(n_ecc_blocks):
        current_key_block = full_key[i * single_ecc_block_size:(i + 1) * single_ecc_block_size]
        current_syndrome_block = code.encode_at_current_rate(current_key_block)
        assert current_syndrome_block.shape == (single_syndrome_block_size,), "Unexpected syndrome shape"
        full_syndrome[i * single_syndrome_block_size: (i + 1) * single_syndrome_block_size] = current_syndrome_block

    full_syndrome[-leftover_key_size:] = full_key[-leftover_key_size:]
    assert full_syndrome.shape == (full_syndrome_size,), "syndrome computation yielded unexpected shape"
    return full_syndrome


def decode_all_blocks(full_noisy_key: npt.NDArray[np.uint8], full_syndrome: npt.NDArray[np.uint8],
                      ecc_code_spec: ECCodeSpec, ch_param_estimate: float) -> npt.NDArray[np.uint8]:
    """
    Split the key and the syndrome into blocks of the sizes that the code expects.
    Perform error correction.
    Assigns leftover bits in the error-corrected key to left-over syndrome bits (see `compute_syndrome_all_blocks`).
    Therefore, try to ensure that `len(full_noisy_key) // ecc_code_spec.ldpc_block_size` is small.

    :param full_noisy_key: 1-D array of bits
    :param full_syndrome:  1-D array of bits
    :param ecc_code_spec: specification of an LDPC code
    :param ch_param_estimate: estimated bit error rate. Used to initialize decoding algorithm. Need not be very precise.
    :return: full error-corrected key. 1-D array of bits of same size as `full_noisy_key`.
    """
    code: RateAdaptiveCode = ecc_code_spec.get_corresponding_code()

    # Split the key into error-correction-blocks. Add left-overs at the end.
    single_ecc_block_size = code.getNCols()
    n_ecc_blocks = len(full_noisy_key) // single_ecc_block_size
    full_error_corrected_key = np.zeros(len(full_noisy_key), dtype=np.uint8)
    single_ecc_syndrome_size = code.get_n_rows_after_rate_adaption()

    for i in range(n_ecc_blocks):
        current_noisy_key_block = full_noisy_key[i * single_ecc_block_size:(i + 1) * single_ecc_block_size]
        current_syndrome_block = full_syndrome[i * single_ecc_syndrome_size:(i + 1) * single_ecc_syndrome_size]
        error_corrected_key_block = np.zeros(code.getNCols(), dtype=np.uint8)
        is_decoding_success: bool = code.decode_default(
            current_noisy_key_block, current_syndrome_block, error_corrected_key_block, ch_param_estimate)

        if not is_decoding_success:
            raise ValueError(f"Frame error: LDPC decoder did not converge. Is the {ch_param_estimate=} too small? "
                    "(This can always happen but should happen very rarely for correctly chosen parameters).")
        full_error_corrected_key[
            i * single_ecc_block_size:(i + 1) * single_ecc_block_size] = error_corrected_key_block

    # The leftover key, which does not fit in the block, is appended to the syndrome
    # This is somewhat inefficient, TODO could use combination of code sizes
    leftover_key_size = len(full_noisy_key) % single_ecc_block_size
    full_error_corrected_key[-leftover_key_size:] = full_syndrome[-leftover_key_size:]
    assert full_error_corrected_key.shape == full_noisy_key.shape, "Error-corrected key has unexpected shape"
    return full_error_corrected_key
