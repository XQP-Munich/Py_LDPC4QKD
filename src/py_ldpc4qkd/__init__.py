from __future__ import annotations

from ._core import *

from dataclasses import dataclass, asdict
from importlib.metadata import version, PackageNotFoundError

import math
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

    def to_dict(self) -> dict:
        metadata = {"ldpc4qkd_version": __version__}
        return asdict(self) | metadata

    @classmethod
    def from_dict(cls, data: dict) -> "ECCodeSpec":
        return cls(**data)

    @classmethod
    def select_suitable(cls, ch_param_estimate, ) -> ECCodeSpec:
        """
        Selects an LDPC code and rate adaption based on requirements.
        This could be improved a lot by taking into account more information,
            such as estimate uncertainty, or requirements about FER, or block size requirements.
        :param ch_param_estimate: estimated parameter of binary symmetric channel
        :return: chosen ECCodeSpec
        """
        ECC_TYPE = "QC-LDPC Protograph-specific-XOR"
        if ch_param_estimate <= 0:
            raise ValueError("ch_param_estimate estimate must be > 0")
        elif ch_param_estimate < 0.01:
            code_id = 1
            f = 6
        elif ch_param_estimate < 0.04:
            code_id = 1
            f = 3
        elif ch_param_estimate < 0.049:
            code_id = 1
            f = 20
        elif ch_param_estimate < 0.07:
            code_id = 4
            f = 20
        elif ch_param_estimate < 0.092:
            code_id = 4
            f = 2
        else:
            raise NotImplementedError("No available code is suitable for requested parameters!")

        code: RateAdaptiveCode = get_rate_adaptive_code(code_id)

        syndrome_bits_per_block = min(code.get_n_rows_mother_matrix(),
                                      math.floor(code.getNCols() * binary_entropy(ch_param_estimate) * f))

        return cls(
            ecc_id=code_id,
            ldpc_block_size=code.getNCols(),
            syndrome_bits_per_block=syndrome_bits_per_block,
            ecc_type=ECC_TYPE,
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
            raise ValueError("LDPC decoder did not converge. Is the ch_param estimate too small?")
        full_error_corrected_key[
            i * single_ecc_block_size:(i + 1) * single_ecc_block_size] = error_corrected_key_block

    leftover_key_size = len(full_noisy_key) % single_ecc_block_size
    full_error_corrected_key[-leftover_key_size:] = full_syndrome[-leftover_key_size:]
    assert full_error_corrected_key.shape == full_noisy_key.shape, "Error-corrected key has unexpected shape"
    return full_error_corrected_key
