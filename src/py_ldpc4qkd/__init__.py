from __future__ import annotations

from ._core import *

from dataclasses import dataclass, asdict
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"


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
    def select_suitable(cls, qber_estimate, ) -> ECCodeSpec:
        """
        Selects an LDPC code and rate adaption based on requirements.
        This choice method could still be improved a lot by taking into account more information,
            such as QBER estimate uncertainty, or requirements about FER, or block size requirements.
        :param qber_estimate: estimated QBER
        :return: chosen ECCodeSpec
        """
        ECC_TYPE = "QC-LDPC Protograph-specific-XOR"
        if qber_estimate <= 0:
            raise ValueError("QBER estimate must be > 0")
        elif qber_estimate < 0.05:
            code_id = 1
        elif qber_estimate < 0.1:
            code_id = 4
        else:
            raise NotImplementedError("No available code is suitable for requested parameters!")

        code: RateAdaptiveCode = get_rate_adaptive_code(code_id)
        return cls(
            ecc_id=code_id,
            ldpc_block_size=code.getNCols(),
            syndrome_bits_per_block=code.get_n_rows_mother_matrix(),
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
