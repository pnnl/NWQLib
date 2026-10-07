"""Block-encoding subroutines."""

from nwqlib.subroutines.block_encoding.banded import (
    BandSpecification,
    build_banded_block_encoding,
)
from nwqlib.subroutines.block_encoding.core import (
    BlockEncoding,
    BlockEncodingPlan,
    block_encoding_top_left,
    build_block_encoding,
    build_block_encoding_from_plan,
    plan_block_encoding,
)
from nwqlib.subroutines.block_encoding.registry import BLOCK_ENCODING_IMPLEMENTATIONS

__all__ = [
    "BLOCK_ENCODING_IMPLEMENTATIONS",
    "BandSpecification",
    "BlockEncoding",
    "BlockEncodingPlan",
    "block_encoding_top_left",
    "build_banded_block_encoding",
    "build_block_encoding",
    "build_block_encoding_from_plan",
    "projector_complement_matrix",
    "plan_block_encoding",
]

from nwqlib.subroutines.dense_matrices import projector_complement_matrix
