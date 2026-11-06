"""Block-encoding subroutines."""

from nwqlib.subroutines.block_encoding.banded import (
    BandSpecification,
    build_banded_block_encoding,
)
from nwqlib.subroutines.block_encoding.core import (
    BlockEncoding,
    BlockEncodingPlan,
    block_encoding_query_cx,
    block_encoding_top_left,
    build_block_encoding,
    build_block_encoding_from_plan,
    build_block_encoding_query,
    plan_block_encoding,
)
from nwqlib.subroutines.block_encoding.nodes import (
    UniformNodeRecipe,
    build_diagonal_table_encoding,
    build_diagonal_table_query,
    build_uniform_node_encoding,
    build_uniform_node_query,
    uniform_node_labels,
    uniform_node_recipes,
    uniform_node_values,
)
from nwqlib.subroutines.block_encoding.joint import (
    AccessSummary,
    ComplexDiagonalTable,
    DiagonalTable,
    JointEncodingPlan,
    access_summary,
    build_access_query,
    build_joint_encoding,
    plan_block_encoding_product,
    plan_convex_contraction,
    plan_joint_generator,
    plan_single_a_generator,
)
from nwqlib.subroutines.block_encoding.registry import BLOCK_ENCODING_IMPLEMENTATIONS

__all__ = [
    "AccessSummary",
    "BLOCK_ENCODING_IMPLEMENTATIONS",
    "BandSpecification",
    "BlockEncoding",
    "BlockEncodingPlan",
    "ComplexDiagonalTable",
    "DiagonalTable",
    "JointEncodingPlan",
    "access_summary",
    "build_access_query",
    "build_joint_encoding",
    "plan_block_encoding_product",
    "plan_convex_contraction",
    "plan_joint_generator",
    "plan_single_a_generator",
    "UniformNodeRecipe",
    "block_encoding_query_cx",
    "block_encoding_top_left",
    "build_banded_block_encoding",
    "build_block_encoding",
    "build_block_encoding_from_plan",
    "build_block_encoding_query",
    "build_diagonal_table_encoding",
    "build_diagonal_table_query",
    "build_uniform_node_encoding",
    "build_uniform_node_query",
    "projector_complement_matrix",
    "plan_block_encoding",
    "uniform_node_labels",
    "uniform_node_recipes",
    "uniform_node_values",
]

from nwqlib.subroutines.dense_matrices import projector_complement_matrix
