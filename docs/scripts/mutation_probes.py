#!/usr/bin/env python3
"""Run scientific mutation probes and require every probe to be killed.

The deduplicated selected tests must pass on unmodified source before any edit.
Each probe then edits one source file, runs its targeted pytest selection, and
restores the original bytes before continuing. Unavailable evidence and
infrastructure errors never count as kills or a successful campaign.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_QUALNAME = "<module>"
# Bound on one pytest run. A mutant can make a selected test loop, and
# without a bound the campaign would never finish and would leave the mutated
# source on disk when its process is killed. A run past the bound is an
# infrastructure error, and _run_probe restores the source before the
# campaign stops. The baseline runs all 205 selected nodes of the 274 probes
# at once. With 143 nodes it took 40 s on an Apple M3 Max with Python
# 3.12.14, so the bound leaves room for a slower host and for a mutant that
# slows a test without making it loop.
PYTEST_TIMEOUT_SECONDS = 600


def _module_path(module: str) -> Path:
    parts = module.split(".")
    if parts[0] != "nwqlib":
        raise RuntimeError(f"module {module!r} is not under src/nwqlib")
    base = ROOT / "src" / Path(*parts)
    if base.is_dir():
        return base / "__init__.py"
    return base.with_suffix(".py")


@dataclass(frozen=True)
class Replacement:
    module: str
    qualname: str
    old: str
    new: str

    @property
    def path(self) -> Path:
        return _module_path(self.module)


@dataclass(frozen=True)
class ResolvedReplacement:
    replacement: Replacement
    path: Path
    start: int
    end: int
    count: int


@dataclass(frozen=True)
class Probe:
    name: str
    replacement: Replacement
    pytest_args: tuple[str, ...]
    # Imports needed by the targeted tests, checked in the target interpreter
    # before the baseline. Missing extras make the campaign unavailable.
    requires: tuple[str, ...] = ()

    @property
    def path(self) -> Path:
        return self.replacement.path


def single_replacement_probe(
    *,
    name: str,
    module: str,
    qualname: str,
    old: str,
    new: str,
    pytest_args: tuple[str, ...],
    requires: tuple[str, ...] = (),
) -> Probe:
    """Build a probe containing exactly one replacement.

    Args:
        name: Stable probe name used in gate output and selection.
        module: Qualified target module below ``src/nwqlib``.
        qualname: Qualified definition containing the target snippet.
        old: Exact source snippet to replace.
        new: Mutated source snippet.
        pytest_args: Targeted pytest node IDs that must kill the mutation.
        requires: Optional imports required before running the probe.

    Returns:
        Probe with one replacement and the supplied test contract.
    """

    return Probe(
        name=name,
        replacement=Replacement(module=module, qualname=qualname, old=old, new=new),
        pytest_args=pytest_args,
        requires=requires,
    )


PROBES = (
    single_replacement_probe(
        name="lanczos_even_readout_sign",
        module="nwqlib.algorithms.lanczos.readout", qualname="signed_outcomes",
        old="        return np.where(index == 0, 1, -1).astype(np.int8)\n",
        new="        return np.where(index == 0, -1, 1).astype(np.int8)\n",
        pytest_args=("tests/test_lanczos_circuits.py::test_shared_walk_metadata_counts_and_padding_readout",),
    ),
    single_replacement_probe(
        name="lanczos_gershgorin_trailing_empty_row",
        module="nwqlib.algorithms._eigen_inputs", qualname="gershgorin_frame",
        old="            radius[nonempty] = np.add.reduceat(re, a.indptr[nonempty])\n",
        new="            radius = np.add.reduceat(re, a.indptr[:-1])\n",
        pytest_args=("tests/test_lanczos.py::test_classical_gershgorin_frame_encloses_the_spectrum_with_empty_rows",),
    ),
    single_replacement_probe(
        name="lanczos_chebyshev_product_factor",
        module="nwqlib.algorithms.lanczos.numerical", qualname="_projected_matrices",
        old="    shifted = shifted / 4\n", new="    shifted = shifted / 2\n",
        pytest_args=('tests/test_lanczos.py::test_signed_centered_pencil_from_independent_moments',),
    ),
    single_replacement_probe(
        name='lanczos_kept_projector_rotation_drop',
        module="nwqlib.algorithms.lanczos.numerical", qualname="_moment_energy_derivatives",
        old="        metric_gradient += rotation + rotation.conj().T\n",
        new="        metric_gradient += 0 * (rotation + rotation.conj().T)\n",
        pytest_args=('tests/test_lanczos.py::test_regularized_sensitivity_rotates_the_kept_projector',),
    ),
    single_replacement_probe(
        name="lanczos_trajectory_view_inverse_drop",
        module="nwqlib.algorithms.lanczos.method", qualname="_trajectory_program",
        old='    experiment = Experiment(name="trajectory", setting="trajectory", observation=observation)\n',
        new=('    first = observation.positions[0]\n'
             '    observation = observation.revise(positions=(\n'
             '        first.revise(view=first.view.revise(inverse="omitted_inverse")), *observation.positions[1:]))\n'
             '    program = program.revise(definitions=program.definitions + (\n'
             '        Definition(id="omitted_inverse", node=Sequence(children=())),))\n'
             '    experiment = Experiment(name="trajectory", setting="trajectory", observation=observation)\n'),
        pytest_args=("tests/test_lanczos_circuits.py::test_native_signed_centered_two_qubit_public_chain",),
    ),
    single_replacement_probe(
        name="lanczos_point_degree_by_order",
        module="nwqlib.algorithms.lanczos.method", qualname="Lanczos.statistics",
        old="                setting = settings[chunk.experiment, chunk.point]\n",
        new=("                setting = (plan.reconstruction.settings[len(grouped)] if chunk.point is not None\n"
             "                           else settings[chunk.experiment, chunk.point])\n"),
        pytest_args=("tests/test_lanczos.py::test_missing_moment_is_partial_and_identity_needs_no_solve",),
    ),
    single_replacement_probe(
        name="inverse_native_multiplexer_table_drop",
        module="nwqlib.subroutines.qiskit_compat", qualname="inverse_realized_gate",
        old="    if isinstance(gate, ControlledGate):\n        native_ucg = False\n    if isinstance(gate, UnitaryGate):\n",
        new="    native_ucg = False\n    if isinstance(gate, UnitaryGate):\n",
        pytest_args=("tests/test_lanczos_circuits.py::test_native_signed_centered_two_qubit_public_chain",),
    ),
    single_replacement_probe(
        name="inverse_native_multiplexer_conjugation_drop",
        module="nwqlib.subroutines.qiskit_compat", qualname="inverse_realized_gate",
        old="        adjoint_table = [matrix.conj().T.copy() for matrix in gate.params]\n",
        new="        adjoint_table = [matrix.T.copy() for matrix in gate.params]\n",
        pytest_args=("tests/test_lanczos_circuits.py::test_native_signed_centered_two_qubit_public_chain",),
    ),
    single_replacement_probe(
        name="inverse_controlled_base_keeps_native_tables",
        module="nwqlib.subroutines.qiskit_compat", qualname="inverse_realized_gate",
        old="        base_inverse = inverse_realized_gate(gate.base_gate, native_ucg=False)\n",
        new="        base_inverse = inverse_realized_gate(gate.base_gate)\n",
        pytest_args=("tests/test_qsp_evolution.py::test_qsp_evolution_synthesizes_each_child_multiplexer_once",),
    ),
    single_replacement_probe(
        name="qsp_evolution_child_inverse_keeps_native_tables",
        module="nwqlib.subroutines.qsp.evolution", qualname="build_qsp_evolution_encoding",
        old="    block_gates = _block_gate_pair(encoding, native_ucg=False)\n",
        new="    block_gates = _block_gate_pair(encoding)\n",
        pytest_args=("tests/test_qsp_evolution.py::test_qsp_evolution_synthesizes_each_child_multiplexer_once",),
    ),
    single_replacement_probe(
        name="public_phase_gate_native_payload_collision",
        module="nwqlib.subroutines._multiplexors",
        qualname="append_control_diagonal_phases",
        old='    gate = Gate("nwqlib_phase_diagonal", len(controls), list(padded))\n',
        new='    gate = Gate("diagonal", len(controls), list(padded))\n',
        pytest_args=(
            "tests/test_backend_baseline.py::"
            "test_public_preparation_circuit_preserves_amplitudes_on_native_aer",
        ),
    ),
    single_replacement_probe(
        name="qcels_unitary_phase_sign",
        module="nwqlib.algorithms.qpe.numerical", qualname="qcels",
        old="    value = angle_principal(-theta_hat) if tau is None else theta_hat/tau\n",
        new="    value = angle_principal(theta_hat) if tau is None else theta_hat/tau\n",
        pytest_args=("tests/test_qpe_workflow.py::test_qcels_nonzero_phase_and_symmetric_two_component_center",),
    ),
    single_replacement_probe(
        name="spe_cdf_phase_sign",
        module="nwqlib.algorithms.qpe.numerical", qualname="spe_cdf",
        old='        cdf += (factor*count)*np.imag(np.exp(1j*power*coordinates)*sample)\n',
        new='        cdf += (factor*count)*np.imag(np.exp(-1j*power*coordinates)*sample)\n',
        pytest_args=("tests/test_qpe_workflow.py::test_spe_fourier_estimator_matches_independent_chebyshev_filter",),
    ),
    single_replacement_probe(
        name="spe_cdf_target_threshold",
        module="nwqlib.algorithms.qpe.numerical", qualname="spe",
        old='    crossing = np.flatnonzero(cdf >= overlap_lower_bound/2)\n',
        new='    crossing = np.flatnonzero(cdf >= overlap_lower_bound)\n',
        pytest_args=("tests/test_qpe_workflow.py::test_spe_cdf_targets_the_low_energy_jump_of_a_mixed_spectrum",),
    ),
    # Saving is where a Result's observation meets its receipt, and loading
    # does not repeat the join. A chunk acquired under another realization
    # would be analyzed with parameters it was not acquired with.
    single_replacement_probe(
        name="saved_observation_realization_join_drop",
        module="nwqlib.saved_evidence",
        qualname="_validate_recorded_data",
        old="                    or chunk.realization_id != receipt.realization_id or not chunk.declares_readout_of(receipt)\n",
        new="                    or not chunk.declares_readout_of(receipt)\n",
        pytest_args=(
            "tests/test_saved_evidence.py::"
            "test_save_refuses_an_observation_that_differs_from_its_attempt_or_receipt[realization]",
        ),
    ),
    # A population other than the receipt's (for example a native-conditioned
    # sample saved as unconditional) changes what the counts estimate.
    single_replacement_probe(
        name="saved_observation_population_join_drop",
        module="nwqlib.saved_evidence",
        qualname="_validate_recorded_data",
        old=("                    or chunk.classical_layout != receipt.classical_layout\n"
             "                    or chunk.population != receipt.population):\n"),
        new="                    or chunk.classical_layout != receipt.classical_layout):\n",
        pytest_args=(
            "tests/test_saved_evidence.py::"
            "test_save_refuses_an_observation_that_differs_from_its_attempt_or_receipt[population]",
        ),
    ),
    single_replacement_probe(
        name="qpe_product_formula_power_time_drop",
        module="nwqlib.subroutines.trotterization.error_budget", qualname="_select_common_step",
        old='            p*dt, hi, ci, counts[p],\n',
        new='            dt, hi, ci, counts[p],\n',
        pytest_args=("tests/test_qpe_efficiency_contract.py::test_common_grid_powers_record_the_checked_emitted_prefix_bounds",),
    ),
    single_replacement_probe(
        name="qpe_pauli_pruned_mass_half",
        module="nwqlib.subroutines.trotterization.error_budget", qualname="_emitted_parts",
        old='    parts = dict(trotter=W*steps*abs(h)**3, pruning=loss*abs(t),\n',
        new='    parts = dict(trotter=W*steps*abs(h)**3, pruning=loss*abs(t)/2,\n',
        pytest_args=("tests/test_qpe_efficiency_contract.py::test_pruning_and_signed_power_time_owned_once",),
    ),
    single_replacement_probe(
        name="qpe_repeat_drop_multiplicity",
        module="nwqlib.algorithms.qpe.method", qualname="_quantum_program",
        old='                                           Repeat(body=block_call(selection.step), count=delta)))\n',
        new='                                           Repeat(body=block_call(selection.step), count=1)))\n',
        pytest_args=("tests/test_qpe_efficiency_contract.py::test_qpe_compact_repeat_and_shot_multiplicity",),
    ),
    # A Suzuki step stores one controlled rotation per term in each sweep of
    # the symmetric formula. Counting one sweep halves the logical operations.
    single_replacement_probe(
        name="qpe_suzuki_step_operations_one_sweep",
        module="nwqlib.algorithms.qpe.powers", qualname="_step_block",
        old='        operations=2 * len(terms),\n',
        new='        operations=len(terms),\n',
        pytest_args=("tests/test_qpe_efficiency_contract.py::test_qpe_compact_repeat_and_shot_multiplicity",),
    ),
    # Each term gives two controlled leaves of 2*w CX. Pricing one leaf per
    # term halves the step law, which the transpiled step count exposes.
    single_replacement_probe(
        name="qpe_suzuki_step_cx_one_leaf_per_term",
        module="nwqlib.algorithms.qpe.powers", qualname="suzuki_step_cx",
        old='    return 4 * sum(len(label) - label.count("I") for label, _ in terms)\n',
        new='    return 2 * sum(len(label) - label.count("I") for label, _ in terms)\n',
        pytest_args=("tests/test_qpe_efficiency_contract.py::test_suzuki_step_cx_law_counts_the_built_step",),
    ),
    single_replacement_probe(
        name="qpe_controlled_suzuki_half_angle_x1_01",
        module="nwqlib.algorithms.qpe.powers", qualname="rotation_schedule",
        old='    half_step = tuple((label, 0.5 * step_time * coefficient) for label, coefficient in terms)\n',
        new='    half_step = tuple((label, 0.505 * step_time * coefficient) for label, coefficient in terms)\n',
        pytest_args=('tests/test_qpe_efficiency_contract.py::test_suzuki_angles_match_independent_two_level_product', 'tests/test_qpe_selected.py::test_native_random_pauli_quadratures_share_each_power'),
    ),
    single_replacement_probe(
        name="rwpe_gaussian_mean_scale_half",
        module="nwqlib.algorithms.qpe.numerical", qualname="rwpe_update",
        old='    updated = mean+(1-2*datum)*std/sqrt(e)\n',
        new='    updated = mean+.5*(1-2*datum)*std/sqrt(e)\n',
        pytest_args=("tests/test_qpe_workflow.py::test_rwpe_one_bit_update_matches_independent_gaussian_integration",),
    ),
    single_replacement_probe(
        name="qhd_imaginary_component_discard",
        module="nwqlib.algorithms.qhd.potential",
        qualname="coerce_real_scalar",
        old="    if scalar.imag != 0.0:\n",
        new="    if abs(scalar.imag) > 1.0e-12:\n",
        pytest_args=("tests/test_qhd_workflow.py::test_objective_domain_precedes_pruning",),
    ),
    single_replacement_probe(
        name="qhd_non_candidate_marginals_swapped",
        module="nwqlib.algorithms.qhd.method",
        qualname="_summarize",
        old=("                block = ordered[ends[i]:ends[i + 1]]\n"
             "                marginals[axis, i] = (sum(map(int, block.flat)) / scale if counts\n"),
        new=("                block = ordered[ends[i]:ends[i + 1]]\n"
             "                marginals[axis, 1 - i if i < 2 else i] = (sum(map(int, block.flat)) / scale if counts\n"),
        pytest_args=("tests/test_qhd_workflow.py::test_observed_candidate_partial_and_invalid_population_without_reference",),
    ),
    single_replacement_probe(
        name="qhd_non_candidate_marginals_swapped_dense",
        module="nwqlib.algorithms.qhd.method",
        qualname="_summarize",
        old=("                block = grid_weights[tuple(cut)]\n"
             "                marginals[axis, i] = (sum(map(int, block.flat)) / scale if counts\n"),
        new=("                block = grid_weights[tuple(cut)]\n"
             "                marginals[axis, 1 - i if i < 2 else i] = (sum(map(int, block.flat)) / scale if counts\n"),
        pytest_args=("tests/test_qhd_readout.py::"
                     "test_the_array_summary_matches_a_scalar_reference_with_ties_and_a_leading_zero_cell",),
    ),
    # A phase on each register's SDK state preparation changes the absolute
    # phase that the exact-amplitude comparison checks.
    single_replacement_probe(
        name="qhd_sdk_preparation_register_phase",
        module="nwqlib.algorithms.qhd.initial_state",
        qualname="append_register_state_preparation",
        old="        circuit.append(StatePreparation(onehot_register_vector(vector)), qubits)\n",
        new="        circuit.append(StatePreparation(1j * onehot_register_vector(vector)), qubits)\n",
        pytest_args=(
            "tests/test_qhd_workflow.py::"
            "test_native_preparation_equals_the_classical_initial_vector[gaussian-qiskit_state_preparation]",
        ),
    ),
    # The chain angle uses the remaining tail norm, cos(theta_m/2) =
    # alpha_(m-1)/r_(m-1). The full norm 1 in its place prepares another state.
    single_replacement_probe(
        name="qhd_chain_angle_full_norm",
        module="nwqlib.algorithms.qhd.initial_state",
        qualname="chain_selection",
        old="        theta = 2.0 * math.atan2(tails[m], alpha[m - 1])\n",
        new="        theta = 2.0 * math.acos(alpha[m - 1])\n",
        pytest_args=(
            "tests/test_qhd_workflow.py::"
            "test_native_preparation_equals_the_classical_initial_vector[gaussian-structured]",
        ),
    ),
    # A resource-only Plan must not charge the chain's CX.
    single_replacement_probe(
        name="qhd_resource_only_charges_chain_cx",
        module="nwqlib.algorithms.qhd.method",
        qualname="QHD._select_native",
        old='        if self.initial_state_preparation == "structured":\n',
        new='        if self.initial_state_preparation != "qiskit_state_preparation":\n',
        pytest_args=(
            "tests/test_qhd_workflow.py::"
            "test_resource_only_initial_state_counts_the_evolution_blocks_alone",
        ),
    ),
    # The classical K = 39 masses exceed one by about 3e-12, beyond the fixed
    # floor but inside the derived expm_multiply window.
    single_replacement_probe(
        name="qhd_host_mass_window_floor",
        module="nwqlib.algorithms.qhd.method",
        qualname="_host_summary",
        old="    window = _host_probability_window(plan)\n",
        new="    window = NUMERICAL_RELATION_RTOL\n",
        pytest_args=(
            "tests/test_qhd_workflow.py::"
            "test_classical_mass_window_follows_the_expm_multiply_generators[schrodinger]",
        ),
    ),
    single_replacement_probe(
        name="qhd_saved_masses_ignore_kernel_window",
        module="nwqlib.algorithms.qhd.records",
        qualname="QHDAnalysis.validate_analysis_masses",
        old='                       if binding.parameter == "probability_window")\n',
        new='                       if binding.parameter == "unrecorded_window")\n',
        pytest_args=(
            "tests/test_qhd_workflow.py::"
            "test_classical_mass_window_follows_the_expm_multiply_generators[schrodinger]",
        ),
    ),
    single_replacement_probe(
        name="expm_multiply_window_without_term_growth",
        module="nwqlib._validation",
        qualname="expm_multiply_call_roundoff",
        old="    growth = math.exp(theta)\n",
        new="    growth = 1.0\n",
        pytest_args=(
            "tests/test_qhd_corrections.py::test_expm_multiply_window_follows_its_stated_relation",
        ),
    ),
    # The most probable point is the lexicographically smallest index within
    # the derived tie window of the computed maximum. Taking the largest
    # index, or comparing exactly, selects another point in the supplied
    # gap-below-window case.
    single_replacement_probe(
        name="qhd_most_probable_largest_index",
        module="nwqlib.algorithms.qhd.method",
        qualname="_summarize",
        old="    first = int(np.argmax(tied))\n",
        new="    first = int(tied.size - 1 - np.argmax(tied[::-1]))\n",
        pytest_args=("tests/test_qhd_readout.py::test_tie_window_is_relative_to_the_computed_maximum",),
    ),
    single_replacement_probe(
        name="qhd_most_probable_exact_comparison",
        module="nwqlib.algorithms.qhd.method",
        qualname="_summarize",
        old="    tie = 0 if window is None else window\n",
        new="    tie = 0\n",
        pytest_args=("tests/test_qhd_readout.py::test_tie_window_is_relative_to_the_computed_maximum",),
    ),
    # The kernel publishes the selected probability and its deficit as two
    # labeled scalars. Swapping them records the zero deficit as the
    # probability, which the Result's population checks reject, so the
    # eigendecomposition oracle test fails.
    single_replacement_probe(
        name="qhd_most_probable_kernel_scalars_swapped",
        module="nwqlib.algorithms.qhd.method",
        qualname="_execute_theory",
        old=('        "most_probable_probability": result["most_probable_probability"],\n'
             '        "most_probable_deficit": result["most_probable_deficit"],\n'),
        new=('        "most_probable_probability": result["most_probable_deficit"],\n'
             '        "most_probable_deficit": result["most_probable_probability"],\n'),
        pytest_args=(
            "tests/test_qhd_readout.py::test_most_probable_point_matches_restricted_eigendecomposition",
        ),
    ),
    # Expanding (x - 1e7)**2 about the origin loses its grid values to
    # cancellation, and so does evaluating the centered tables at x itself.
    single_replacement_probe(
        name="qhd_expansion_about_the_origin",
        module="nwqlib.algorithms.qhd.objective",
        qualname="expansion_centers",
        old="    return tuple(min(max(0.0, float(lower)), float(upper)) for lower, upper in bounds)\n",
        new="    return tuple(0.0 for _lower, _upper in bounds)\n",
        pytest_args=("tests/test_qhd_workflow.py::test_objective_far_from_the_origin_keeps_its_grid_values",),
    ),
    single_replacement_probe(
        name="qhd_centered_table_at_uncentered_point",
        module="nwqlib.algorithms.qhd.compiler",
        qualname="QHDCompiler._objective_grid_values",
        old="            np.array([self.grid.grid_value(var_index, grid_index) - self.decomposer.centers[var_index]\n",
        new="            np.array([self.grid.grid_value(var_index, grid_index)\n",
        pytest_args=("tests/test_qhd_workflow.py::test_objective_far_from_the_origin_keeps_its_grid_values",),
    ),
    single_replacement_probe(
        name="qhd_support_cache_recomputed",
        module="nwqlib.algorithms.qhd.compiler",
        qualname="QHDCompiler._objective_grid_values",
        old="        cached = self._grid_value_cache.get(support)\n",
        new="        cached = None\n",
        pytest_args=("tests/test_qhd_workflow.py::test_compact_support_cache_survives_step_reuse",),
    ),
    # The array table producer admits an entry only with an imaginary part of
    # exactly zero, as the scalar conversion does.
    single_replacement_probe(
        name="qhd_table_imaginary_component_discard",
        module="nwqlib.algorithms.qhd.compiler",
        qualname="evaluate_chunk",
        old="    good = np.isfinite(values.real) & np.isfinite(values.imag) & (values.imag == 0)\n",
        new="    good = np.isfinite(values.real) & np.isfinite(values.imag) & (abs(values.imag) <= 1.0e-12)\n",
        pytest_args=("tests/test_qhd_workflow.py::test_objective_domain_precedes_pruning",),
    ),
    # A refinement grid point and a split score read the stored table entries
    # at their grid index instead of evaluating the support expressions again.
    single_replacement_probe(
        name="qhd_refinement_grid_value_reevaluated",
        module="nwqlib.algorithms.qhd.refinement",
        qualname="_tabulated_objective",
        old="        if indices is None:\n",
        new="        if True:\n",
        pytest_args=(
            "tests/test_qhd_refinement.py::test_stall_split_follows_the_end_mass_lemma_and_the_clearest_valley",
        ),
    ),
    # The stored extrema of a support table are selected from its entries and
    # validated against them.
    single_replacement_probe(
        name="qhd_support_extrema_unchecked",
        module="nwqlib.algorithms.qhd.records",
        qualname="SupportValues._extrema_of_the_array",
        old="        if (self.minimum, self.maximum, self.magnitude) != table_extrema(array):\n",
        new="        if False:\n",
        pytest_args=("tests/test_qhd_workflow.py::test_archive_keeps_support_tables_and_schedule",),
    ),
    # The tables evaluate the expanded support expressions, whose trees can be
    # far larger than the objective's, and sp.expand also multiplies out
    # half-integer, negative and exponent-sum powers.
    single_replacement_probe(
        name="qhd_table_work_without_expression_nodes",
        module="nwqlib.algorithms.qhd.method",
        qualname="QHD._admit_symbolic_work",
        old="            work += tuples * (nodes[support] + len(support))\n",
        new="            work += tuples * (1 + len(support))\n",
        pytest_args=(
            "tests/test_qhd_workflow.py::"
            "test_symbolic_work_is_admitted_before_expansion_and_table_evaluation",
        ),
    ),
    single_replacement_probe(
        name="qhd_expansion_bound_integer_powers_only",
        module="nwqlib.algorithms.qhd.objective",
        qualname="monomial_bound",
        old="        r = sum((a for a in sp.Add.make_args(node.exp) if a.is_Rational), sp.S.Zero)\n",
        new="        r = node.exp if node.exp.is_Integer and node.exp >= 0 else sp.S.Zero\n",
        pytest_args=(
            "tests/test_qhd_workflow.py::"
            "test_symbolic_work_is_admitted_before_expansion_and_table_evaluation",
        ),
    ),
    single_replacement_probe(
        name="qhd_kinetic_coefficient_x1_01",
        module="nwqlib.algorithms.qhd.kinetic",
        qualname="KineticCompiler.compile",
        old="coefficient = _normal_quotient(-kinetic_weight, 4.0 * spacing * spacing,\n",
        new="coefficient = _normal_quotient(-1.01 * kinetic_weight, 4.0 * spacing * spacing,\n",
        pytest_args=("tests/test_qhd_workflow.py::test_qhd_statevector_matches_ir_product_theory_distribution",),
    ),
    single_replacement_probe(
        name="qhd_second_order_reverted_to_forward",
        module="nwqlib.algorithms.qhd.kinetic",
        qualname="KineticCompiler.compile",
        old="            if trotter_order == 2\n",
        new="            if False\n",
        pytest_args=("tests/test_qhd_corrections.py::test_actual_second_order_taylor_and_forward_falsifier",),
    ),
    # The wrap link (0, K-1) closes the periodic grid's chain of links into a
    # cycle. Without it the hopping blocks restrict to the Dirichlet chain
    # instead of the circulant stencil.
    single_replacement_probe(
        name="qhd_periodic_wrap_link_missing",
        module="nwqlib.algorithms.qhd.grid",
        qualname="OneHotGrid.num_links",
        old='        return k if self.boundary == "periodic" else k - 1\n',
        new="        return k - 1\n",
        pytest_args=(
            "tests/test_qhd_periodic.py::test_periodic_hopping_blocks_restrict_to_the_circulant_stencil",
        ),
    ),
    # The periodic grid excludes the endpoint upper, which is the same point
    # as lower, so its spacing is L/K. The endpoint-inclusive L/(K-1) moves
    # every coordinate and rescales the kinetic term.
    single_replacement_probe(
        name="qhd_periodic_endpoint_inclusive_spacing",
        module="nwqlib.algorithms.qhd.grid",
        qualname="OneHotGrid.spacing",
        old="            intervals = k\n",
        new="            intervals = k - 1\n",
        pytest_args=(
            "tests/test_qhd_periodic.py::test_periodic_classical_model_matches_circulant_eigendecomposition",
        ),
    ),
    # The Schrodinger and split-step flavors evolve the support tables without
    # the objective constant, whose global phase is restored once. With the constant in the
    # potential diagonal, c = 1e16 rounds the tables' variation away and the
    # probabilities of c + g differ from those of g.
    single_replacement_probe(
        name="qhd_schrodinger_potential_with_constant",
        module="nwqlib.algorithms.qhd.method",
        qualname="_evolve_restricted",
        old="        real = potential_diagonal(plan.reconstruction, grid)\n",
        new="        real = potential_diagonal(plan.reconstruction, grid) + plan.reconstruction.constant\n",
        pytest_args=(
            "tests/test_qhd_workflow.py::"
            "test_classical_probabilities_do_not_depend_on_an_additive_constant[schrodinger]",
        ),
    ),
    single_replacement_probe(
        name="qhd_split_step_potential_with_constant",
        module="nwqlib.algorithms.qhd.split_step",
        qualname="potential_diagonal",
        old="    potential = np.zeros((k,) * d)\n",
        new="    potential = np.full((k,) * d, reconstruction.constant)\n",
        pytest_args=(
            "tests/test_qhd_workflow.py::"
            "test_classical_probabilities_do_not_depend_on_an_additive_constant[split_step]",
        ),
    ),
    # The split-step kinetic factor needs the orthonormal DST-I, which is its
    # own inverse. Without the normalization a transform pair scales the state
    # by 2 (K + 1).
    single_replacement_probe(
        name="qhd_split_step_dst_unnormalized",
        module="nwqlib.algorithms.qhd.split_step",
        qualname="_dst",
        old='    parts = scipy.fft.dst(parts, type=1, axis=axis, norm="ortho", overwrite_x=True, workers=1)\n',
        new='    parts = scipy.fft.dst(parts, type=1, axis=axis, norm="backward", overwrite_x=True, workers=1)\n',
        pytest_args=(
            "tests/test_qhd_split_step.py::test_split_step_matches_an_independent_strang_product",
            "tests/test_qhd_split_step.py::test_kinetic_transforms_are_orthonormal_eigenbases_of_the_stencils",
        ),
    ),
    # The spectral energy uses the signed Fourier index. The unsigned index
    # gives mode K - 1 the energy of momentum K - 1 instead of 1.
    single_replacement_probe(
        name="qhd_split_step_unsigned_spectral_index",
        module="nwqlib.algorithms.qhd.split_step",
        qualname="kinetic_eigenvalues",
        old="        signed = np.where(index < -(-k // 2), index, index - k)\n",
        new="        signed = index\n",
        pytest_args=(
            "tests/test_qhd_split_step.py::test_split_step_matches_an_independent_strang_product",
            "tests/test_qhd_split_step.py::"
            "test_spectral_and_finite_difference_energies_at_nyquist_and_low_momentum",
        ),
    ),
    # The split-step state budget charges the rounding of each kinetic angle
    # alpha E. With a huge first kinetic integral that rounding dominates the
    # state error, and a budget without it is exceeded by the kernel's state.
    single_replacement_probe(
        name="qhd_split_step_budget_without_phase_formation",
        module="nwqlib.algorithms.qhd.split_step",
        qualname="state_error",
        old="        + (EIGENVALUE_ROUNDOFF + 2) * abs(dt * kinetic_weight) * largest\n",
        new="        + 0.0\n",
        pytest_args=(
            "tests/test_qhd_split_step.py::"
            "test_split_step_budget_covers_the_formation_of_large_phase_angles",
        ),
    ),
    # The kernel's transforms run with one worker, the setting of the
    # scratch measurement. Without the argument SciPy takes the worker count
    # from scipy.fft.set_workers.
    single_replacement_probe(
        name="qhd_split_step_fft_ambient_workers",
        module="nwqlib.algorithms.qhd.split_step",
        qualname="_transform",
        old='        return call(state, axis=axis, norm="ortho", overwrite_x=True, workers=1)\n',
        new='        return call(state, axis=axis, norm="ortho", overwrite_x=True)\n',
        pytest_args=(
            "tests/test_qhd_split_step.py::test_split_step_transforms_run_with_one_worker",
        ),
    ),
    # A kept classical state needs the objective constant's phase to within
    # less than pi. Without the allowance check, c = 1e17 over two unit steps
    # is admitted with an allowance of 67 rad.
    single_replacement_probe(
        name="qhd_constant_phase_allowance_unchecked",
        module="nwqlib.algorithms.qhd.method",
        qualname="_admit_constant_phase",
        old="    if not allowance < pi:\n",
        new="    if False:\n",
        pytest_args=(
            "tests/test_qhd_workflow.py::test_classical_probabilities_do_not_depend_on_an_additive_constant",
        ),
    ),
    # Planning stores the start vector's error bound, and the windows read it.
    # Recomputing it from the Method evaluates the initial state after
    # planning, outside the admission.
    single_replacement_probe(
        name="qhd_start_error_reevaluated",
        module="nwqlib.algorithms.qhd.method",
        qualname="_host_start_error",
        old="    return plan.reconstruction.initial_state_error\n",
        new="    return evaluate_initial_state(plan.method.initial_state, _grid(plan))[1]\n",
        pytest_args=(
            "tests/test_qhd_workflow.py::test_initial_state_is_evaluated_once_at_planning",
        ),
    ),
    # The initial state's planning allowance adds the normalized arrays, the
    # stored payload and fixed bookkeeping to the per-point coefficient,
    # which alone falls below the traced stage on small grids.
    single_replacement_probe(
        name="qhd_initial_state_bytes_per_point_only",
        module="nwqlib.algorithms.qhd.method",
        qualname="_initial_state_bytes",
        old="    return _INITIAL_STATE_BYTES * d * k + arrays + stored + _INITIAL_STATE_OBJECT_BYTES * (d + 1)\n",
        new="    return _INITIAL_STATE_BYTES * d * k\n",
        pytest_args=(
            "tests/test_qhd_workflow.py::test_initial_state_planning_stage_fits_its_measured_allowance",
        ),
    ),
    # The DST of the Dirichlet grids also runs with one worker.
    single_replacement_probe(
        name="qhd_split_step_dst_ambient_workers",
        module="nwqlib.algorithms.qhd.split_step",
        qualname="_dst",
        old='    parts = scipy.fft.dst(parts, type=1, axis=axis, norm="ortho", overwrite_x=True, workers=1)\n',
        new='    parts = scipy.fft.dst(parts, type=1, axis=axis, norm="ortho", overwrite_x=True)\n',
        pytest_args=(
            "tests/test_qhd_split_step.py::test_split_step_transforms_run_with_one_worker",
        ),
    ),
    # A kept native or ir_product state needs the compiled phase within an
    # allowance below pi. Without the check, c = 1e17 over two unit steps is
    # admitted with a ledger allowance of 377 rad.
    single_replacement_probe(
        name="qhd_compiled_phase_allowance_unchecked",
        module="nwqlib.algorithms.qhd.method",
        qualname="_admit_compiled_phase",
        old="    if not allowance < pi:\n",
        new="    if False:\n",
        pytest_args=(
            "tests/test_qhd_workflow.py::test_circuit_route_kept_state_needs_a_phase_allowance_below_pi",
        ),
    ),
    # The compiler forms each constant coefficient from the stored binary64
    # constant. Multiplying the unrounded symbolic constant forms another
    # product than the one the ledger allowance assumes.
    single_replacement_probe(
        name="qhd_constant_coefficient_from_symbolic_constant",
        module="nwqlib.algorithms.qhd.compiler",
        qualname="QHDCompiler.build_step_pauli_ir",
        old='        constant_objective = coerce_real_scalar(self.decomposer.constant, context="constant objective")\n',
        new="        constant_objective = self.decomposer.constant\n",
        pytest_args=(
            "tests/test_qhd_workflow.py::"
            "test_compiler_forms_the_constant_coefficient_from_the_stored_binary64_constant",
        ),
    ),
    # The phase allowances round every operation upward. Round-to-nearest
    # products leave the ledger allowance below its exact value.
    single_replacement_probe(
        name="qhd_phase_allowance_products_to_nearest",
        module="nwqlib.algorithms.qhd.method",
        qualname="_mul_up",
        old="    return 0.0 if a == 0 or b == 0 else nextafter(a * b, inf)\n",
        new="    return a * b\n",
        pytest_args=(
            "tests/test_qhd_workflow.py::test_compiled_phase_allowances_are_evaluated_upward",
        ),
    ),
    # The phase ledger accumulates with Neumaier's compensation. Dropping the
    # correction leaves a plain running sum, whose error on a c = 1e16 ledger
    # exceeds the compensated bound.
    single_replacement_probe(
        name="qhd_phase_ledger_without_compensation",
        module="nwqlib.algorithms.qhd.compiler",
        qualname="QHDCompiler._compensated_add",
        old="        result = total + correction\n",
        new="        result = total\n",
        pytest_args=(
            "tests/test_qhd_workflow.py::test_compensated_ledger_sum_meets_its_bound_where_a_plain_sum_does_not",
        ),
    ),
    # Neumaier's residual takes the larger-magnitude operand first. Without the
    # branch the residual of 1e100 + 1 is lost, as in Kahan-Babuska summation.
    single_replacement_probe(
        name="qhd_phase_ledger_compensation_without_branch",
        module="nwqlib.algorithms.qhd.compiler",
        qualname="QHDCompiler._compensated_add",
        old="        if abs(high) >= abs(value):\n",
        new="        if True:\n",
        pytest_args=(
            "tests/test_qhd_workflow.py::test_phase_ledger_compensation_recovers_what_a_plain_sum_loses",
        ),
    ),
    # The classical constant-phase allowance is evaluated upward. Rounded to
    # nearest, the boundary constant 2414255523835869.5 over seven steps of
    # test_classical_probabilities_do_not_depend_on_an_additive_constant gives
    # 3.1415926535897927, below math.pi, and is admitted.
    single_replacement_probe(
        name="qhd_constant_phase_allowance_to_nearest",
        module="nwqlib.algorithms.qhd.method",
        qualname="_admit_constant_phase",
        old="    allowance = _add_up(_mul_up(coefficient, s_c), _upward(omitted))\n",
        new="    allowance = 3 * u * abs(c) * fsum(abs(dt * b) for _t, _a, b in reconstruction.step_weights)\n",
        pytest_args=(
            "tests/test_qhd_workflow.py::test_classical_probabilities_do_not_depend_on_an_additive_constant",
        ),
    ),
    # A native rejection ranks the ledger components with the native
    # consumer terms. Without them the witness -1e16 x**2 + 3e15 names the
    # constant instead of the projector component.
    single_replacement_probe(
        name="qhd_native_rejection_without_consumer_components",
        module="nwqlib.algorithms.qhd.method",
        qualname="_admit_compiled_phase",
        old="        components = _native_components(reconstruction, components, walsh)\n",
        new="        pass\n",
        pytest_args=(
            "tests/test_qhd_workflow.py::test_circuit_route_kept_state_needs_a_phase_allowance_below_pi",
        ),
    ),
    # A binary rejection ranks the Walsh components. Taking the one-hot
    # projector set instead leaves the Walsh formation unnamed, so the
    # constant-table witness names the final assignment.
    single_replacement_probe(
        name="qhd_binary_rejection_without_walsh_components",
        module="nwqlib.algorithms.qhd.method",
        qualname="_admit_compiled_phase",
        old="        components = _native_components(reconstruction, components, walsh)\n",
        new="        components = _native_components(reconstruction, components)\n",
        pytest_args=(
            "tests/test_qhd_binary.py::test_kept_binary_state_charges_walsh_identity_formation[quantum-schrodinger]",
        ),
    ),
    # The shifted-cubic kinetic weight is (2/(s + t))**3. A square instead of
    # a cube is a plausible transcription error that still decays in t.
    single_replacement_probe(
        name="qhd_shifted_cubic_kinetic_exponent",
        module="nwqlib.algorithms.qhd.schedules",
        qualname="ShiftedCubicSchedule.kinetic_weight",
        old='        return _admitted(x * x * x, f"the kinetic schedule weight a({t!r})", lambda: (\n',
        new='        return _admitted(x * x, f"the kinetic schedule weight a({t!r})", lambda: (\n',
        pytest_args=("tests/test_qhd_schedules.py::test_point_values_equal_their_formulas",),
    ),
    # Step k integrates over [k dt, (k + 1) dt]. An off-by-one end integrates
    # two steps, which the commuting exact propagator detects.
    single_replacement_probe(
        name="qhd_integrated_step_end_off_by_one",
        module="nwqlib.algorithms.qhd.schedules",
        qualname="step_weights",
        old="        start, time, end = step * dt, (step + 0.5) * dt, (step + 1) * dt\n",
        new="        start, time, end = step * dt, (step + 0.5) * dt, (step + 2) * dt\n",
        pytest_args=("tests/test_qhd_schedules.py::test_integrated_rule_is_exact_when_the_hamiltonians_commute",),
    ),
    single_replacement_probe(
        name="qhd_statevector_theory_ir_product_sign",
        module="nwqlib.algorithms.qhd.theory",
        qualname="apply_hopping",
        old="    view[lo] = c * x0 - 1j * (s * x1)\n    view[hi] = c * x1 - 1j * (s * x0)\n",
        new="    view[lo] = c * x0 + 1j * (s * x1)\n    view[hi] = c * x1 + 1j * (s * x0)\n",
        pytest_args=(
            "tests/test_qhd_workflow.py::test_qhd_statevector_matches_ir_product_theory_distribution",
            "tests/test_qhd_workflow.py::test_onehot_ir_product_matches_dense_exponentials_of_its_stored_blocks",
        ),
    ),
    # The direct one-hot projector multiplies its active slice by exp(-i angle) relative to the rest,
    # and the decoder sums the invalid population itself rather than subtracting the valid mass.
    single_replacement_probe(
        name="qhd_onehot_projector_active_phase_sign",
        module="nwqlib.algorithms.qhd.theory",
        qualname="apply_projector",
        old="    z1 = z0 * np.exp(-1j * float(angle))\n",
        new="    z1 = z0 * np.exp(1j * float(angle))\n",
        pytest_args=(
            "tests/test_qhd_workflow.py::test_onehot_ir_product_matches_dense_exponentials_of_its_stored_blocks",
        ),
    ),
    single_replacement_probe(
        name="qhd_decoder_invalid_mass_by_subtraction",
        module="nwqlib.algorithms.qhd.decoding",
        qualname="decode_statevector_arrays",
        old="        invalid = fsum(streamed(invalid_words))\n",
        new="        invalid = total - fsum(valid.reshape(-1).tolist())\n",
        pytest_args=("tests/test_qhd_readout.py::test_decoded_invalid_mass_is_summed_directly",),
    ),
    single_replacement_probe(
        name="qhd_paired_hopping_angle_x1_01",
        module="nwqlib.subroutines.hamiltonian_evolution.pauli_evolution",
        qualname="_hopping_parameter",
        old="    theta = (2.0 * time_step * coefficient) * 2.0\n",
        new="    theta = (2.0 * time_step * coefficient) * 2.02\n",
        pytest_args=(
            "tests/test_qhd_efficiency_contract.py::"
            "test_paired_hopping_matches_independent_dense_reference_with_two_cx",
        ),
    ),
    # The unsigned index of Liu et al.'s Eq. (90) gives (K-1)**2 at momentum
    # -1, so both the expansion and the spectral circuit leave q**2.
    single_replacement_probe(
        name="qhd_binary_unsigned_k2_index",
        module="nwqlib.algorithms.qhd.binary",
        qualname="signed_square_walsh",
        old="    weights = [1 << level for level in range(bits - 1)] + [-(1 << (bits - 1))]\n",
        new="    weights = [1 << level for level in range(bits - 1)] + [1 << (bits - 1)]\n",
        pytest_args=(
            "tests/test_qhd_binary.py::test_signed_square_expansion_equals_q_squared",
            "tests/test_qhd_binary.py::test_binary_circuit_equals_independent_product_at_three_bits[spectral]",
        ),
    ),
    # Omitting both swap layers while keeping the unpermuted phase table
    # applies R D R in place of D.
    single_replacement_probe(
        name="qhd_binary_missing_bit_reversal_relabel",
        module="nwqlib.algorithms.qhd.binary",
        qualname="kinetic_table",
        old="    return table.permuted(bit_reversal(bits)) if relabel else table\n",
        new="    return table\n",
        pytest_args=(
            "tests/test_qhd_binary.py::test_binary_circuit_equals_independent_product",
        ),
    ),
    # A full binary-register array reaches lexicographic grid order by reversing its variable axes, which the
    # native joint mass and the verification restriction read.
    single_replacement_probe(
        name="qhd_binary_register_array_axis_order",
        module="nwqlib.algorithms.qhd.binary",
        qualname="lexicographic_register_array",
        old="    return np.asarray(register).reshape((k,) * d).transpose(tuple(reversed(range(d)))).reshape(-1)\n",
        new="    return np.asarray(register).reshape((k,) * d).reshape(-1)\n",
        pytest_args=("tests/test_qhd_refinement.py::test_native_binary_joint_mass_decodes_the_binary_register",),
    ),
    # A streamed refused readout grid that gathers binary amplitudes with the variable axes reversed, and
    # a binary product that reserves no evolution state beside its model.
    single_replacement_probe(
        name="qhd_refinement_streamed_binary_axis_order",
        module="nwqlib.algorithms.qhd.refinement",
        qualname="_LevelReadout._stream",
        old="                index |= (np.int64(1) << (j * k + points[j])) if self.bits is None else points[j] << (j * self.bits)\n",
        new="                index |= (np.int64(1) << (j * k + points[j])) if self.bits is None else points[j] << ((d - 1 - j) * self.bits)\n",
        pytest_args=("tests/test_qhd_readout.py::test_a_refused_readout_grid_is_streamed_with_the_same_probabilities",),
    ),
    single_replacement_probe(
        name="qhd_binary_product_without_state_reservation",
        module="nwqlib.algorithms.qhd.theory",
        qualname="run_binary_product",
        old="    model = BinaryModel(grid, method, reconstruction.support_values, held_bytes=held)\n",
        new="    model = BinaryModel(grid, method, reconstruction.support_values, held_bytes=0)\n",
        pytest_args=("tests/test_qhd_resources.py::test_binary_census_and_evolution_states_are_admitted_with_the_model_before_synthesis",),
    ),
    # Lexicographic order as the table address transposes the coupled table
    # of two variables.
    single_replacement_probe(
        name="qhd_binary_variable_axis_permutation",
        module="nwqlib.algorithms.qhd.binary",
        qualname="address_table",
        old="    return np.asarray(values, dtype=float).reshape(shape).transpose(tuple(reversed(range(support_size)))).ravel()\n",
        new="    return np.asarray(values, dtype=float).reshape(shape).ravel()\n",
        pytest_args=(
            "tests/test_qhd_binary.py::test_binary_circuit_equals_independent_product",
        ),
    ),
    # The augmented-Lagrangian trajectories of the plan's grid oracle fix the
    # point, multiplier and penalty of every round, so a sign or a dropped
    # term in the update rules changes a recorded round.
    single_replacement_probe(
        name="al_lambda_update_sign",
        module="nwqlib.algorithms.qhd.constrained",
        qualname="_update",
        old="    lambda_plus = tuple(multiplier + penalty * value for multiplier, value in zip(lambda_bar, h, strict=True))\n",
        new="    lambda_plus = tuple(multiplier - penalty * value for multiplier, value in zip(lambda_bar, h, strict=True))\n",
        pytest_args=(
            "tests/test_qhd_augmented_lagrangian.py::test_best_observed_trajectories_match_the_grid_oracle[equality]",
        ),
    ),
    single_replacement_probe(
        name="al_phr_shift_sign",
        module="nwqlib.algorithms.qhd.constrained",
        qualname="_inner_objective",
        old="            terms.append(half * (sp.Max(0, g + shift) ** 2 - shift**2))\n",
        new="            terms.append(half * (sp.Max(0, g - shift) ** 2 - shift**2))\n",
        pytest_args=(
            "tests/test_qhd_augmented_lagrangian.py::test_best_observed_trajectories_match_the_grid_oracle[phr]",
        ),
    ),
    # Only the complementarity trajectory has a round with mu-bar + rho g < 0.
    single_replacement_probe(
        name="al_mu_update_without_positive_part",
        module="nwqlib.algorithms.qhd.constrained",
        qualname="_update",
        old="    mu_plus = tuple(max(0.0, multiplier + penalty * value) for multiplier, value in zip(mu_bar, g, strict=True))\n",
        new="    mu_plus = tuple(multiplier + penalty * value for multiplier, value in zip(mu_bar, g, strict=True))\n",
        pytest_args=(
            "tests/test_qhd_augmented_lagrangian.py::"
            "test_best_observed_trajectories_match_the_grid_oracle[complementarity]",
        ),
    ),
    single_replacement_probe(
        name="al_penalty_test_direction",
        module="nwqlib.algorithms.qhd.constrained",
        qualname="_next_penalty",
        old="    if previous is None or measure <= options.reduction_ratio * previous:\n",
        new="    if previous is None or measure >= options.reduction_ratio * previous:\n",
        pytest_args=(
            "tests/test_qhd_augmented_lagrangian.py::test_best_observed_trajectories_match_the_grid_oracle[equality]",
        ),
    ),
    # In the complementarity trajectory both penalty increases come from V
    # at strictly feasible points with positive multipliers.
    single_replacement_probe(
        name="al_measure_without_complementarity",
        module="nwqlib.algorithms.qhd.constrained",
        qualname="_update",
        old="    measure = max(h_norm, max(map(abs, shifts), default=0.0))\n",
        new="    measure = h_norm\n",
        pytest_args=(
            "tests/test_qhd_augmented_lagrangian.py::"
            "test_best_observed_trajectories_match_the_grid_oracle[complementarity]",
        ),
    ),
    single_replacement_probe(
        name="al_stationarity_with_truncated_multipliers",
        module="nwqlib.algorithms.qhd.constrained",
        qualname="_rounds",
        old='        stationarity, unavailable, more, extra = _stationarity(setup, chosen["point"], update["lambda_plus"],\n'
            '                                                               update["mu_plus"])\n',
        new='        stationarity, unavailable, more, extra = _stationarity(setup, chosen["point"], update["lambda_next"],\n'
            '                                                               update["mu_next"])\n',
        pytest_args=(
            "tests/test_qhd_augmented_lagrangian.py::"
            "test_stationarity_uses_the_tentative_multipliers_when_the_safeguard_truncates",
        ),
    ),
    single_replacement_probe(
        name="al_complementarity_on_unnormalized_constraints",
        module="nwqlib.algorithms.qhd.constrained",
        qualname="_update",
        old="    complementarity = max(h_norm, max((abs(min(-value, multiplier)) for value, multiplier in zip(g, mu_plus)),\n",
        new="    complementarity = max(h_norm, max((abs(min(-value, multiplier)) for value, multiplier in zip(g_raw, mu_plus)),\n",
        pytest_args=(
            "tests/test_qhd_augmented_lagrangian.py::"
            "test_rescaling_a_term_leaves_the_normalized_run_and_rescales_the_multipliers[constraint]",
        ),
    ),
    # A refined round chooses the least recorded relative L_k, with earlier ties.
    # The combination test independently checks analytic L_k minimization for
    # its reported level points. Under those premises, the last level or the
    # least-f level can exceed the test's derived KKT-distance bound.
    single_replacement_probe(
        name="al_refined_point_from_last_level",
        module="nwqlib.algorithms.qhd.constrained",
        qualname="_rounds",
        old="            chosen = _refined_choice(refined, refined.best, options.inner_point, setup, form)\n",
        new="            chosen = _refined_choice(refined, refined.levels[-1], options.inner_point, setup, form)\n",
        pytest_args=(
            "tests/test_qhd_augmented_lagrangian.py::"
            "test_refined_rounds_stay_within_the_distance_bound_to_the_kkt_point",
        ),
    ),
    single_replacement_probe(
        name="al_refined_point_by_original_objective",
        module="nwqlib.algorithms.qhd.constrained",
        qualname="_rounds",
        old="            chosen = _refined_choice(refined, refined.best, options.inner_point, setup, form)\n",
        new="            chosen = _refined_choice(refined, min(refined.levels, key=lambda level: _value("
            "setup.objective, level.point)), options.inner_point, setup, form)\n",
        pytest_args=(
            "tests/test_qhd_augmented_lagrangian.py::"
            "test_refined_rounds_stay_within_the_distance_bound_to_the_kkt_point",
        ),
    ),
    single_replacement_probe(
        name="native_y_basis_sdg_to_s",
        module="nwqlib.subroutines.hamiltonian_evolution.pauli_evolution",
        qualname="apply_pauli_rotation",
        old="circuit.sdg(qubit)\n",
        new="circuit.s(qubit)\n",
        pytest_args=(
            "tests/test_hamiltonian_evolution.py::test_apply_pauli_rotation_matches_pauli_evolution_gate",
        ),
    ),
    # Provider formulas are compared with independent closed forms rather
    # than another implementation path.
    single_replacement_probe(
        name="lchs_kernel_negative_exponent_x1_01",
        module="nwqlib.algorithms.lchs.providers",
        qualname="eq7_kernel_function",
        old="    return np.exp(-(1.0 + 1.0j * z_value) ** beta) / eq7_cbeta(beta)\n",
        new="    return 1.01 * np.exp(-(1.0 + 1.0j * z_value) ** beta) / eq7_cbeta(beta)\n",
        pytest_args=(
            "tests/test_lchs_providers.py::"
            "test_near_optimal_eq7_provider_matches_independent_closed_form",
        ),
    ),
    single_replacement_probe(
        name="lchs_cauchy_kernel_x1_01",
        module="nwqlib.algorithms.lchs.providers",
        qualname="_cauchy_provider_coefficient",
        old="    return complex(1.0 / (pi * (1.0 + k_value**2)))\n",
        new="    return complex(1.01 / (pi * (1.0 + k_value**2)))\n",
        pytest_args=(
            "tests/test_lchs_providers.py::test_every_allowed_pair_has_independent_coefficients_and_status"
            "[kernel_request2-quadrature_request2-uncertified]",
        ),
    ),
    single_replacement_probe(
        name="lchs_low_somma_gaussian_denominator_double",
        module="nwqlib.algorithms.lchs.providers",
        qualname="low_somma_fhat_2",
        old="        * exp(c_value - (k_value**2 + 1.0) / (4.0 * gamma**2))\n",
        new="        * exp(c_value - (k_value**2 + 1.0) / (8.0 * gamma**2))\n",
        pytest_args=(
            "tests/test_lchs_providers.py::"
            "test_low_somma_paper_profile_grid_matches_independent_closed_form",
        ),
    ),
    single_replacement_probe(
        name="lchs_low_somma_profile_radius_half",
        module="nwqlib.algorithms.lchs.providers",
        qualname="_resolve_low_somma_pair_profile",
        old="    radius = 2.0 * c_value * gamma**2\n",
        new="    radius = c_value * gamma**2\n",
        pytest_args=(
            "tests/test_lchs_providers.py::"
            "test_low_somma_paper_profile_grid_matches_independent_closed_form",
        ),
    ),
    single_replacement_probe(
        name="lchs_low_somma_drop_positive_endpoint",
        module="nwqlib.algorithms.lchs.providers",
        qualname="_build_symmetric_uniform_trapezoid",
        old=(
            "    nodes = tuple(float(index * step) for index in "
            "range(-index_limit, index_limit + 1))\n"
        ),
        new=(
            "    nodes = tuple(float(index * step) for index in range(-index_limit, index_limit))\n"
        ),
        pytest_args=(
            "tests/test_lchs_providers.py::"
            "test_low_somma_paper_profile_grid_matches_independent_closed_form",
        ),
    ),
    single_replacement_probe(
        name="lchs_signed_binary_sign_transition_shift",
        module="nwqlib.algorithms.lchs.providers",
        qualname="_build_signed_binary",
        old="    sign_transition = 2 ** (num_qubits - 1)\n",
        new="    sign_transition = 2**num_qubits\n",
        pytest_args=(
            "tests/test_lchs_providers.py::"
            "test_signed_binary_nodes_and_affine_certificate_are_exact",
        ),
    ),
    single_replacement_probe(
        name="lchs_composite_gauss_weight_x1_01",
        module="nwqlib.algorithms.lchs.providers",
        qualname="composite_gauss_grid",
        old="    mapped_weights = 0.5 * h1 * base_weights\n",
        new="    mapped_weights = 0.505 * h1 * base_weights\n",
        pytest_args=(
            "tests/test_lchs_providers.py::"
            "test_default_provider_pair_matches_pre_provider_construction_at_roundoff",
        ),
    ),
    single_replacement_probe(
        name="lchs_provider_finalizer_coefficient_half",
        module="nwqlib.algorithms.lchs.providers",
        qualname="_resolve_coefficient_context",
        old='            coefficient = weight * kernel.coefficient(node, pair_context)\n',
        new='            coefficient = 0.5 * weight * kernel.coefficient(node, pair_context)\n',
        pytest_args=(
            "tests/test_lchs_providers.py::"
            "test_every_allowed_pair_has_independent_coefficients_and_status",
        ),
    ),
    single_replacement_probe(
        name="lchs_host_pauli_rotation_sign_flip",
        module="nwqlib.algorithms.lchs.host_pf",
        qualname="apply_node",
        old="                result += (-1j*sin(angle))*scratch\n",
        new="                result += (1j*sin(angle))*scratch\n",
        pytest_args=("tests/test_lchs_primary.py::test_selected_pauli_vector_action_preserves_order_phase_source_and_replay",),
    ),
    single_replacement_probe(
        name="lchs_select_angle_time_half",
        module="nwqlib.algorithms.lchs.time_independent_terms",
        qualname="_build_product_formula_select_plan",
        old="                    multiplier * elapsed_time * coefficient.real / steps\n",
        new="                    0.5 * multiplier * elapsed_time * coefficient.real / steps\n",
        pytest_args=(
            "tests/test_lchs_select_synthesis.py::"
            "test_generated_angle_tables_match_independent_suzuki_products",
        ),
    ),
    single_replacement_probe(
        name="lchs_select_inactive_repetition_reactivated",
        module="nwqlib.algorithms.lchs.time_independent_terms",
        qualname="_build_product_formula_select_plan",
        old="                    if repetition < steps and steps > 0\n",
        new="                    if repetition <= steps and steps > 0\n",
        pytest_args=(
            "tests/test_lchs_select_synthesis.py::test_budgeted_plan_zeroes_inactive_repetitions",
        ),
    ),
    single_replacement_probe(
        name="lchs_order_one_schedule_to_suzuki",
        module="nwqlib.algorithms.lchs.select_synthesis",
        qualname="product_formula_occurrence_template",
        old="        synthesis = LieTrotter()\n",
        new="        synthesis = SuzukiTrotter(order=2)\n",
        pytest_args=("tests/test_lchs_structured_select.py::test_structured_and_multiplexed_select_match_independent_dense_blocks[_mixed_problem-2-1]",),
    ),
    single_replacement_probe(
        name="trotter_pauli_triangle_order_one_factor_half",
        module="nwqlib.subroutines.trotterization.error_budget",
        qualname="coefficient_up",
        old="        return Fraction(sum_up(part1)) * Fraction(2) ** (2 * e)\n",
        new="        return Fraction(sum_up(part1)) / 2 * Fraction(2) ** (2 * e)\n",
        pytest_args=(
            "tests/test_trotterization.py::"
            "test_two_term_pauli_laws_zero_coefficients_and_identity_contract",
        ),
    ),
    single_replacement_probe(
        name="trotter_pauli_triangle_order_two_nested_factor_half",
        module="nwqlib.subroutines.trotterization.error_budget",
        qualname="coefficient_up",
        old="    w = Fraction(sum_up(part12)) / 3 + Fraction(sum_up(part24)) / 6\n",
        new="    w = Fraction(sum_up(part12)) / 6 + Fraction(sum_up(part24)) / 6\n",
        pytest_args=(
            "tests/test_trotterization.py::"
            "test_two_term_pauli_laws_zero_coefficients_and_identity_contract",
        ),
    ),
    # Relative pruning preserves the original H-error scale and rejects a
    # fully discarded nonzero L before QSP phase selection.
    single_replacement_probe(
        name="lchs_compiled_qsp_relative_pruning_drop",
        module="nwqlib.algorithms.lchs.compiled_selection",
        qualname="_compiled_qsp_part_plans",
        old='    kept, _pruned_mass_bound = prune_pauli_terms_relative(\n        (*l_decomposition.terms, *h_decomposition.terms)\n    )\n',
        new='    kept = (*l_decomposition.terms, *h_decomposition.terms)\n',
        pytest_args=('tests/test_lchs_compiled_select.py::test_selected_qsp_pruned_h_and_near_antihermitian_l_keep_original_scale',),
    ),
    # A two-child QSP generator gate is controlled by the combine qubit and
    # then by the parity qubit. Pricing it as two controls at once puts the
    # law below the transpiled built SELECT.
    single_replacement_probe(
        name="lchs_qsp_nested_controls_as_two_at_once",
        module="nwqlib.algorithms.lchs.compiled_selection",
        qualname="compiled_select_cx_projection",
        old="        twice = _twice_controlled_kind_cx()\n",
        new="        twice = _controlled_kind_cx(2)\n",
        pytest_args=(
            "tests/test_lchs_resource_structural_law.py::"
            "test_compiled_qsp_select_cx_law_bounds_the_built_select",
        ),
    ),
    # Qiskit's noancilla RY with two or three controls is a Gray-code circuit
    # of 8 or 20 CX, above the MCRZ count of 4 or 14.
    single_replacement_probe(
        name="lchs_controlled_ry_gray_code_drop",
        module="nwqlib.algorithms.lchs.compiled_selection",
        qualname="_controlled_kind_cx",
        old='    gray_code_ry = 3 * (1 << controls) - 4 if controls <= 3 else cost["rz"]\n',
        new='    gray_code_ry = cost["rz"]\n',
        pytest_args=(
            "tests/test_lchs_resource_structural_law.py::"
            "test_controlled_census_prices_bound_qiskit_for_every_kind",
        ),
    ),
    # A triangular matrix must reach scipy.sparse.linalg.expm. SciPy 1.18.1's
    # scipy.linalg.expm cancels in the superdiagonal of a triangular input
    # with close diagonal entries.
    single_replacement_probe(
        name="lchs_reference_triangular_kernel_drop",
        module="nwqlib.algorithms.lchs.references",
        qualname="_exponential",
        old="        if upper or lower:\n",
        new="        if False:\n",
        pytest_args=(
            "tests/test_lchs_verification.py::"
            "test_closed_form_is_accurate_for_close_diagonal_entries_and_singular_a",
        ),
    ),
    # Planning must refuse the coefficient table that the LCU intake of a
    # branch-controlled SELECT refuses at construction.
    single_replacement_probe(
        name="lchs_branch_select_planning_intake_drop",
        module="nwqlib.algorithms.lchs.parameters",
        qualname="select_parameters",
        old="    if selected_select == 'branch_controlled':\n",
        new="    if False:\n",
        pytest_args=(
            "tests/test_lchs_select_synthesis.py::"
            "test_planning_refuses_a_branch_select_that_the_lcu_intake_refuses",
        ),
    ),
    single_replacement_probe(
        name="lchs_compiled_pruned_h_error_drop",
        module='nwqlib.algorithms.lchs.native',
        qualname='build_selected_evolution',
        old='joint.error_bound+missing_h',
        new='joint.error_bound',
        pytest_args=('tests/test_lchs_compiled_select.py::test_selected_qsp_pruned_h_and_near_antihermitian_l_keep_original_scale',),
    ),
    # Native Pauli values and shot counts have distinct scalar reconstruction.
    # Each probe has an independent analytic-value witness.
    single_replacement_probe(
        name='gcim_native_pauli_expectation_x1_01',
        module='nwqlib.algorithms.gcim.adapt_acquisition',
        qualname='_read_values',
        old='result = {v.label: v.value for v in chunk.values}',
        new='result = {v.label: 1.01*v.value for v in chunk.values}',
        pytest_args=('tests/test_adapt_gcim.py::test_query_completeness_and_exact_point_identity',),
    ),
    single_replacement_probe(
        name='gcim_count_expectation_x1_01',
        module='nwqlib.algorithms.gcim.adapt_acquisition',
        qualname='_read_values',
        old='/ total\n',
        new='/ total * 1.01\n',
        pytest_args=('tests/test_gcim_pencil.py::test_ancilla_population_normalization_and_incremental_pricing',),
    ),
    single_replacement_probe(
        name='gcim_diagonal_overlap_batch_restore',
        module='nwqlib.algorithms.gcim.pencil',
        qualname='matrix_element_count',
        old='2 * (pair_count - diagonal_pair_count)',
        new='2 * pair_count',
        pytest_args=('tests/test_gcim_pencil.py::test_ancilla_population_normalization_and_incremental_pricing',),
    ),
    single_replacement_probe(
        name='gcim_diagonal_overlap_identity_zero',
        module='nwqlib.algorithms.gcim.pencil',
        qualname='assemble_pair_pencil',
        old='                overlap[left, left] = 1.0\n',
        new='                overlap[left, left] = 0.0\n',
        pytest_args=('tests/test_gcim_production.py::test_complex_phase_and_identity_share_actual_gram',),
    ),

    # In the banded construction alpha
    # feeds both the PREP amplitudes and the recorded subnormalization, so a
    # 1% inflation breaks the machine-precision embedding identity (top-left
    # block == A / alpha). The banded target is deliberate: dense_dilation is
    # self-consistent for any alpha >= ||A||_2 (an inflated alpha yields a
    # valid encoding with the inflated alpha recorded), so an embedding test
    # alone cannot kill an alpha mutation there — only the analytic-alpha
    # asserts can.
    single_replacement_probe(
        name="block_encoding_alpha_x1_01",
        module="nwqlib.subroutines.block_encoding.banded",
        qualname="build_banded_block_encoding",
        old="    alpha = float(np.sum(magnitudes))\n",
        new="    alpha = 1.01 * float(np.sum(magnitudes))\n",
        pytest_args=(
            "tests/test_block_encoding.py::test_embedding_identity_reproduces_scaled_matrix",
        ),
    ),
    single_replacement_probe(
        name="dense_unitary_cx_qsd_upper_bound_23_to_24",
        module="nwqlib.backends.resources",
        qualname="dense_unitary_cx_qsd_upper_bound",
        old="    return (23 * 4**total_qubits - 72 * 2**total_qubits + 64) // 48\n",
        new="    return (24 * 4**total_qubits - 72 * 2**total_qubits + 64) // 48\n",
        pytest_args=(
            "tests/test_block_encoding.py::"
            "test_dense_unitary_cx_qsd_upper_bound_values_and_domain",
        ),
    ),
    single_replacement_probe(
        name="projected_ucg_core_cx_plus1",
        module="nwqlib.subroutines._multiplexors",
        qualname="projected_unitary_resource_law",
        old="    core = (1 << effective_control_count) - 1\n",
        new="    core = (1 << effective_control_count)\n",
        pytest_args=("tests/test_block_encoding.py::test_dependency_projected_ucg_basis_scaling",),
    ),
    single_replacement_probe(
        name="banded_ucrz_cx_plus1",
        module="nwqlib.backends.resources",
        qualname="block_encoding_per_query_cx",
        old='            * int(counts["ucrz_basis_cx_per_gate"])\n',
        new='            * (int(counts["ucrz_basis_cx_per_gate"]) + 1)\n',
        pytest_args=(
            "tests/test_block_encoding.py::"
            "test_banded_multiplexor_law_tracks_measured_basis_scaling",
        ),
    ),
    # Perturb the projector phase with block ancillas present. The selectors
    # cover this path through QSVT and evolution, not the ancilla-free pair.
    single_replacement_probe(
        name="qsp_phase_factor_plus_0_01",
        module="nwqlib.subroutines.qsp.evolution",
        qualname="_append_projector_phase",
        old=(
            "    circuit.append(RZGate(2.0 * angle), [signal])\n"
            "    if pair_qubit is not None:\n"
        ),
        new=(
            "    circuit.append(RZGate(2.0 * angle + 0.01), [signal])\n"
            "    if pair_qubit is not None:\n"
        ),
        pytest_args=(
            "tests/test_qsp_evolution.py::test_qsvt_trivial_phases_block_encode_chebyshev_of_matrix",
            "tests/test_qsp_evolution.py::test_qsp_evolution_matches_dense_expm_within_documented_bound",
        ),
    ),
    # 1 - x**2 loses the relative accuracy of s^2 at the endpoint nodes,
    # which the exact rational T_255 comparison detects.
    single_replacement_probe(
        name="qsp_signal_root_cancelling_form",
        module="nwqlib.subroutines.qsp.phases",
        qualname="_signal_matrices",
        old="    root = np.sqrt(np.clip((1.0 - x_values) * (1.0 + x_values), 0.0, None))\n",
        new="    root = np.sqrt(np.clip(1.0 - x_values**2, 0.0, None))\n",
        pytest_args=("tests/test_qsp_evolution.py::test_all_zero_phases_realize_chebyshev_t_d",),
    ),
    # Without the Newton start, the LCHS guide's first problem fails at both
    # margins on its degree-5 sine target.
    single_replacement_probe(
        name="qsp_newton_start_skipped",
        module="nwqlib.subroutines.qsp.phases",
        qualname="solve_symmetric_qsp_phases",
        old="        candidate, candidate_best = _damped_newton(residual_and_jacobian, start)\n",
        new="        candidate, candidate_best = free, best\n",
        pytest_args=("tests/test_qsp_evolution.py::test_lchs_guide_first_problem_plans_qsp_evolution",),
    ),
    # A Newton start after every L-BFGS start would also run on targets that
    # L-BFGS already solves, and its residual evaluations would change the
    # count that the parities and margins share under max_evaluations.
    single_replacement_probe(
        name="qsp_newton_start_unconditional",
        module="nwqlib.subroutines.qsp.phases",
        qualname="solve_symmetric_qsp_phases",
        old="    if best >= residual_tolerance:\n        # Near max|f| = 1, L-BFGS from this start has no convergence\n",
        new="    if True:\n        # Near max|f| = 1, L-BFGS from this start has no convergence\n",
        pytest_args=(
            "tests/test_qsp_evolution.py::"
            "test_qsp_evolution_evaluation_limit_is_shared_by_parities_and_margins",
        ),
    ),
    # A NaN maximum from a nonfinite L-BFGS point compares as converged, so
    # the Newton start is skipped and the verification rejects the NaN phases.
    single_replacement_probe(
        name="qsp_nonfinite_residual_kept_as_nan",
        module="nwqlib.subroutines.qsp.phases",
        qualname="_max_abs_residual",
        old="    return value if np.isfinite(value) else np.inf\n",
        new="    return value\n",
        pytest_args=("tests/test_qsp_evolution.py::test_nonfinite_lbfgs_point_falls_through_to_the_newton_start",),
    ),
    # Compensation is an LCHS recovery convention, not a rewrite of the
    # generic BlockEncoding contract. Dropping the deterministic OAA deficit
    # from the raw bound must violate the two-frame contract test.
    single_replacement_probe(
        name="qsp_raw_oaa_deficit_drop",
        module="nwqlib.subroutines.qsp.evolution",
        qualname="qsp_evolution_error_terms",
        old='    error_bound = None if part_error is None else (\n        deficit\n        + 6.0 * part_error\n',
        new='    error_bound = None if part_error is None else (\n        0.0\n        + 6.0 * part_error\n',
        pytest_args=(
            "tests/test_lchs_compiled_select.py::"
            "test_qsp_oaa_raw_block_contract_and_compensated_metadata",
        ),
    ),
    # The independent joint-alpha check separates correct structure from a shared normalization error.
    single_replacement_probe(
        name="lchs_qsp_plan_alpha_k_x1_01",
        module="nwqlib.subroutines.qsp.evolution",
        qualname="_control_diagonal_generator_layout",
        old='        "alpha": float(sum(child.alpha * scale for child, scale in active)),\n',
        new='        "alpha": 1.01 * float(sum(child.alpha * scale for child, scale in active)),\n',
        pytest_args=('tests/test_qsp_evolution.py::test_lchs_qsp_selected_quantities_reach_actual_native_program',),
    ),
    # The multiplexed-rotation
    # angle law theta_b = 2 arccos(d_b / d_max) is what makes the encoded
    # block equal the control-register diagonal; scaling the angles breaks
    # the machine-precision kron identity in the composition test.
    single_replacement_probe(
        name="compiled_select_diagonal_angle_x1_01",
        module="nwqlib.subroutines.qsp.evolution",
        qualname="_diagonal_rotation_circuit",
        old="    angles = 2.0 * np.arccos(np.clip(diagonal / max_abs, -1.0, 1.0))\n",
        new="    angles = 2.02 * np.arccos(np.clip(diagonal / max_abs, -1.0, 1.0))\n",
        pytest_args=(
            "tests/test_lchs_compiled_select.py::test_control_diagonal_generator_encoding_composition",
        ),
    ),
    # The conformance test recomputes the
    # implementation-derived counts from the constructed compiled SELECT, so
    # an inflated query law diverges from the measured multiplexed-rotation
    # and child-call counts.
    single_replacement_probe(
        name="compiled_select_query_law_plus1",
        module="nwqlib.subroutines.qsp.evolution",
        qualname="compiled_select_resource_law",
        old="    queries = QSP_OAA_QUERY_MULTIPLIER * (cos_degree + sin_degree)\n",
        new="    queries = QSP_OAA_QUERY_MULTIPLIER * (cos_degree + sin_degree) + 1\n",
        pytest_args=(
            "tests/test_lchs_compiled_select.py::test_compiled_select_gate_count_law_conformance",
        ),
    ),
    # The recursive-decomposition test independently counts UCR/UCGate/
    # diagonal basis structure for zero through five controls, so doubling
    # the table-size owner must disagree with every constructed gate.
    single_replacement_probe(
        name="multiplexor_table_size_double",
        module="nwqlib.subroutines._multiplexors",
        qualname="multiplexor_resource_law",
        old="    table_size = 1 << control_qubits\n",
        new="    table_size = 2 << control_qubits\n",
        pytest_args=(
            "tests/test_multiplexors.py::"
            "test_multiplexor_resource_law_matches_recursive_decomposition",
        ),
    ),
    single_replacement_probe(
        name="lchs_pf_ucr_cx_plus_occurrences",
        module="nwqlib.subroutines._multiplexors",
        qualname="product_formula_select_resource_law",
        old=('    multiplexed_cx = occurrence_count * ucr_law["basis_cx_gates"]\n'),
        new=(
            '    multiplexed_cx = occurrence_count * ucr_law["basis_cx_gates"] + occurrence_count\n'
        ),
        pytest_args=(
            "tests/test_lchs_resource_structural_law.py::"
            "test_product_formula_law_matches_recursive_built_select",
        ),
    ),
    single_replacement_probe(
        name="lchs_pf_parity_cx_factor_three",
        module="nwqlib.subroutines._multiplexors",
        qualname="product_formula_select_resource_law",
        old="    parity_cx = sum(2 * (support_size - 1) for support_size in support_sizes) * int(plan.repetitions)\n",
        new="    parity_cx = sum(3 * (support_size - 1) for support_size in support_sizes) * int(plan.repetitions)\n",
        pytest_args=(
            "tests/test_lchs_resource_structural_law.py::"
            "test_product_formula_law_matches_recursive_built_select",
        ),
    ),
    single_replacement_probe(
        name="lchs_pf_control_diagonal_cx_plus1",
        module="nwqlib.subroutines._multiplexors",
        qualname="product_formula_select_resource_law",
        old='    diagonal_cx = diagonal_law["basis_cx_gates"] if has_diagonal else 0\n',
        new='    diagonal_cx = (diagonal_law["basis_cx_gates"] if has_diagonal else 0) + 1\n',
        pytest_args=(
            "tests/test_lchs_resource_structural_law.py::"
            "test_product_formula_law_matches_recursive_built_select",
        ),
    ),
    # Affine lowering applies one controlled rotation for each exact-nonzero
    # coefficient; the structural test counts those controls and their CX
    # decompositions instead of mirroring the formula.
    single_replacement_probe(
        name="affine_rotation_active_count_plus1",
        module="nwqlib.subroutines._multiplexors",
        qualname="affine_rotation_resource_law",
        old=("    controlled_rotations = sum(angle != 0.0 for angle in affine.coefficients)\n"),
        new=("    controlled_rotations = sum(angle != 0.0 for angle in affine.coefficients) + 1\n"),
        pytest_args=(
            "tests/test_multiplexors.py::test_affine_rotation_resource_law_matches_lowering",
        ),
    ),
    # The structured SELECT tests reconstruct every control-basis block from
    # independent dense Pauli exponentials, so shifting a bit weight changes
    # both the certificate and the realized block family.
    single_replacement_probe(
        name="lchs_affine_table_bit_weight_shift",
        module="nwqlib.subroutines._multiplexors",
        qualname="affine_angle_table_values",
        old="            if branch & (1 << qubit)\n",
        new="            if branch & (2 << qubit)\n",
        pytest_args=(
            "tests/test_lchs_structured_select.py::"
            "test_structured_and_multiplexed_select_match_independent_dense_blocks",
        ),
    ),
    # Recursive decomposition counts the built CRZ gates independently of the
    # closed-form structural law, so one extra CX cannot survive.
    single_replacement_probe(
        name="lchs_structured_select_cx_plus1",
        module="nwqlib.subroutines._multiplexors",
        qualname="product_formula_select_resource_law",
        old="        structured_cx = sum(\n",
        new="        structured_cx = 1 + sum(\n",
        pytest_args=("tests/test_lchs_structured_select.py::test_structured_and_multiplexed_select_match_independent_dense_blocks",),
    ),
    # The actual mixed-source selection binds each Gauss application's elapsed
    # time to its k-address, before the one selected QSP phase fit.
    single_replacement_probe(
        name="compiled_select_inhomogeneous_time_fold_drop",
        module="nwqlib.algorithms.lchs.parameters",
        qualname="select_parameters",
        old="diagonal_l,diagonal_h = layout.elapsed_times*layout.k_values/problem.elapsed_time,layout.elapsed_times/problem.elapsed_time\n",
        new="diagonal_l,diagonal_h = layout.k_values,layout.elapsed_times/problem.elapsed_time\n",
        pytest_args=(
            "tests/test_lchs_qsp_source.py::test_actual_qsp_source_binds_elapsed_diagonals_recovery_and_physical_phase",
        ),
    ),
    # A scaled homogeneous H diagonal likewise sits
    # below the synthesis bound at validation scale; the exact recorded-
    # diagonal pin (all-ones H) is the guard.
    single_replacement_probe(
        name="compiled_select_homogeneous_h_diagonal_x1_01",
        module="nwqlib.algorithms.lchs.parameters",
        qualname="select_parameters",
        old="diagonal_l,diagonal_h = data.quadrature.k_nodes,np.ones(len(coefficients))\n",
        new="diagonal_l,diagonal_h = data.quadrature.k_nodes,np.full(len(coefficients),1.01)\n",
        pytest_args=(
            "tests/test_lchs_qsp_source.py::test_actual_qsp_source_binds_elapsed_diagonals_recovery_and_physical_phase",
        ),
    ),
    # The LCHS recovery layer divides out the QSP evolution's known OAA
    # amplitude factor. Dropping the shared scale revives the deterministic
    # deficit floor in both homogeneous and inhomogeneous statevector paths.
    single_replacement_probe(
        name="lchs_compiled_recovery_scale_drop",
        module="nwqlib.algorithms.lchs.parameters",
        qualname="select_parameters",
        old="qsp_recovery=prepared.recovery_scale,",
        new="qsp_recovery=1.0,",
        pytest_args=(
            "tests/test_lchs_qsp_source.py::test_actual_qsp_source_binds_elapsed_diagonals_recovery_and_physical_phase",
        ),
    ),
    # The periodic Strang bound is B/r**2 with B one third of the weighted
    # branch sum. B without the third selects too many steps.
    single_replacement_probe(
        name="lchs_periodic_strang_step_selection_drops_third",
        module="nwqlib.algorithms.lchs.periodic",
        qualname="select_periodic_parameters",
        old="    exact = Fraction(elapsed)**3/3*sum(",
        new="    exact = Fraction(elapsed)**3*sum(",
        pytest_args=(
            "tests/test_lchs_periodic.py::test_periodic_synthesis_tolerance_selects_the_smallest_strang_step_count",
        ),
    ),
    # Change the selected inverse polynomial after rescale selection. The
    # public host action uses an independent supplied diagonal eigensystem.
    single_replacement_probe(
        name="qls_inverse_coefficient_x1_01",
        module="nwqlib.algorithms.qls.method",
        qualname="_realize",
        old="    coefficients = np.asarray(r.polynomial.coefficients) / r.polynomial.rescale\n",
        new="    coefficients = np.asarray(r.polynomial.coefficients) / r.polynomial.rescale\n"
        "    coefficients[1] *= 1.01\n",
        pytest_args=(
            "tests/test_qls_primary.py::test_selected_polynomial_exact_shape_independent_eigensystem",
        ),
    ),
    single_replacement_probe(
        name="qls_inverse_candidate_certificate_disabled",
        module="nwqlib.subroutines.qsp.inverse",
        qualname="_fit_inverse_chebyshev",
        old=(
            "        if certificate <= epsilon_inv:\n"
            "            passing = (degree, coefficients, certificate, node_count)\n"
            "            break\n"
        ),
        new=("        passing = (degree, coefficients, certificate, node_count)\n        break\n"),
        pytest_args=(
            "tests/test_qls_workflow.py::test_fit_integer_degree_anchors_with_recorded_margin",
        ),
    ),
    single_replacement_probe(
        name="qls_shortcut_dilation_missing_child_query",
        module="nwqlib.algorithms.qls.quantum",
        qualname="_program",
        old='                g_query("dilation_one", controls=((dilation, 1),)),\n',
        new='',
        pytest_args=(
            "tests/test_qls_quantum.py::test_actual_query_projector_and_phase_populations_match_independent_counts",
        ),
    ),
    # The admitting ceiling counts expression evaluations as well as
    # lifecycle steps. Dropping them reports a value that still refuses.
    single_replacement_probe(
        name="qls_admission_requirement_drops_lowering_arguments",
        module="nwqlib.algorithms.qls.quantum",
        qualname="_lowering_admission_work",
        old="            admission.tick(2 * len(node.arguments))\n",
        new="            pass\n",
        pytest_args=(
            "tests/test_qls_quantum.py::test_admission_refusal_names_a_value_that_admits_preparation_and_lowering",
        ),
    ),
    single_replacement_probe(
        name="kr_eta_x1_01",
        module="nwqlib.subroutines.qsp.shortcut",
        qualname="dalzell_eta_from_precision",
        old="    return epsilon_inv / sqrt(2.0)\n",
        new="    return 1.01 * epsilon_inv / sqrt(2.0)\n",
        pytest_args=(
            "tests/test_qls_workflow.py::"
            "test_shortcut_eq17_window_arithmetic_and_poisson_anchor",
        ),
    ),
    single_replacement_probe(
        name="jw_phase_string_sign_flip",
        module="nwqlib.operators._fermion",
        qualname="_ladder_image",
        old="    y_coefficient = -0.5j if creation else 0.5j\n",
        new="    y_coefficient = 0.5j if creation else 0.5j\n",
        pytest_args=("tests/test_fermionic_pool.py::test_native_jw_creation_phase_content",),
    ),
    # Swapping the annihilation order of the UCCSD double negates its Eq. E5
    # generator, so the independent Eq. E7 closed form catches it.
    single_replacement_probe(
        name="pool_doubles_index_swap",
        module="nwqlib.subroutines.fermionic_pool",
        qualname="enumerate_uccsd_sd_pool",
        old="                            base_terms=((1.0, ((a, 1), (b, 1), (i, 0), (j, 0))),),\n",
        new="                            base_terms=((1.0, ((a, 1), (b, 1), (j, 0), (i, 0))),),\n",
        pytest_args=(
            "tests/test_fermionic_pool.py::"
            "test_uccsd_pool_members_match_e4_e5_generators_and_closed_forms",
        ),
    ),
    # Omitting the energy subtraction corrupts the explicitly requested
    # direct residual, even for an identity shift of the Hamiltonian.
    single_replacement_probe(
        name='adapt_direct_residual_drop_energy',
        module='nwqlib.algorithms.gcim.adapt_verification',
        qualname='verify',
        old='- result.eigenvalue * state',
        new='- 0.0 * state',
        pytest_args=('tests/test_adapt_verification.py::test_zero_residual_of_excited_state_does_not_identify_ground',),
    ),
    # Reversing the circuit-model commutator product order negates [H, A_i],
    # so the independent coefficient-only sign relation catches it.
    single_replacement_probe(
        name='adapt_circuit_gradient_commutator_sign',
        module='nwqlib.algorithms.gcim.adapt_inputs',
        qualname='_commutator_arrays',
        old='        x, z, phase = product_words(h.x[hi], h.z[hi], a.x[ai], a.z[ai])\n',
        new='        x, z, phase = product_words(a.x[ai], a.z[ai], h.x[hi], h.z[hi])\n',
        pytest_args=('tests/test_adapt_gcim.py::test_processed_commutator_sign_at_subnormal_squared_scale',),
    ),
    # Widening the UCCSD virtual-orbital
    # list to every spin orbital breaks the occupied->virtual restriction,
    # so the hand-computed combinatorial pool-size pins catch the inflation.
    single_replacement_probe(
        name="uccsd_occ_virt_restriction_break",
        module="nwqlib.subroutines.fermionic_pool",
        qualname="_occupation_sets",
        old="    virtual = tuple(mode for mode, bit in enumerate(bits) if bit == 0)\n",
        new="    virtual = tuple(mode for mode, bit in enumerate(bits))\n",
        pytest_args=(
            "tests/test_fermionic_pool.py::test_uccsd_pool_size_pins_from_hand_combinatorics",
        ),
    ),
    single_replacement_probe(
        name="qeb_z_ladder_injected",
        module="nwqlib.operators._fermion",
        qualname="_ladder_image",
        old='    parity = mapping == "jw"\n',
        new='    parity = True\n',
        pytest_args=(
            "tests/test_factorized_operators.py::"
            "test_fermion_order_exact_cancellation_and_jw_sign",
        ),
    ),
    single_replacement_probe(
        name="ceo_ovp_drop_crossed_qe",
        module="nwqlib.subroutines.fermionic_pool",
        qualname="enumerate_ceo_ovp_pool",
        old='                        pauli = _sum_paulis((direct, tuple((p, sign * c) for p, c in crossed)))\n',
        new='                        pauli = _sum_paulis((direct,))\n',
        pytest_args=(
            "tests/test_fermionic_pool.py::"
            "test_ceo_ovp_plus_minus_match_independent_dense_direct_crossed_identity",
        ),
    ),
    single_replacement_probe(
        name="spin_squared_ladder_sign_flip",
        module="nwqlib.subroutines.fermionic_pool",
        qualname="apply_spin_squared",
        old="        minus_terms.append((1.0, ((down_p, 1), (up_p, 0))))\n",
        new="        minus_terms.append((-1.0, ((down_p, 1), (up_p, 0))))\n",
        pytest_args=(
            "tests/test_adapt_gcim_efficiency_contract.py::"
            "test_matrix_free_spin_squared_action_matches_validation_matrix",
        ),
    ),
    single_replacement_probe(
        name="gcim_active_space_core_energy_drop",
        module="nwqlib.algorithms.gcim.chemistry",
        qualname="build_gcim_chemistry_problem",
        old="        constant_energy = float(core_energy)\n",
        new="        constant_energy = nuclear_repulsion\n",
        pytest_args=(
            "tests/test_gcim_chemistry.py::test_lih_cas22_active_space_matches_pyscf_casci",
        ),
        requires=("pyscf", "openfermion"),
    ),
    single_replacement_probe(
        name='gcim_gradient_pool_union_dedup_drop',
        module='nwqlib.algorithms.gcim.adapt_inputs',
        qualname='prepare_adapt_inputs',
        old='    screen_groups, energy_groups = groups(screen), groups(energy)\n',
        new='    screen_groups, energy_groups = (), groups(energy)\n',
        pytest_args=("tests/test_gcim_native_chain.py::test_tiny_actual_adapt_consumer_keeps_native_generator_basis_and_sampled_provenance",),
    ),
    single_replacement_probe(
        name='adapt_gradient_x1_01',
        module='nwqlib.algorithms.gcim.adapt_acquisition',
        qualname='gradients_from_column',
        old='2.0 * np.vdot(h0_state, apply_generator(generator, state)).real',
        new='2.02 * np.vdot(h0_state, apply_generator(generator, state)).real',
        pytest_args=('tests/test_adapt_primary.py::test_product_screen_and_tie_survive_exact_ritz_solution[classical]',),
    ),
    # Without the stage admission a circuit limit refuses a quantum ADAPT
    # matrix stage only at the query that exceeds it, after part of the stage.
    single_replacement_probe(
        name="adapt_matrix_stage_admission_drop",
        module="nwqlib.algorithms.gcim.adapt_acquisition",
        qualname="drive_adapt",
        old="                    run.check_capacity(preparations=count, circuits=count, "
            "shots=count * (plan.shots or 0))\n",
        new="                    pass\n",
        pytest_args=(
            "tests/test_adapt_primary.py::"
            "test_matrix_stage_limit_refuses_before_its_first_query_and_extension_resumes_it",
        ),
        requires=("pyscf", "openfermion"),
    ),
    # A Y-action phase error corrupts both action and exponential paths.
    single_replacement_probe(
        name="adapt_matrix_free_y_phase_flip",
        module="nwqlib.operators._pauli",
        qualname="apply_terms",
        old="    rotated = [_rotate(complex(c), -int(count))\n",
        new="    rotated = [_rotate(complex(c), int(count))\n",
        pytest_args=(
            "tests/test_adapt_gcim_efficiency_contract.py::"
            "test_matrix_free_generator_action_and_exponential_match_dense_reference",
        ),
    ),
    single_replacement_probe(
        name="lchs_numerical_psd_gate_drop",
        module="nwqlib.algorithms.lchs.solution_error_budget",
        qualname="application_stage_bounds",
        old="        if not psd_premise_satisfied:\n",
        new="        if False and not psd_premise_satisfied:\n",
        pytest_args=(
            "tests/test_lchs_solution_error_budget.py::"
            "test_stage_bound_suppression_and_invalid_inputs",
        ),
    ),
    # Composite Gauss now ends exactly at K; summaries project the selected
    # provider rather than reevaluating a bound at a second effective cutoff.
    # Protect the actual provider bound consumed by the public reconstruction.
    single_replacement_probe(
        name="lchs_provider_quadrature_bound_drop",
        module="nwqlib.algorithms.lchs.providers",
        qualname="_build_composite_gauss",
        old="        quadrature_error_bound=quadrature_bound,\n",
        new="        quadrature_error_bound=0.0,\n",
        pytest_args=(
            "tests/test_lchs_providers.py::test_default_pair_primary_record_preserves_provider_values",
        ),
    ),
    single_replacement_probe(
        name="lchs_duhamel_growth_drop",
        module="nwqlib.algorithms.lchs.inhomogeneous_theory",
        qualname="_duhamel_quadrature_error_bound",
        old="        + rho_plus * float(final_time)\n",
        new="        + 0.0 * rho_plus * float(final_time)\n",
        pytest_args=('tests/test_lchs_numerics.py::test_duhamel_remainder_uses_physical_growth_and_exact_zero_routes',),
    ),
    single_replacement_probe(
        name="qsp_complete_analytic_tail_drop",
        module="nwqlib.subroutines.qsp.evolution",
        qualname="jacobi_anger_expansion",
        old="    combined_analytic_tail = 2.0 * parity_analytic_tail\n",
        new="    combined_analytic_tail = 0.0 * parity_analytic_tail\n",
        pytest_args=(
            "tests/test_qsp_evolution.py::"
            "test_jacobi_anger_records_complete_finite_and_analytic_tail",
        ),
    ),
    single_replacement_probe(
        name="qsp_child_physical_error_drop",
        module="nwqlib.subroutines.qsp.evolution",
        qualname="qsp_evolution_error_terms",
        old=("    child_error = (0.0 if evolution_time == 0 else None if child_error_bound is None\n"
             "                   else abs(float(evolution_time)) * float(child_error_bound))\n"),
        new="    child_error = 0.0\n",
        pytest_args=(
            "tests/test_qsp_evolution.py::"
            "test_qsp_child_operator_error_uses_physical_time_before_oaa",
        ),
    ),
    single_replacement_probe(
        name="lchs_pf_prune_cost_drop",
        module="nwqlib.algorithms.lchs.time_independent_terms",
        qualname="_budgeted_trotter_node_record",
        old="    prune_cost = upper_pruning_error(final_time, combined.pruned_l1_mass)\n",
        new="    prune_cost = 0.0\n",
        pytest_args=('tests/test_lchs_numerics.py::test_pruned_mass_and_empty_branch_accounting',),
    ),
    single_replacement_probe(
        name="lchs_qsp_gamma_drop",
        module="nwqlib.algorithms.lchs.solution_error_budget",
        qualname="_solution_stage_value",
        old="        return _stage_product(name, gamma, compensated_recovery_error_bound)\n",
        new="        return _stage_product(name, compensated_recovery_error_bound)\n",
        pytest_args=('tests/test_lchs_solution_error_budget.py::test_stage_bound_manifest_and_composition',),
    ),
    single_replacement_probe(
        name="direct_state_preparation_uniform_exactness_drop",
        module="nwqlib.subroutines.state_preparation.direct",
        qualname="_build_normalized_state_preparation",
        old=(
            "    elif nonzero.size == normalized_state.size and np.all(\n"
            "        normalized_state == common_amplitude\n"
            "    ):\n"
        ),
        new=(
            "    elif nonzero.size == normalized_state.size and np.allclose(\n"
            "        normalized_state, common_amplitude\n"
            "    ):\n"
        ),
        pytest_args=(
            "tests/test_mps_state_preparation.py::"
            "test_direct_preparation_exact_classifier_rejects_approximate_uniform_states",
        ),
    ),
    # Controlled dense definitions must use the exact synthesis. Letting Qiskit
    # unroll the nested UnitaryGate, or snapping Weyl coordinates at a
    # fidelity-scale window, brings back entry errors of order t.
    single_replacement_probe(
        name="controlled_dense_rewrite_drop",
        module="nwqlib.subroutines.qiskit_compat",
        qualname="controlled",
        old="    gate = _exact_dense_definitions(gate, {})\n",
        new="    pass\n",
        pytest_args=(
            "tests/test_scientist_lchs.py::"
            "test_dense_select_branches_match_exact_controlled_exponentials",
        ),
    ),
    # A directly controlled UnitaryGate must use the exact synthesis of its
    # controlled matrix. Qiskit's UnitaryGate.control keeps errors up to its
    # allclose tolerance and raises for RX(1e-7) with three controls.
    single_replacement_probe(
        name="controlled_unitary_exact_drop",
        module="nwqlib.subroutines.qiskit_compat",
        qualname="controlled",
        old="        return _controlled_unitary(gate, num_ctrl_qubits, ctrl_state)\n",
        new="        return gate.control(num_ctrl_qubits, ctrl_state=ctrl_state, annotated=False)\n",
        pytest_args=(
            "tests/test_lcu_subroutines.py::"
            "test_dense_select_is_exact_for_small_angle_branches",
        ),
    ),
    # The dense LCU and QPE admissions must charge the branch synthesis
    # before it starts.
    single_replacement_probe(
        name="lcu_select_synthesis_admission_drop",
        module="nwqlib.subroutines.lcu.core",
        qualname="_admit_lcu",
        old="    synthesized = n if a else 0\n",
        new="    synthesized = 0\n",
        pytest_args=(
            "tests/test_lcu_subroutines.py::"
            "test_dense_select_admission_counts_branch_synthesis_before_it_starts",
        ),
    ),
    single_replacement_probe(
        name="qpe_dense_power_synthesis_admission_drop",
        module="nwqlib.algorithms.qpe.powers",
        qualname="_dense_block",
        old="    work = matrix_work + 8 * dimension**2 + synthesis_work\n",
        new="    work = matrix_work + 8 * dimension**2 + 0 * synthesis_work\n",
        pytest_args=(
            "tests/test_dense_synthesis.py::"
            "test_dense_qpe_admission_charges_the_controlled_synthesis",
        ),
    ),
    single_replacement_probe(
        name="dense_weyl_zero_window_1e5",
        module="nwqlib.subroutines._dense_synthesis",
        qualname="_reduced_weyl",
        old="    zero = np.abs(coords) <= ROUNDING_WINDOW\n",
        new="    zero = np.abs(coords) <= 1e-5\n",
        pytest_args=(
            "tests/test_dense_synthesis.py::"
            "test_near_special_two_qubit_unitary_keeps_its_entangling_part",
        ),
    ),
    # A block that passes no diagonal on must leave the next block without
    # one, also when the batched A.2 loop discards the rest of its batch,
    # and the frames of the discarded blocks must be computed again. The
    # public exactness test reaches both through Kronecker products.
    single_replacement_probe(
        name="dense_a2_stale_diagonal_after_restart",
        module="nwqlib.subroutines._dense_synthesis",
        qualname="_apply_a2",
        old="            delta, run = None, 0\n",
        new="            run = 0\n",
        pytest_args=(
            "tests/test_dense_synthesis.py::"
            "test_a2_batches_restart_after_a_block_that_keeps_its_own_circuit",
            "tests/test_dense_synthesis.py::test_dense_unitary_circuit_is_exact_to_rounding[5-kronecker]",
            "tests/test_dense_synthesis.py::test_dense_unitary_circuit_is_exact_to_rounding[6-kronecker]",
        ),
    ),
    single_replacement_probe(
        name="dense_a2_discarded_frames_kept",
        module="nwqlib.subroutines._dense_synthesis",
        qualname="_apply_a2",
        old="        accepted = int(failed[0]) + 1 if len(failed) else size\n",
        new="        accepted = size\n",
        pytest_args=(
            "tests/test_dense_synthesis.py::"
            "test_a2_batches_restart_after_a_block_that_keeps_its_own_circuit",
            "tests/test_dense_synthesis.py::test_dense_unitary_circuit_is_exact_to_rounding[5-kronecker]",
            "tests/test_dense_synthesis.py::test_dense_unitary_circuit_is_exact_to_rounding[6-kronecker]",
        ),
    ),
    # Each probe below weakens the work law of the exact dense synthesis, the
    # count of syntheses, one charge or one limit check. Its test states the
    # boundary with the law written out independently of dense_synthesis_size.
    single_replacement_probe(
        name="dense_synthesis_work_law_half_cubic_term",
        module="nwqlib.subroutines._dense_synthesis",
        qualname="dense_synthesis_size",
        old="    work = 113 * size**3 // 4 + (num_qubits**2 + 16 * num_qubits + 512) * size**2\n",
        new="    work = 113 * size**3 // 8 + (num_qubits**2 + 16 * num_qubits + 512) * size**2\n",
        pytest_args=(
            "tests/test_qls_quantum.py::test_controlled_dense_query_synthesis_is_admitted_at_planning",
        ),
    ),
    single_replacement_probe(
        name="qls_query_synthesis_admission_drop",
        module="nwqlib.algorithms.qls.host_planning",
        qualname="_admit_query_synthesis",
        old="            widths += 2 * [dimension.bit_length()]\n",
        new="            widths += []\n",
        pytest_args=(
            "tests/test_qls_quantum.py::test_controlled_dense_query_synthesis_is_admitted_at_planning",
        ),
    ),
    single_replacement_probe(
        name="lchs_dense_select_synthesis_admission_drop",
        module="nwqlib.algorithms.lchs.parameters",
        qualname="construction_work",
        old="            work += physical*(synthesis_work+control_work)\n",
        new="            work += physical*control_work\n",
        pytest_args=(
            "tests/test_lchs_resource_structural_law.py::test_dense_exact_select_charges_the_syntheses_it_makes",
        ),
    ),
    single_replacement_probe(
        name="lchs_compiled_select_synthesis_admission_drop",
        module="nwqlib.algorithms.lchs.parameters",
        qualname="construction_work",
        old="            work += sum(size[0] for size in sizes)\n",
        new="            work += 0\n",
        pytest_args=(
            "tests/test_lchs_resource_structural_law.py::"
            "test_compiled_qsp_select_charges_the_syntheses_of_its_dense_children",
        ),
    ),
    # Each dense branch, node or representative eigensystem is charged the
    # spectral law of time_independent_terms, not one D-square product.
    single_replacement_probe(
        name="lchs_dense_select_exponential_admission_one_product",
        module="nwqlib.algorithms.lchs.parameters",
        qualname="construction_work",
        old="        work = branch_work + (len(calls)-len(timed) if not identity else len(calls))*dimension**2\n",
        new="        work = (eigensystems+len(timed))*dimension**3 + (len(calls)-len(timed) if not identity else len(calls))*dimension**2\n",
        pytest_args=(
            "tests/test_lchs_resource_structural_law.py::test_dense_exact_select_charges_the_syntheses_it_makes",
        ),
    ),
    single_replacement_probe(
        name="lchs_host_exponential_work_one_product",
        module="nwqlib.algorithms.lchs.host",
        qualname="_classical_work",
        old="        inputs=inputs,eigensystems=eigensystems)\n",
        new="        inputs=inputs,eigensystems=0)\n",
        pytest_args=(
            "tests/test_lchs_primary.py::test_dense_host_admits_every_node_exponential_before_planning_succeeds",
        ),
    ),
    single_replacement_probe(
        name="lchs_grid_exponential_admission_one_product",
        module="nwqlib.algorithms.lchs.selected_grid",
        qualname="grid_reference",
        old="        dense_work(checks, counts, work=work, peak_bytes=phase_bytes+80*d)\n",
        new="        dense_work(checks, counts, work=eigensystems*d**3, peak_bytes=phase_bytes+80*d)\n",
        pytest_args=(
            "tests/test_lchs_verification.py::test_selected_grid_admits_its_node_exponentials_before_the_first",
        ),
    ),
    # A spectral norm is charged the bidiagonalization of its values-only
    # SVD, not one D-square product.
    single_replacement_probe(
        name="lchs_duhamel_norm_svd_one_product",
        module="nwqlib.algorithms.lchs.inhomogeneous_theory",
        qualname="_duhamel_quadrature_error_bound",
        old="    work = 8*d+32+(singular_values_work(d)+4*d*d if matrix_norm is None else 0)\n",
        new="    work = 8*d+32+(d**3+4*d*d if matrix_norm is None else 0)\n",
        pytest_args=(
            "tests/test_lchs_numerics.py::test_duhamel_remainder_admits_the_singular_values_of_its_norm",
        ),
    ),
    single_replacement_probe(
        name="lchs_refinement_norm_svd_one_product",
        module="nwqlib.algorithms.lchs.refinement",
        qualname="_spectral_norm",
        old="        dense_work(checks, counts, work=singular_values_work(d), peak_bytes=64*d*d)\n",
        new="        dense_work(checks, counts, work=d**3, peak_bytes=64*d*d)\n",
        pytest_args=(
            "tests/test_lchs_verification.py::test_refinement_spectral_norm_admits_its_singular_values",
        ),
    ),
    single_replacement_probe(
        name="lchs_fixed_pf_dense_norm_one_product",
        module="nwqlib.algorithms.lchs.time_independent_terms",
        qualname="_fixed_trotter_certificate_records",
        old="            work = products*p*d**3+order*p*singular_values_work(d)+24*p*d*d\n",
        new="            work = products*p*d**3+order*p*d**3+24*p*d*d\n",
        pytest_args=(
            "tests/test_lchs_verification.py::"
            "test_fixed_pf_dense_commutators_admit_the_singular_values_of_their_norms",
        ),
    ),
    single_replacement_probe(
        name="qls_reference_spectrum_svd_one_product",
        module="nwqlib.algorithms.qls.verification",
        qualname="_verify",
        old="            + (singular_values_work(d) + 3 * d * d) * int(spectrum))\n",
        new="            + (d**3 + 3 * d * d) * int(spectrum))\n",
        pytest_args=(
            "tests/test_qls_primary.py::test_reference_spectrum_admits_the_singular_values_it_computes",
        ),
    ),
    # A spectral norm without singular vectors is charged its singular
    # values, and a full SVD whose frames are kept is charged 8 D**3.
    single_replacement_probe(
        name="qls_general_spectrum_full_svd_charge",
        module="nwqlib.algorithms.qls.host_planning",
        qualname="_spectrum",
        old="            work, data = singular_values_work(d) + 8 * d * d, 16 * (5 * d * d + 3 * d)\n",
        new="            work, data = 8 * d**3 + 8 * d * d, 16 * (5 * d * d + 3 * d)\n",
        pytest_args=(
            "tests/test_qls_primary.py::test_general_spectrum_admits_the_singular_values_without_vectors",
        ),
    ),
    single_replacement_probe(
        name="block_encoding_norm_full_svd_charge",
        module="nwqlib.subroutines.block_encoding.core",
        qualname="_dense_norm_work",
        old="    return 8 * dimension**3 if full_svd else singular_values_work(dimension)\n",
        new="    return 8 * dimension**3\n",
        pytest_args=(
            "tests/test_block_encoding.py::test_dense_dilation_planning_admits_the_singular_values_of_its_norm",
            "tests/test_block_encoding.py::test_pauli_dense_dilation_admits_the_singular_values_of_its_norm",
        ),
    ),
    single_replacement_probe(
        name="block_encoding_full_svd_norm_values_charge",
        module="nwqlib.subroutines.block_encoding.core",
        qualname="_dense_norm_work",
        old="    return 8 * dimension**3 if full_svd else singular_values_work(dimension)\n",
        new="    return singular_values_work(dimension)\n",
        pytest_args=(
            "tests/test_block_encoding.py::test_dense_dilation_planning_admits_the_singular_values_of_its_norm",
        ),
    ),
    single_replacement_probe(
        name="block_encoding_explicit_norm_full_svd_wiring",
        module="nwqlib.subroutines.block_encoding.core",
        qualname="_plan_block_encoding",
        old="            normalization=normalization, dense_norm=dense_norm, full_svd=full_svd_requested,\n",
        new="            normalization=normalization, dense_norm=dense_norm, full_svd=True,\n",
        pytest_args=(
            "tests/test_block_encoding.py::test_dense_dilation_planning_admits_the_singular_values_of_its_norm",
        ),
    ),
    single_replacement_probe(
        name="block_encoding_pauli_conversion_norm_full_svd_wiring",
        module="nwqlib.subroutines.block_encoding.core",
        qualname="_plan_block_encoding",
        old="        norm_work = _dense_norm_work(d, full_svd_requested) if dense_alpha is None else 0\n",
        new="        norm_work = _dense_norm_work(d, True) if dense_alpha is None else 0\n",
        pytest_args=(
            "tests/test_block_encoding.py::test_pauli_dense_dilation_admits_the_singular_values_of_its_norm",
        ),
    ),
    single_replacement_probe(
        name="block_encoding_pauli_normalization_full_svd_wiring",
        module="nwqlib.subroutines.block_encoding.core",
        qualname="_plan_block_encoding",
        old="        if work_used + _dense_norm_work(d, full_svd_requested) > max_work:\n",
        new="        if work_used + _dense_norm_work(d, True) > max_work:\n",
        pytest_args=(
            "tests/test_block_encoding.py::test_pauli_dense_dilation_admits_the_singular_values_of_its_norm",
        ),
    ),
    # The guard around onenormest must return NumPy's global generator to the
    # caller's state.
    single_replacement_probe(
        name="norm_estimate_state_restore_drop",
        module="nwqlib._linalg_laws",
        qualname="seeded_norm_estimates",
        old="        np.random.set_state(state)\n",
        new="        pass\n",
        pytest_args=(
            "tests/test_qhd_workflow.py::test_classical_host_evolution_keeps_the_global_random_state",
            "tests/test_lchs_verification.py::test_triangular_reference_exponential_keeps_the_global_random_state",
        ),
    ),
    # Each expm_multiply call of the classical QHD kernel is charged its
    # Taylor and norm-estimation products, not one visit of the state.
    single_replacement_probe(
        name="qhd_expm_multiply_work_one_visit",
        module="nwqlib.algorithms.qhd.method",
        qualname="restricted_sizes",
        old="        size += 2 * entries + 5 * dimension + work\n",
        new="        size += 2 * entries + 5 * dimension + dimension\n",
        pytest_args=(
            "tests/test_qhd_workflow.py::test_classical_host_admits_the_norm_estimation_of_expm_multiply",
        ),
    ),
    # Box refinement grows each axis interval toward the larger marginal
    # neighbor and keeps half a cell beyond each kept grid point. The box-rule
    # test recomputes both from the recorded marginals.
    single_replacement_probe(
        name="qhd_refinement_interval_toward_smaller_neighbor",
        module="nwqlib.algorithms.qhd.refinement",
        qualname="_axis_interval",
        old="        if right is None or (left is not None and left >= right):\n",
        new="        if right is None or (left is not None and left <= right):\n",
        pytest_args=(
            "tests/test_qhd_refinement.py::"
            "test_next_box_follows_the_recorded_marginals_under_the_centered_rule",
        ),
    ),
    single_replacement_probe(
        name="qhd_refinement_full_cell_beyond_kept_point",
        module="nwqlib.algorithms.qhd.refinement",
        qualname="_next_box",
        old="        lower = a if first == 0 else _face(x[first - 1], x[first])\n",
        new="        lower = a if first == 0 else x[first - 1]\n",
        pytest_args=(
            "tests/test_qhd_refinement.py::"
            "test_next_box_follows_the_recorded_marginals_under_the_centered_rule",
        ),
    ),
    # The joint box mass bound follows from the union bound. Neither eta nor
    # the product of the axis masses bounds the joint mass of a correlated
    # distribution.
    single_replacement_probe(
        name="qhd_refinement_joint_bound_is_eta",
        module="nwqlib.algorithms.qhd.refinement",
        qualname="_refine",
        old="            joint_mass_bound=_joint_mass_bound(masses),\n",
        new="            joint_mass_bound=options.mass_threshold,\n",
        pytest_args=(
            "tests/test_qhd_refinement.py::"
            "test_joint_mass_bound_is_attained_and_is_neither_eta_nor_the_product",
        ),
    ),
    single_replacement_probe(
        name="qhd_refinement_joint_bound_is_axis_product",
        module="nwqlib.algorithms.qhd.refinement",
        qualname="_refine",
        old="            joint_mass_bound=_joint_mass_bound(masses),\n",
        new="            joint_mass_bound=__import__('math').prod(masses),\n",
        pytest_args=(
            "tests/test_qhd_refinement.py::"
            "test_joint_mass_bound_is_attained_and_is_neither_eta_nor_the_product",
        ),
    ),
    # A stall split continues in the region whose most probable point has the
    # lower relative objective, and splits the axis with the largest ratio of
    # the smaller peak to the valley. The split test states both from an exact
    # distribution where the lower-scoring region holds less probability and
    # another axis has a valley of smaller mass.
    single_replacement_probe(
        name="qhd_refinement_split_scores_with_wrong_sign",
        module="nwqlib.algorithms.qhd.refinement",
        qualname="_stall_split",
        old="    scores, relative = zip(*(score(indices) for indices in modes))\n",
        new="    scores, relative = zip(*((value, -r) for value, r in (score(indices) for indices in modes)))\n",
        pytest_args=(
            "tests/test_qhd_refinement.py::"
            "test_stall_split_follows_the_end_mass_lemma_and_the_clearest_valley",
        ),
    ),
    # A valley that the readout cannot resolve takes no part in the split.
    single_replacement_probe(
        name="qhd_refinement_split_without_resolution_screen",
        module="nwqlib.algorithms.qhd.refinement",
        qualname="_clearest_valley",
        old="            if p[v] < minor and admits(axis, v):\n",
        new="            if p[v] < minor:\n",
        pytest_args=(
            "tests/test_qhd_refinement.py::"
            "test_stall_split_follows_the_end_mass_lemma_and_the_clearest_valley",
        ),
    ),
    single_replacement_probe(
        name="qhd_refinement_split_at_smallest_valley_mass",
        module="nwqlib.algorithms.qhd.refinement",
        qualname="_clearest_valley",
        old="                key = (separation, Fraction(minor))\n",
        new="                key = (-Fraction(p[v]), Fraction(minor))\n",
        pytest_args=(
            "tests/test_qhd_refinement.py::"
            "test_stall_split_follows_the_end_mass_lemma_and_the_clearest_valley",
        ),
    ),
    # The one-hot rotation law, the budgeted T estimate and the splitting
    # bounds. The first three count the emitted circuit wrongly: the often
    # used 2**s - 1 projector law, which misses the MCX rotations of Qiskit's
    # MCPhase from s = 8, the rotations that rotation_threshold removed, and
    # K - 1 chain links for a vector with a zero tail. The last two drop the
    # link split, which a constant potential isolates.
    single_replacement_probe(
        name="qhd_rotation_law_projector_power_of_two",
        module="nwqlib.algorithms.qhd.resources",
        qualname="_projector_slots",
        old="    return slots, fixed_arbitrary, fixed_t\n",
        new="    return {a / 2 ** (s - 1): 2**s - 1}, 0, 0\n",
        pytest_args=("tests/test_qhd_resources.py::test_projector_rotations_match_the_qiskit_definitions",),
    ),
    single_replacement_probe(
        name="qhd_rotation_law_counts_pruned_rotations",
        module="nwqlib.algorithms.qhd.resources",
        qualname="rotation_population",
        old="    slots = Counter()\n",
        new="    slots = Counter({1.0: 2 * reconstruction.dropped_count})\n",
        pytest_args=("tests/test_qhd_resources.py::test_rotation_law_counts_the_emitted_circuit_after_pruning",),
    ),
    single_replacement_probe(
        name="qhd_rotation_law_counts_every_chain_link",
        module="nwqlib.algorithms.qhd.resources",
        qualname="rotation_population",
        old="            for theta in chain_angles(alpha):\n",
        new="            for theta in chain_angles(alpha) + [1.0] * (len(alpha) - 1 - len(chain_angles(alpha))):\n",
        pytest_args=("tests/test_qhd_resources.py::test_rotation_law_counts_the_emitted_circuit_after_pruning",),
    ),
    single_replacement_probe(
        name="qhd_t_estimate_undivided_budget",
        module="nwqlib.algorithms.qhd.resources",
        qualname="synthesis_projection",
        old="    rotation_epsilon = _downward(Fraction(remaining) / rotations) if rotations else None\n",
        new="    rotation_epsilon = _downward(Fraction(remaining)) if rotations else None\n",
        pytest_args=("tests/test_qhd_resources.py::test_t_estimate_of_a_small_circuit",),
    ),
    single_replacement_probe(
        name="qhd_clifford_distance_full_angle",
        module="nwqlib.algorithms.qhd.resources",
        qualname="_distance_bound",
        old="    return min(Fraction(2), delta / 2 * (1 - y2 / 6 + y2 * y2 / 120))\n",
        new="    return min(Fraction(2), delta * (1 - y2 / 6 + y2 * y2 / 120))\n",
        pytest_args=("tests/test_qhd_resources.py::test_clifford_distance_and_budgeted_replacement",),
    ),
    # The distance encloses the mathematical multiple of pi/2. Reducing by
    # the binary64 pi alone puts 2**52 binary64 quarter turns at distance
    # zero, which replaces a rotation 0.138 away from any Clifford.
    single_replacement_probe(
        name="qhd_clifford_distance_binary64_pi",
        module="nwqlib.algorithms.qhd.resources",
        qualname="_distance_bound",
        old="    delta = max(abs(exact - k * _PI_LOW / 2), abs(exact - k * _PI_UP / 2))\n",
        new="    delta = abs(exact - k * _PI_LOW / 2)\n",
        pytest_args=("tests/test_qhd_resources.py::test_clifford_distance_and_budgeted_replacement",),
    ),
    # The T estimate takes the negated logarithm of a subnormal allowance,
    # whose reciprocal overflows.
    single_replacement_probe(
        name="qhd_t_estimate_reciprocal_allowance",
        module="nwqlib.algorithms.qhd.resources",
        qualname="synthesis_projection",
        old="        value = population.exact_t + T_PER_PRECISION_BIT * rotations * -log2(rotation_epsilon)\n",
        new="        value = population.exact_t + T_PER_PRECISION_BIT * rotations * log2(1 / rotation_epsilon)\n",
        pytest_args=("tests/test_qhd_resources.py::test_t_estimate_of_a_small_circuit",),
    ),
    # Each probe drops one term of the binary angle-formation bound. The
    # public one-bit Plans of the component test carry exactly that term's
    # discrepancy, so the oracle then exceeds the bound.
    single_replacement_probe(
        name="qhd_binary_formation_drops_walsh_coefficients",
        module="nwqlib.algorithms.qhd.circuit_errors",
        qualname="binary_angle_formation",
        old="                    bound = abs(exact) * radii + eps_x * total_c + _U * abs(x) * total_c\n",
        new="                    bound = eps_x * total_c + _U * abs(x) * total_c\n",
        pytest_args=("tests/test_qhd_resources.py::test_binary_angle_formation_charges_each_component",),
    ),
    # The exponent term eps_x C matters where x = fl(dt a) is inexact, which
    # the walsh_exponent case, with a quadratic weight gamma > 0, provides.
    single_replacement_probe(
        name="qhd_binary_formation_drops_walsh_exponent",
        module="nwqlib.algorithms.qhd.circuit_errors",
        qualname="binary_angle_formation",
        old="                    bound = abs(exact) * radii + eps_x * total_c + _U * abs(x) * total_c\n",
        new="                    bound = abs(exact) * radii + _U * abs(x) * total_c\n",
        pytest_args=("tests/test_qhd_resources.py::test_binary_angle_formation_charges_each_component",),
    ),
    single_replacement_probe(
        name="qhd_binary_formation_drops_walsh_angle_rounding",
        module="nwqlib.algorithms.qhd.circuit_errors",
        qualname="binary_angle_formation",
        old="                    bound = abs(exact) * radii + eps_x * total_c + _U * abs(x) * total_c\n",
        new="                    bound = abs(exact) * radii + eps_x * total_c\n",
        pytest_args=("tests/test_qhd_resources.py::test_binary_angle_formation_charges_each_component",),
    ),
    single_replacement_probe(
        name="qhd_binary_formation_drops_dense_kinetic_energy",
        module="nwqlib.algorithms.qhd.circuit_errors",
        qualname="binary_angle_formation",
        old="                bound = _U * abs(x) * largest + eps_x * largest + abs(exact) * radius\n",
        new="                bound = _U * abs(x) * largest + eps_x * largest\n",
        pytest_args=("tests/test_qhd_resources.py::test_binary_angle_formation_charges_each_component",),
    ),
    single_replacement_probe(
        name="qhd_binary_formation_drops_dense_kinetic_rounding",
        module="nwqlib.algorithms.qhd.circuit_errors",
        qualname="binary_angle_formation",
        old="                bound = _U * abs(x) * largest + eps_x * largest + abs(exact) * radius\n",
        new="                bound = eps_x * largest + abs(exact) * radius\n",
        pytest_args=("tests/test_qhd_resources.py::test_binary_angle_formation_charges_each_component",),
    ),
    # A recorded count without its Plan does not prove an exact law.
    single_replacement_probe(
        name="qhd_run_totals_trust_a_recorded_count",
        module="nwqlib.algorithms.qhd.resources",
        qualname="run_resources",
        old="    upper = encoding == \"binary\" or unproved or any(\n",
        new="    upper = encoding == \"binary\" or any(\n",
        pytest_args=("tests/test_qhd_resources.py::test_run_totals_are_sums_over_the_recorded_circuits",),
    ),
    # The hopping parameter doubles the stored angle's product. Forming 4 dt
    # first overflows for a large step with a small coefficient.
    single_replacement_probe(
        name="qhd_hopping_parameter_four_dt_first",
        module="nwqlib.subroutines.hamiltonian_evolution.pauli_evolution",
        qualname="_hopping_parameter",
        old="    theta = (2.0 * time_step * coefficient) * 2.0\n",
        new="    theta = 4.0 * time_step * coefficient\n",
        pytest_args=("tests/test_qhd_resources.py::test_hopping_parameter_is_twice_the_stored_angle",),
    ),
    # The Gaussian direction charge is the stored aggregate plus the norm
    # allowance, not the sqrt(2) cap.
    single_replacement_probe(
        name="qhd_gaussian_preparation_coarse_charge",
        module="nwqlib.algorithms.qhd.circuit_errors",
        qualname="preparation_error",
        old="        direction = _U * Fraction(reconstruction.initial_state_error) + d * (3 * _U / (1 - 2 * _U) + root * _TAU)\n",
        new="        direction = _SQRT2_UP\n",
        pytest_args=("tests/test_qhd_resources.py::test_gaussian_preparation_entry_bounds_the_prepared_chain",),
    ),
    single_replacement_probe(
        name="qhd_first_order_split_drops_link_commutator",
        module="nwqlib.algorithms.qhd.evolution_bounds",
        qualname="_splitting",
        old="        return alpha * beta * c / 2 + alpha * alpha * gamma / 2\n",
        new="        return alpha * beta * c / 2\n",
        pytest_args=("tests/test_qhd_resources.py::test_evolution_bound_dominates_the_time_ordered_error",),
    ),
    # The time-ordering and midpoint-quadrature terms dropped, which the
    # K = 2 cases of the evolution test isolate.
    single_replacement_probe(
        name="qhd_time_ordering_dropped",
        module="nwqlib.algorithms.qhd.evolution_bounds",
        qualname="_schedule_terms",
        old="    timing = None if c is None else delta ** 3 * (a_max * b1 + b_max * a1) * c / 12\n",
        new="    timing = None if c is None else 0 * delta ** 3 * (a_max * b1 + b_max * a1) * c / 12\n",
        pytest_args=("tests/test_qhd_resources.py::test_evolution_bound_dominates_the_time_ordered_error",),
    ),
    single_replacement_probe(
        name="qhd_midpoint_quadrature_dropped",
        module="nwqlib.algorithms.qhd.evolution_bounds",
        qualname="_schedule_terms",
        old="    quadrature = None if mu_t is None or mu_v is None else delta ** 3 * (a2 * mu_t + b2 * mu_v) / 24\n",
        new="    quadrature = None if mu_t is None or mu_v is None else 0 * delta ** 3 * (a2 * mu_t + b2 * mu_v) / 24\n",
        pytest_args=("tests/test_qhd_resources.py::test_evolution_bound_dominates_the_time_ordered_error",),
    ),
    # The coefficient residual at the step's left end instead of its
    # midpoint, and omitted projectors charged at the full projector norm
    # instead of the traceless one.
    single_replacement_probe(
        name="qhd_coefficient_residual_at_left_endpoint",
        module="nwqlib.algorithms.qhd.evolution_bounds",
        qualname="_coefficient_term",
        old="        a, b = schedule.exact_weights((lower + upper) / 2)\n",
        new="        a, b = schedule.exact_weights(lower)\n",
        pytest_args=("tests/test_qhd_resources.py::test_coefficient_residual_matches_exact_integrals_and_midpoints",),
    ),
    single_replacement_probe(
        name="qhd_omitted_blocks_full_projector_norm",
        module="nwqlib.algorithms.qhd.circuit_errors",
        qualname="block_errors",
        old="        (1 - Fraction(1, 2 ** len(support))) * sum(abs(Fraction(v)) for v in values)\n",
        new="        sum(abs(Fraction(v)) for v in values)\n",
        pytest_args=("tests/test_qhd_resources.py::test_error_ledger_names_every_stage",),
    ),
    single_replacement_probe(
        name="qhd_second_order_split_drops_layer_commutators",
        module="nwqlib.algorithms.qhd.evolution_bounds",
        qualname="_splitting",
        old="    return alpha * alpha * beta * d_t / 12 + alpha * beta * beta * d_v / 24 + alpha ** 3 * (j_e / 12 + j_o / 24)\n",
        new="    return alpha * alpha * beta * d_t / 12 + alpha * beta * beta * d_v / 24\n",
        pytest_args=("tests/test_qhd_resources.py::test_evolution_bound_dominates_the_time_ordered_error",),
    ),
    # The binary probes count b controlled phases per QFT instead of
    # b (b - 1)/2 in the binary law, two phase gates per controlled phase in
    # the angle population instead of three, drop the potential/kinetic
    # commutator of the binary first-order split, and charge binary pruning
    # at the whole dropped angle sum instead of the reconstruction's bound.
    single_replacement_probe(
        name="qhd_binary_qft_rotations_per_qubit",
        module="nwqlib.algorithms.qhd.binary",
        qualname="qft_controlled_phases",
        old="    return sum(bits - r for r in range(1, cutoff + 1))\n",
        new="    return bits\n",
        pytest_args=("tests/test_qhd_resources.py::test_binary_rotations_match_the_emitted_circuit",),
    ),
    single_replacement_probe(
        name="qhd_binary_population_two_phases_per_controlled_phase",
        module="nwqlib.algorithms.qhd.resources",
        qualname="_binary_population",
        old="        slots[magnitude] += 3 * kinetic\n",
        new="        slots[magnitude] += 2 * kinetic\n",
        pytest_args=("tests/test_qhd_resources.py::test_binary_rotations_match_the_emitted_circuit",),
    ),
    single_replacement_probe(
        name="qhd_binary_first_order_split_drops_commutator",
        module="nwqlib.algorithms.qhd.evolution_bounds",
        qualname="_splitting",
        old="            return None if c is None else alpha * beta * c / 2\n",
        new="            return None if c is None else 0 * alpha * beta * c / 2\n",
        pytest_args=("tests/test_qhd_resources.py::test_binary_evolution_bound_dominates_the_time_ordered_error",),
    ),
    single_replacement_probe(
        name="qhd_binary_pruning_at_the_dropped_angle_sum",
        module="nwqlib.algorithms.qhd.resources",
        qualname="_binary_sources",
        old="        entry(\"rotation_pruning\", \"bound\", r.pruning_error_bound,\n",
        new="        entry(\"rotation_pruning\", \"bound\", r.dropped_angle_sum,\n",
        pytest_args=("tests/test_qhd_resources.py::test_binary_error_ledger_takes_the_reconstruction_bounds",),
    ),
    # The binary rotation law shares the one-hot law's selected_logical basis,
    # which the default resource estimate reads. In the cx basis the default
    # estimate reports no binary rotation count.
    single_replacement_probe(
        name="qhd_binary_rotation_law_in_cx_basis",
        module="nwqlib.algorithms.qhd.method",
        qualname="QHD._select_binary_native",
        old='                    metric="arbitrary_rotations", basis="selected_logical", value=sum(b.rotations for b in blocks),\n',
        new='                    metric="arbitrary_rotations", basis="cx", value=sum(b.rotations for b in blocks),\n',
        pytest_args=("tests/test_qhd_resources.py::test_default_estimate_reports_the_rotation_counts_of_both_encodings",),
    ),
    # The phase ledger's contribution count M has one owner. Counting one
    # potential factor for second order undercounts the projector identities
    # that the compiler records.
    single_replacement_probe(
        name="qhd_phase_contributions_one_potential_factor",
        module="nwqlib.algorithms.qhd.method",
        qualname="_phase_contributions",
        old="    factors = 1 if method.trotter_order == 1 else 2\n",
        new="    factors = 1\n",
        pytest_args=("tests/test_qhd_workflow.py::"
                     "test_compensated_ledger_sum_meets_its_bound_where_a_plain_sum_does_not",),
    ),
    # Planning refuses a spectral grid whose Nyquist energy pi**2/(2 h**2)
    # overflows. Without the check, the spacing 1.5e-154 is planned with
    # infinite spectral energies and coefficients.
    single_replacement_probe(
        name="qhd_spectral_nyquist_energy_unchecked",
        module="nwqlib.algorithms.qhd.method",
        qualname="QHD.plan",
        old="                if not isfinite(2.0 * (wave * wave)):\n",
        new="                if False:\n",
        pytest_args=("tests/test_qhd_binary.py::test_spectral_kinetic_refuses_an_overflowing_nyquist_energy",),
    ),
    # Each probe below removes a term of a dense linear-algebra law or one of
    # its charges. Its test compares the charge with a lower bound or a count
    # of the kernel's work, or with the law written out by hand.
    single_replacement_probe(
        name="expm_law_squarings_drop",
        module="nwqlib._linalg_laws",
        qualname="expm_requirements",
        old="    squarings = ceil(log2(norm)) if norm > 1 else 0\n",
        new="    squarings = 0\n",
        pytest_args=(
            "tests/test_lchs_verification.py::test_exponential_work_is_refused_before_the_kernel",
        ),
    ),
    single_replacement_probe(
        name="expm_multiply_law_norm_estimation_drop",
        module="nwqlib._linalg_laws",
        qualname="expm_multiply_requirements",
        old="        products += 968\n",
        new="        products += 0\n",
        pytest_args=(
            "tests/test_qhd_workflow.py::test_classical_host_admits_the_norm_estimation_of_expm_multiply",
        ),
    ),
    single_replacement_probe(
        name="singular_values_law_after_reduction_drop",
        module="nwqlib._linalg_laws",
        qualname="singular_values_work",
        old="    return (4 * dimension**3 + 2) // 3 + 32 * dimension * (dimension + 1)\n",
        new="    return (4 * dimension**3 + 2) // 3\n",
        pytest_args=(
            "tests/test_lchs_verification.py::test_refinement_spectral_norm_admits_its_singular_values",
            "tests/test_lchs_numerics.py::test_duhamel_remainder_admits_the_singular_values_of_its_norm",
        ),
    ),
    single_replacement_probe(
        name="lchs_exponential_law_first_call_only",
        module="nwqlib.algorithms.lchs.time_independent_terms",
        qualname="_spectral_host_requirements",
        old="    work = D + eigensystems*(8*D**3 + 34*D**2) + K*((r+1)*D**2 + (2*r+3)*D + 6*A*D)\n",
        new="    work = D + min(eigensystems, 1)*(8*D**3 + 34*D**2) + K*((r+1)*D**2 + (2*r+3)*D + 6*A*D)\n",
        pytest_args=(
            "tests/test_lchs_primary.py::test_dense_host_admits_every_node_exponential_before_planning_succeeds",
            "tests/test_lchs_verification.py::test_selected_grid_admits_its_node_exponentials_before_the_first",
        ),
    ),
    single_replacement_probe(
        name="qls_classical_eigensystem_one_product",
        module="nwqlib.algorithms.qls.host_planning",
        qualname="original_factor_laws",
        old="        return hermitian_eigensystem_work(d), 16 * d * d + 8 * d, 64 * d * d + 32 * d\n",
        new="        return d**3, 16 * d * d + 8 * d, 64 * d * d + 32 * d\n",
        pytest_args=(
            "tests/test_qls_primary.py::test_classical_model_admits_its_decompositions_with_vectors",
        ),
    ),
    single_replacement_probe(
        name="qls_classical_shortcut_decompositions_one_product",
        module="nwqlib.algorithms.qls.host_planning",
        qualname="selected_work",
        old="(hermitian_eigensystem_work(h) if h else 8 * a**3)",
        new="(h**3 if h else a**3)",
        pytest_args=(
            "tests/test_qls_primary.py::test_classical_model_admits_its_decompositions_with_vectors",
        ),
    ),
    single_replacement_probe(
        name="qpe_dense_power_eigensystem_drop",
        module="nwqlib.algorithms.qpe.powers",
        qualname="_dense_block",
        old="        matrix_work = hermitian_eigensystem_work(dimension) + dimension**3\n",
        new="        matrix_work = dimension**3\n",
        pytest_args=(
            "tests/test_dense_synthesis.py::test_dense_qpe_admission_charges_the_controlled_synthesis",
        ),
    ),
    single_replacement_probe(
        name="qsp_evolution_sine_pass_synthesis_drop",
        module="nwqlib.subroutines.qsp.evolution",
        qualname="build_qsp_evolution_encoding",
        old="    admit_dense_syntheses(dense_synthesis_widths(cos_gate) + dense_synthesis_widths(sin_gate),\n",
        new="    admit_dense_syntheses(dense_synthesis_widths(cos_gate),\n",
        pytest_args=(
            "tests/test_qsp_evolution.py::"
            "test_qsp_evolution_admits_the_dense_syntheses_of_both_passes_before_the_first",
        ),
    ),
    single_replacement_probe(
        name="qsp_generator_branch_synthesis_admission_drop",
        module="nwqlib.subroutines.qsp.evolution",
        qualname="_combine_generator_children",
        old="        admit_dense_syntheses([width for charge in charges for width in charge[0]],\n",
        new="        admit_dense_syntheses([],\n",
        pytest_args=(
            "tests/test_qsp_evolution.py::test_joint_generator_admits_the_syntheses_of_its_controlled_branches",
        ),
    ),
    single_replacement_probe(
        name="coherent_qpe_synthesis_admission_drop",
        module="nwqlib.subroutines.qpe.coherent",
        qualname="build_coherent_qpe_circuit",
        old="    if num_phase_qubits * (dimension**3 + synthesis_work) > integer(max_work, \"max_work\", 1):\n",
        new="    if num_phase_qubits * dimension**3 > integer(max_work, \"max_work\", 1):\n",
        pytest_args=(
            "tests/test_dense_synthesis.py::test_coherent_qpe_admits_its_powers_and_their_controlled_synthesis",
        ),
    ),
    single_replacement_probe(
        name="controlled_dense_encoding_reservation_drop",
        module="nwqlib.blocks.selection",
        qualname="transform_block",
        old="            work += sum(dense_synthesis_size(width)[0] for width in dense_synthesis_widths(circuit))\n",
        new="            work += 0\n",
        pytest_args=(
            "tests/test_semantic_blocks.py::test_controlled_dense_encoding_reserves_its_exact_synthesis",
        ),
    ),
    single_replacement_probe(
        name="basis_lowering_synthesis_admission_drop",
        module="nwqlib.subroutines.qiskit_compat",
        qualname="exact_dense_unitaries",
        old="        if max_work is not None and work > max_work:\n",
        new="        if False:\n",
        pytest_args=(
            "tests/test_dense_synthesis.py::test_basis_lowering_admits_its_syntheses_before_the_first_one",
        ),
    ),
    single_replacement_probe(
        name="dense_synthesis_widths_matrix_key_drop",
        module="nwqlib.subroutines.qiskit_compat",
        qualname="_dense_unitary_widths",
        old='                key = dense_matrix_key(matrix) if cached else ("dense_unitary", matrix.tobytes(), item.label)\n',
        new='                key = dense_matrix_key(matrix) if cached else ("dense_unitary", item.label)\n',
        pytest_args=(
            "tests/test_dense_synthesis.py::test_basis_lowering_admits_its_syntheses_before_the_first_one",
        ),
    ),
    single_replacement_probe(
        name="noisy_aer_run_synthesis_limit_drop",
        module="nwqlib.backends.qiskit_aer",
        qualname="_lower_aer_circuit",
        old="        circuit = (decompose.exact_synthesis or exact_dense_unitaries)(circuit)\n",
        new="        circuit = exact_dense_unitaries(circuit)\n",
        pytest_args=(
            "tests/test_dense_synthesis.py::test_noisy_aer_run_admits_dense_synthesis_against_its_limit",
            "tests/test_synthesis_admission.py::test_noisy_aer_lowering_takes_its_syntheses_from_the_run_cache",
        ),
    ),
    single_replacement_probe(
        name="nwqsim_run_synthesis_limit_drop",
        module="nwqlib.backends.nwqsim",
        qualname="NWQSimBackend._lower",
        old="        native = transpile(run._exact_dense_unitaries(logical) if self.optimization_level <= 1 else logical,\n",
        new=("        native = transpile(__import__('nwqlib.subroutines.qiskit_compat', fromlist=['x'])"
             ".exact_dense_unitaries(logical) if self.optimization_level <= 1 else logical,\n"),
        pytest_args=(
            "tests/test_nwqsim_targets.py::test_prepare_admits_dense_synthesis_against_the_run_limit",
        ),
    ),
    single_replacement_probe(
        name="basis_lowering_run_charge_drop",
        module="nwqlib.subroutines.qiskit_compat",
        qualname="exact_dense_unitaries",
        old="        if charge is not None and work:\n",
        new="        if False:\n",
        pytest_args=(
            "tests/test_dense_synthesis.py::test_basis_lowering_admits_its_syntheses_before_the_first_one",
            "tests/test_dense_synthesis.py::test_noisy_aer_run_admits_dense_synthesis_against_its_limit",
            "tests/test_nwqsim_targets.py::test_prepare_admits_dense_synthesis_against_the_run_limit",
        ),
    ),
    # Each probe below weakens the gate-wise control law, the count of control
    # steps or one charge of it, a supplied circuit's syntheses or the Run's
    # cumulative synthesis limit. The test of a probe that weakens a control
    # charge states the boundary with the census written out and the
    # instructions of each gate kind counted from the installed Qiskit.
    single_replacement_probe(
        name="gatewise_control_work_law_half_gate_term",
        module="nwqlib.subroutines._dense_synthesis",
        qualname="gatewise_control_size",
        old="    return (2048 * gates + 16 * instructions, 1024 * gates + 96 * instructions + 1024 * heavy + 65536,\n",
        new="    return (1024 * gates + 16 * instructions, 1024 * gates + 96 * instructions + 1024 * heavy + 65536,\n",
        pytest_args=(
            "tests/test_qls_quantum.py::test_controlled_dense_query_synthesis_is_admitted_at_planning",
        ),
    ),
    single_replacement_probe(
        name="dense_control_counts_multiplicity_drop",
        module="nwqlib.subroutines.qiskit_compat",
        qualname="dense_control_counts",
        old="                total += occurrences(instruction.operation)\n",
        new="                total |= occurrences(instruction.operation)\n",
        pytest_args=(
            "tests/test_synthesis_admission.py::test_dense_control_counts_follow_every_occurrence",
            "tests/test_qsp_evolution.py::"
            "test_qsp_evolution_admits_the_dense_syntheses_of_both_passes_before_the_first",
        ),
    ),
    single_replacement_probe(
        name="qls_query_control_admission_drop",
        module="nwqlib.algorithms.qls.host_planning",
        qualname="_admit_query_synthesis",
        old="            steps += 2 * [gatewise_control_counts(dimension.bit_length(), controls)]\n",
        new="            steps += []\n",
        pytest_args=(
            "tests/test_qls_quantum.py::test_controlled_dense_query_synthesis_is_admitted_at_planning",
        ),
    ),
    single_replacement_probe(
        name="qls_dilation_shortcut_controls_as_one",
        module="nwqlib.algorithms.qls.host_planning",
        qualname="_admit_query_synthesis",
        old="    controls = 2 if method.solver == \"shortcut_dilation\" else 1\n",
        new="    controls = 1\n",
        pytest_args=(
            "tests/test_qls_quantum.py::test_controlled_dense_query_synthesis_is_admitted_at_planning",
        ),
    ),
    single_replacement_probe(
        name="qls_supplied_rhs_specializations_halved",
        module="nwqlib.algorithms.qls.host_planning",
        qualname="_admit_query_synthesis",
        old="        specializations = 2 if method.solver == \"shortcut_native_svp\" else 4\n",
        new="        specializations = 2\n",
        pytest_args=(
            "tests/test_synthesis_admission.py::"
            "test_qls_shortcut_admits_the_supplied_rhs_syntheses_it_controls",
        ),
    ),
    single_replacement_probe(
        name="lchs_dense_select_control_admission_drop",
        module="nwqlib.algorithms.lchs.parameters",
        qualname="construction_work",
        old="            work += physical*(synthesis_work+control_work)\n",
        new="            work += physical*synthesis_work\n",
        pytest_args=(
            "tests/test_lchs_resource_structural_law.py::test_dense_exact_select_charges_the_syntheses_it_makes",
        ),
    ),
    single_replacement_probe(
        name="lchs_twice_controlled_child_as_once",
        module="nwqlib.algorithms.lchs.compiled_selection",
        qualname="compiled_select_dense_controls",
        old="    twice = [sum(counts) for counts in zip(*(_twice_controlled_dense_counts(width) for width in widths))]\n",
        new="    twice = [sum(counts) for counts in zip(*(gatewise_control_counts(width, 1) for width in widths))]\n",
        pytest_args=(
            "tests/test_lchs_resource_structural_law.py::"
            "test_compiled_qsp_select_charges_the_syntheses_of_its_dense_children",
        ),
    ),
    single_replacement_probe(
        name="qsp_evolution_pass_control_drop",
        module="nwqlib.subroutines.qsp.evolution",
        qualname="build_qsp_evolution_encoding",
        old="                          controls=(dense_control_counts(cos_gate, 1), dense_control_counts(sin_gate, 1)),\n",
        new="                          controls=(),\n",
        pytest_args=(
            "tests/test_qsp_evolution.py::"
            "test_qsp_evolution_admits_the_dense_syntheses_of_both_passes_before_the_first",
        ),
    ),
    single_replacement_probe(
        name="lowering_controlled_transform_charge_drop",
        module="nwqlib.blocks.lowering",
        qualname="_lower_qiskit",
        old="                    if synthesis_charge is not None:\n                        charge_control(result, record.signature.name)\n",
        new="                    pass\n",
        pytest_args=(
            "tests/test_semantic_blocks.py::test_lowering_admits_a_controlled_dense_encoding_before_its_synthesis",
            "tests/test_synthesis_admission.py::"
            "test_run_admits_the_controlled_transform_of_a_supplied_basis_before_its_synthesis",
        ),
    ),
    single_replacement_probe(
        name="run_synthesis_total_drop",
        module="nwqlib._prepared_execution",
        qualname="Run._charge_synthesis",
        old="        if state[\"synthesis_work\"] + work <= limit:\n",
        new="        if work <= limit:\n",
        pytest_args=(
            "tests/test_synthesis_admission.py::"
            "test_run_synthesizes_each_basis_unitary_once_and_again_after_reopening",
        ),
    ),
    single_replacement_probe(
        name="run_refused_preparation_release_drop",
        module="nwqlib._prepared_execution",
        qualname="Run._charge_synthesis",
        old="        if item is not None and item.preparation_id == identity:\n",
        new="        if False:\n",
        pytest_args=(
            "tests/test_synthesis_admission.py::"
            "test_run_synthesizes_each_basis_unitary_once_and_again_after_reopening",
        ),
    ),
    single_replacement_probe(
        name="run_direct_synthesis_limit_drop",
        module="nwqlib._prepared_execution",
        qualname="Run._charge_synthesis",
        old="        if state[\"synthesis_work\"] + work > limit:\n",
        new="        if False:\n",
        pytest_args=(
            "tests/test_synthesis_admission.py::"
            "test_run_synthesizes_each_basis_unitary_once_and_again_after_reopening",
        ),
    ),
    # The Run's synthesis cache: one shared cache for the preparations of a
    # Run and none for a direct backend call, only misses charged, content
    # keys and a copy of the stored circuit for each gate.
    single_replacement_probe(
        name="run_synthesis_cache_drop",
        module="nwqlib._prepared_execution",
        qualname="Run._exact_dense_unitaries",
        old='        cache = {} if state["synthesis_preparation"] is None else state["synthesis_cache"]\n',
        new="        cache = {}\n",
        pytest_args=(
            "tests/test_synthesis_admission.py::"
            "test_run_synthesizes_each_basis_unitary_once_and_again_after_reopening",
        ),
    ),
    single_replacement_probe(
        name="run_direct_synthesis_cached",
        module="nwqlib._prepared_execution",
        qualname="Run._exact_dense_unitaries",
        old='        cache = {} if state["synthesis_preparation"] is None else state["synthesis_cache"]\n',
        new='        cache = state["synthesis_cache"]\n',
        pytest_args=(
            "tests/test_synthesis_admission.py::"
            "test_run_synthesizes_each_basis_unitary_once_and_again_after_reopening",
        ),
    ),
    single_replacement_probe(
        name="synthesis_cache_hit_charged",
        module="nwqlib.subroutines.qiskit_compat",
        qualname="exact_dense_unitaries",
        old="                             if key not in cache))\n",
        new="                             if key is not None))\n",
        pytest_args=(
            "tests/test_synthesis_admission.py::test_synthesis_cache_reuses_the_circuit_that_a_new_synthesis_makes",
        ),
    ),
    single_replacement_probe(
        name="synthesis_cache_identity_key",
        module="nwqlib.subroutines.qiskit_compat",
        qualname="dense_matrix_key",
        old="    return matrix.shape, matrix.dtype.str, matrix.tobytes()\n",
        new="    return matrix.shape, matrix.dtype.str, id(matrix)\n",
        pytest_args=(
            "tests/test_synthesis_admission.py::test_synthesis_cache_reuses_the_circuit_that_a_new_synthesis_makes",
        ),
    ),
    single_replacement_probe(
        name="synthesis_cache_shared_circuit",
        module="nwqlib.subroutines.qiskit_compat",
        qualname="_exact_dense_definitions",
        old="                exact.definition = cache[stored].copy()\n",
        new="                exact.definition = cache[stored]\n",
        pytest_args=(
            "tests/test_synthesis_admission.py::test_synthesis_cache_reuses_the_circuit_that_a_new_synthesis_makes",
        ),
    ),
    # The dense control route: the rule of "auto", the whole-matrix
    # construction and its census, and each owner's whole-matrix charge
    # and CX law.
    single_replacement_probe(
        name="dense_control_route_auto_threshold_two",
        module="nwqlib.subroutines._dense_synthesis",
        qualname="select_dense_control_route",
        old='        return "whole_matrix" if num_controls <= AUTO_WHOLE_MATRIX_MAX_CONTROLS else "gatewise"\n',
        new='        return "whole_matrix" if num_controls <= AUTO_WHOLE_MATRIX_MAX_CONTROLS + 1 else "gatewise"\n',
        pytest_args=(
            "tests/test_dense_control_route.py::test_auto_takes_the_whole_matrix_route_for_one_control_only",
            "tests/test_dense_control_route.py::test_qls_controlled_query_applies_the_dilation_under_its_control_value_on_both_routes",
        ),
    ),
    single_replacement_probe(
        name="controlled_synthesis_census_multiplexor_drop",
        module="nwqlib.subroutines._dense_synthesis",
        qualname="controlled_synthesis_gate_census",
        old='    return {"cx": 2 * block["cx"] + half, "u": 2 * block["u"], "rz": 2 * block["rz"] + half,\n',
        new='    return {"cx": 2 * block["cx"], "u": 2 * block["u"], "rz": 2 * block["rz"] + half,\n',
        pytest_args=(
            "tests/test_dense_control_route.py::test_controlled_synthesis_census_bounds_the_built_whole_matrix_circuit",
        ),
    ),
    single_replacement_probe(
        name="whole_matrix_phase_gate_drop",
        module="nwqlib.subroutines.qiskit_compat",
        qualname="_controlled_whole_matrix",
        old="        circuit.append(PhaseGate(phase) if k == 1 else PhaseGate(phase).control(k - 1), controls)\n",
        new="        pass\n",
        pytest_args=("tests/test_dense_control_route.py::test_both_routes_give_the_controlled_matrix",),
    ),
    single_replacement_probe(
        name="whole_matrix_open_control_drop",
        module="nwqlib.subroutines.qiskit_compat",
        qualname="_controlled_whole_matrix",
        old="    open_controls = [qubit for qubit in controls if not state >> qubit & 1]\n",
        new="    open_controls = []\n",
        pytest_args=("tests/test_dense_control_route.py::test_both_routes_give_the_controlled_matrix",),
    ),
    single_replacement_probe(
        name="whole_matrix_distributed_charge_drop",
        module="nwqlib.subroutines.qiskit_compat",
        qualname="dense_control_charges",
        old="                        controlled_widths.append(item.operation.num_qubits + num_controls)\n",
        new="                        controlled_widths.append(item.operation.num_qubits)\n",
        pytest_args=(
            "tests/test_qsp_evolution.py::test_joint_generator_admits_the_syntheses_of_its_controlled_branches",
        ),
    ),
    single_replacement_probe(
        name="qls_whole_matrix_query_charge_drop",
        module="nwqlib.algorithms.qls.host_planning",
        qualname="_admit_query_synthesis",
        old="                controlled_widths += 2 * [dimension.bit_length() + controls]\n",
        new="                controlled_widths += 2 * [dimension.bit_length()]\n",
        pytest_args=(
            "tests/test_qls_quantum.py::test_controlled_dense_query_synthesis_is_admitted_at_planning",
        ),
    ),
    single_replacement_probe(
        name="qls_query_route_drop",
        module="nwqlib.algorithms.qls.quantum",
        qualname="_program",
        old="    route = (method.dense_control_route if implementation.domain == \"dense_dilation\"\n",
        new="    route = (\"gatewise\" if implementation.domain == \"dense_dilation\"\n",
        pytest_args=(
            "tests/test_dense_control_route.py::test_qls_controlled_query_applies_the_dilation_under_its_control_value_on_both_routes",
            "tests/test_qls_quantum.py::test_dense_query_syntheses_of_a_run_match_the_planning_charge",
        ),
    ),
    single_replacement_probe(
        name="lchs_whole_matrix_branch_cx_law_drop",
        module="nwqlib.algorithms.lchs.native",
        qualname="dense_branch_select_cx",
        old='        return controlled_synthesis_gate_census(num_system_qubits + address_bits)["cx"]\n',
        new='        return controlled_synthesis_gate_census(num_system_qubits + address_bits - 1)["cx"]\n',
        pytest_args=(
            "tests/test_lchs_resource_structural_law.py::test_dense_exact_select_cx_law_bounds_the_built_select",
        ),
    ),
    single_replacement_probe(
        name="lchs_whole_matrix_branch_work_drop",
        module="nwqlib.algorithms.lchs.parameters",
        qualname="construction_work",
        old="            work += physical*synthesis_work\n",
        new="            work += synthesis_work\n",
        pytest_args=(
            "tests/test_lchs_resource_structural_law.py::test_dense_exact_select_charges_the_syntheses_it_makes",
        ),
    ),
    single_replacement_probe(
        name="lchs_source_preparation_matrix_charge_drop",
        module="nwqlib.algorithms.lchs.parameters",
        qualname="construction_work",
        old="                work += 2*16*dimension**3 + physical*dimension**3\n",
        new="                work += physical*dimension**3\n",
        pytest_args=(
            "tests/test_lchs_resource_structural_law.py::test_dense_exact_select_charges_the_syntheses_it_makes",
        ),
    ),
    single_replacement_probe(
        name="lchs_source_preparation_cx_on_whole_matrix",
        module="nwqlib.algorithms.lchs.parameters",
        qualname="construction_work",
        old="    if data.source_layout is not None and not whole:\n",
        new="    if data.source_layout is not None:\n",
        pytest_args=(
            "tests/test_lchs_resource_structural_law.py::test_dense_exact_select_cx_law_bounds_a_source_layout",
        ),
    ),
    single_replacement_probe(
        name="lchs_select_route_drop",
        module="nwqlib.algorithms.lchs.native",
        qualname="construct_select",
        old="            for k in data.quadrature.k_nodes)), route=data.method.dense_control_route)\n",
        new="            for k in data.quadrature.k_nodes)))\n",
        pytest_args=(
            "tests/test_dense_control_route.py::test_lchs_dense_select_applies_each_branch_under_its_address_on_both_routes",
        ),
    ),
    single_replacement_probe(
        name="compiled_select_whole_matrix_synthesis_width_drop",
        module="nwqlib.algorithms.lchs.compiled_selection",
        qualname="compiled_select_controlled_syntheses",
        old="    return tuple(child.system_qubits + child.num_ancillas + 1 for child in (l_part, h_part)\n",
        new="    return tuple(child.system_qubits + child.num_ancillas for child in (l_part, h_part)\n",
        pytest_args=(
            "tests/test_lchs_resource_structural_law.py::test_compiled_qsp_select_charges_the_syntheses_of_its_dense_children",
        ),
    ),
    single_replacement_probe(
        name="compiled_select_whole_matrix_parity_census_drop",
        module="nwqlib.algorithms.lchs.compiled_selection",
        qualname="compiled_select_dense_controls",
        old="        unrolled = [sum(counts) for counts in zip(*(census_control_counts(controlled_synthesis_gate_census(width + 1), 1)\n",
        new="        unrolled = [sum(counts) for counts in zip(*(census_control_counts(controlled_synthesis_gate_census(width), 1)\n",
        pytest_args=(
            "tests/test_lchs_resource_structural_law.py::test_compiled_qsp_select_charges_the_syntheses_of_its_dense_children",
        ),
    ),
    single_replacement_probe(
        name="compiled_select_whole_matrix_child_cx_drop",
        module="nwqlib.algorithms.lchs.compiled_selection",
        qualname="compiled_select_cx_projection",
        old="                generator_cx += (_census_cx(_controlled_dense_child_census(child.system_qubits + child.num_ancillas),\n",
        new="                generator_cx += (0*_census_cx(_controlled_dense_child_census(child.system_qubits + child.num_ancillas),\n",
        pytest_args=(
            "tests/test_lchs_resource_structural_law.py::test_compiled_qsp_select_cx_law_bounds_the_built_select",
        ),
    ),
    single_replacement_probe(
        name="mps_two_qubit_synthesis_admission_drop",
        module="nwqlib.subroutines.state_preparation.mps_circuit",
        qualname="build_mps_circuit_state_preparation",
        old="        if syntheses * dense_synthesis_size(2)[0] > max_svd_work:\n",
        new="        if False:\n",
        pytest_args=(
            "tests/test_mps_state_preparation.py::"
            "test_mps_builder_admits_its_syntheses_and_layered_construction_first",
        ),
        requires=("scikit_tt",),
    ),
    # scikit_tt repeats its SVD sweep after every extracted gate, work that
    # the layered law charges before the construction starts.
    single_replacement_probe(
        name="mps_layered_construction_admission_drop",
        module="nwqlib.subroutines.state_preparation.mps",
        qualname="admit_layered_construction",
        old="    if work > max_svd_work:\n",
        new="    if False:\n",
        pytest_args=(
            "tests/test_mps_state_preparation.py::"
            "test_mps_builder_admits_its_syntheses_and_layered_construction_first",
        ),
        requires=("scikit_tt",),
    ),
    single_replacement_probe(
        name="adapt_supplied_reference_synthesis_admission_drop",
        module="nwqlib.algorithms.gcim.adapt",
        qualname="ADAPT.plan",
        old="        if execution == \"quantum\" and reference.preparation.implementation == \"qiskit.supplied\":\n",
        new="        if False:\n",
        pytest_args=(
            "tests/test_synthesis_admission.py::test_adapt_admits_the_controlled_synthesis_of_a_supplied_reference",
        ),
    ),
    single_replacement_probe(
        name="adapt_verification_reference_synthesis_charge_drop",
        module="nwqlib.algorithms.gcim.adapt_verification",
        qualname="_lowered_native_circuits",
        old="    if synthesis_work > max_work:\n",
        new="    if False:\n",
        pytest_args=(
            "tests/test_synthesis_admission.py::test_adapt_admits_the_controlled_synthesis_of_a_supplied_reference",
        ),
    ),
)


def _line_offsets(text: str) -> list[int]:
    # Split on "\n" only: str.splitlines also breaks on \x0b/\x0c/\x85/
    # U+2028/U+2029, which Python's tokenizer does not count as line breaks,
    # so such a character inside a string literal would desynchronize these
    # offsets from ast line numbers.
    offsets = [0]
    position = 0
    while (newline := text.find("\n", position)) != -1:
        offsets.append(newline + 1)
        position = newline + 1
    if position < len(text):
        offsets.append(len(text))
    return offsets


def _node_segment_bounds(text: str, node: ast.AST) -> tuple[int, int]:
    offsets = _line_offsets(text)
    start = offsets[node.lineno - 1] + node.col_offset
    end = offsets[node.end_lineno]
    return start, end


def _parse_module(probe_name: str, replacement: Replacement, text: str) -> ast.Module:
    try:
        return ast.parse(text)
    except SyntaxError as exc:
        raise RuntimeError(f"{probe_name}: module {replacement.module!r} does not parse") from exc


def _qualname_bounds(
    probe_name: str,
    replacement: Replacement,
    text: str,
    tree: ast.Module | None = None,
) -> tuple[int, int]:
    if replacement.qualname == MODULE_QUALNAME:
        return 0, len(text)

    if tree is None:
        tree = _parse_module(probe_name, replacement, text)

    matches: list[ast.AST] = []

    def visit(node: ast.AST, stack: tuple[str, ...]) -> None:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            stack = (*stack, node.name)
            if ".".join(stack) == replacement.qualname:
                matches.append(node)
        for child in ast.iter_child_nodes(node):
            visit(child, stack)

    visit(tree, ())
    if not matches:
        raise RuntimeError(
            f"{probe_name}: qualname {replacement.qualname!r} not found in "
            f"module {replacement.module!r}"
        )
    if len(matches) != 1:
        raise RuntimeError(
            f"{probe_name}: qualname {replacement.qualname!r} is ambiguous in "
            f"module {replacement.module!r}"
        )
    return _node_segment_bounds(text, matches[0])


def _resolve_replacement_in_text(
    probe_name: str,
    replacement: Replacement,
    path: Path,
    text: str,
    tree: ast.Module | None = None,
) -> ResolvedReplacement:
    start, end = _qualname_bounds(probe_name, replacement, text, tree)
    count = text[start:end].count(replacement.old)
    if not count:
        raise RuntimeError(
            f"{probe_name}: snippet {replacement.old!r} has zero occurrences in "
            f"{replacement.module}:{replacement.qualname}"
        )
    return ResolvedReplacement(replacement, path, start, end, count)


def _resolve_replacement(
    probe_name: str,
    replacement: Replacement,
) -> ResolvedReplacement:
    """Resolve one replacement against the file on disk (convenience form)."""

    path = replacement.path
    if not path.exists():
        raise RuntimeError(f"{probe_name}: module path does not exist: {path}")
    return _resolve_replacement_in_text(
        probe_name, replacement, path, path.read_bytes().decode("utf-8")
    )


def _mutate(text: str, resolved: ResolvedReplacement) -> str:
    replacement = resolved.replacement
    segment = text[resolved.start : resolved.end]
    mutated_segment = segment.replace(replacement.old, replacement.new)
    return text[: resolved.start] + mutated_segment + text[resolved.end :]


# Prints the arguments that fail to import and exits 1 when there is one.
_MISSING_IMPORTS = (
    "import importlib, sys\n"
    "missing = []\n"
    "for name in sys.argv[1:]:\n"
    "    try:\n"
    "        importlib.import_module(name)\n"
    "    except ImportError:\n"
    "        missing.append(name)\n"
    "print(' '.join(missing))\n"
    "raise SystemExit(1 if missing else 0)\n"
)


def _check_baseline(selected: tuple[Probe, ...], python: str) -> None:
    requires = tuple(dict.fromkeys(module for probe in selected for module in probe.requires))
    if requires:
        available = subprocess.run(
            [python, "-c", _MISSING_IMPORTS, *requires],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if available.returncode != 0:
            missing = available.stdout.split() or list(requires)
            needers = [probe.name for probe in selected if set(probe.requires) & set(missing)]
            raise RuntimeError(
                f"baseline: required imports unavailable: {', '.join(missing)}, needed by "
                f"{', '.join(needers)}. Install the extras that provide them, or select "
                f"other probes with --probe.\n{available.stderr}"
            )
    nodes = tuple(dict.fromkeys(node for probe in selected for node in probe.pytest_args))
    if not nodes:
        raise RuntimeError("baseline: no selected pytest nodes")
    failed, output = _run_pytest(nodes, python, "baseline")
    if failed:
        raise RuntimeError(f"baseline: selected tests failed on unmodified source\n{output.strip()}")
    print(f"baseline: PASSED ({len(nodes)} selected pytest nodes)")


def _run_pytest(pytest_args: tuple[str, ...], python: str, label: str) -> tuple[int, str]:
    with tempfile.TemporaryDirectory(prefix="nwqlib-pytest-") as directory:
        report_path = Path(directory) / "outcomes.xml"
        try:
            completed = subprocess.run(
                [python, "-m", "pytest", *pytest_args, "--color=no", f"--junitxml={report_path}"],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=PYTEST_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            # subprocess.run has already killed the pytest process.
            raise RuntimeError(
                f"{label}: pytest infrastructure error: no result within "
                f"{PYTEST_TIMEOUT_SECONDS} s for {pytest_args}"
            ) from exc
        strict_xpass = False
        if completed.returncode in (0, 1):
            try:
                report = ET.parse(report_path)
                failures = report.findall(".//testcase/failure")
            except (OSError, ET.ParseError) as exc:
                raise RuntimeError(
                    f"{label}: pytest infrastructure error: missing or invalid JUnit report\n"
                    f"{completed.stdout}{completed.stderr}"
                ) from exc
            # JUnit failure messages come from pytest's test outcome. Captured
            # application output and terminal formatting cannot create XPASS.
            strict_xpass = any(
                failure.get("message", "").startswith("[XPASS(strict)]") for failure in failures
            )
    output = completed.stdout + completed.stderr
    passed, failed = _pytest_counts(completed.stdout)
    # Exit 1 also covers setup/teardown errors; only actual test failures can
    # kill a mutant. Collection, usage and interpreter errors are infrastructure.
    if (
        completed.returncode not in (0, 1)
        or _count_summary(completed.stdout, "errors?")
        or report.findall(".//testcase/error")
        or failed != len(failures)
        or (completed.returncode == 1 and failed == 0)
    ):
        raise RuntimeError(
            f"{label}: pytest infrastructure error (exit {completed.returncode}) "
            f"for {pytest_args}\n{output.strip()}"
        )
    if (
        passed + failed == 0
        or len(report.findall(".//testcase")) != passed + failed
        or report.findall(".//testcase/skipped")
        or any(
            _count_summary(completed.stdout, status) for status in ("skipped", "xfailed", "xpassed")
        )
        # Pytest includes strict XPASS in its failed count even though the
        # test body passed. It cannot witness a mutation's effect.
        or strict_xpass
    ):
        raise RuntimeError(
            f"{label}: selected pytest evidence unavailable (skipped, xfail, XPASS or empty)\n"
            f"{output.strip()}"
        )
    return failed, output


def _run_probe(probe: Probe, python: str) -> str:
    """Apply one resolved source mutation, evaluate its selected tests and restore the exact
    original bytes.
    """
    replacement = probe.replacement
    path = replacement.path
    if not path.exists():
        raise RuntimeError(f"{probe.name}: module path does not exist: {path}")
    original_bytes = path.read_bytes()
    original_text = original_bytes.decode("utf-8")
    tree = (
        None
        if replacement.qualname == MODULE_QUALNAME
        else _parse_module(probe.name, replacement, original_text)
    )
    resolved = _resolve_replacement_in_text(
        probe.name,
        replacement,
        path,
        original_text,
        tree,
    )
    mutated = _mutate(original_text, resolved)
    original_sha = hashlib.sha256(original_bytes).hexdigest()
    print(f"{probe.name}: OCCURRENCES={resolved.count}")
    try:
        path.write_bytes(mutated.encode("utf-8"))
        _clear_pycache(path)
        failed, pytest_output = _run_pytest(probe.pytest_args, python, probe.name)
        killed = bool(failed)
        status = "KILLED" if killed else "SURVIVED"
        print(f"{probe.name}: {status}")
        if killed:
            failed_ids = re.findall(r"^FAILED (\S+)", pytest_output, re.MULTILINE)
            print(
                f"{probe.name}: FAILED_TEST_IDS="
                + (",".join(failed_ids) if failed_ids else "<unreported>")
            )
        return "killed" if killed else "survived"
    # Restore original bytes even when pytest or reporting fails, and verify
    # the restoration digest before another check can use this checkout.
    finally:
        path.write_bytes(original_bytes)
        _clear_pycache(path)
        restored_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{probe.name}: RESTORED_SHA256={path.relative_to(ROOT)}={original_sha}")
        if restored_sha != original_sha:
            raise RuntimeError(
                f"{probe.name}: restore SHA mismatch: {path}: "
                f"expected {original_sha}, got {restored_sha}"
            )


def _pytest_counts(output: str) -> tuple[int, int]:
    return _count_summary(output, "passed"), _count_summary(output, "failed")


_PYTEST_SUMMARY = re.compile(
    r"^=*\s*(?:\d+ [a-z]+(?:, \d+ [a-z]+)*|no tests ran) in \d+(?:\.\d+)?s(?: \(\d+:\d\d:\d\d\))?\s*=*$"
)


def _count_summary(output: str, label: str) -> int:
    # Pytest's result summary is its last line of the form "1 failed, 2 passed
    # in 0.51s". Captured output and tracebacks may themselves contain status
    # words and must not count. A native library can still write to stdout
    # after that line when the process exits, as LAPACK does for an argument
    # error, so the summary is the last line of that form, not the last line.
    summaries = [line for line in output.splitlines() if _PYTEST_SUMMARY.match(line.strip())]
    summary = summaries[-1] if summaries else ""
    return sum(int(match.group(1)) for match in re.finditer(rf"(\d+) {label}\b", summary))


def _clear_pycache(path: Path) -> None:
    pycache = path.parent / "__pycache__"
    if not pycache.exists():
        return
    stem = path.stem
    for bytecode in pycache.glob(f"{stem}.*.pyc"):
        bytecode.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--probe",
        choices=[probe.name for probe in PROBES],
        action="append",
        help="Run only the named probe. May be repeated.",
    )
    args = parser.parse_args()
    selected = PROBES
    if args.probe:
        names = set(args.probe)
        selected = tuple(probe for probe in PROBES if probe.name in names)

    python = sys.executable
    for probe in selected:
        _resolve_replacement(probe.name, probe.replacement)
    _check_baseline(selected, python)
    results = {probe.name: _run_probe(probe, python) for probe in selected}
    survived = [name for name, status in results.items() if status == "survived"]
    if survived:
        print("SURVIVED probes: " + ", ".join(survived), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
