"""LCHS PREP, planned evolution and inverse PREP in the shared Program."""

from nwqlib._limits import ResourceLimitError, exceeds
from typing import NamedTuple

import numpy as np

from nwqlib._preparation_laws import direct_preparation_cx_bound, direct_preparation_controlled_cx_bound
from nwqlib._quantum_readout import qwc_groups
from nwqlib.amplitudes import AmplitudeReadout
from nwqlib.artifacts import ArrayOutput
from nwqlib.blocks.records import BlockSemantics, SelectedConstruction, SelectedDefinition
from nwqlib.blocks.selection import SelectedBlock, _signature, select_pauli_parity, select_preparation, transform_block
from nwqlib.core.planning import Plan, Experiment, ObservationSpec, ReadoutDetails
from nwqlib.core.records import Basis, InputRef
from nwqlib.evidence.records import Evidence
from nwqlib.ir import Allocate, Binding, BlockCall, ClassicalValue, Definition, Measure, MeasurementBatch, MetadataRef, PortMap, Program, Register, Release, Sequence, Setting
from nwqlib.operators.access import _check_bytes
from nwqlib.operators.inputs import _digest
from nwqlib.problems.inputs import StatePreparationSpec, compose_recovery
from nwqlib.problems.records import Solution, StateVector, Samples, NormalizedExpectation, NormSquared, QuadraticForm
from nwqlib.resources.records import ResourceLaw, Workspace
from .compiled_selection import _bytes_left
from .primary_records import ASSERTED_L_BOUNDS, METHOD, LCHSFixedShotPrecision, LCHSReconstruction
from .solution_error_budget import psd_recovery_scale


def target_identity(target):
    """Return the SHA-256 content hash of the normalized PREP tensor, its dtype, shape and C-order bytes.

    Planning records it as the PREP input and derives preparation_identity
    from it.
    """
    return _digest("mps.normalized.C-order", (len(target),), (target,))


def preparation_identity(identity,decomposition,layers):
    """Return the hash of the tensor's content hash (target_identity), the chosen cores and the layered synthesis settings."""
    settings = None if decomposition is None else (
        decomposition.num_qubits,decomposition.original_dimension,decomposition.bond_dimensions,
        decomposition.max_bond_dim,decomposition.threshold,decomposition.discarded_weight,
        decomposition.input_id,decomposition.input_norm)
    return _digest('lchs.selected_preparation',(identity,layers,settings),
        () if decomposition is None else decomposition.cores)


def select_vector_preparation(name, target, *, method, decomposition=None, layers=2):
    """Record the normalized tensor and the optional cores already chosen."""
    from .native import construct_preparation
    q = len(target).bit_length()-1
    if decomposition is None:
        identity = target_identity(target)
    else:
        from nwqlib.subroutines.state_preparation.mps import _selected_target_id
        identity = _selected_target_id(target, decomposition)
    signature = _signature(name, "lchs.preparation", tuple((f"system_{j}", 1) for j in range(q)))
    input_ref = InputRef(identity=identity,representation="normalized_vector",source=METHOD)
    basis = Basis(identity="lchs-prep",dimension=len(target),ordering="little-endian qubits")
    work, known_bytes = max(1,q)*len(target), 16*len(target)
    if decomposition is not None:
        # The bytes that the builder checks when the block is lowered
        # (native.construct_preparation). The work of the whole construction
        # was compared with max_svd_work before the TT-SVD
        # (parameters._admit_mps_circuit), so lowering cannot refuse a
        # planned construction.
        from nwqlib.subroutines.state_preparation.mps import admit_layered_construction
        work, known_bytes = admit_layered_construction(decomposition.bond_dimensions, layers,
            max_svd_work=None, max_bytes=method.max_bytes)
    preparation = StatePreparationSpec(input=input_ref,basis=basis,physical_scale=compose_recovery(1.),
        required_ancillas=0,inverse_available=True,control_available=True,
        implementation="qiskit.direct" if decomposition is None else "qiskit.mps_circuit",
        approximation=("ideal direct preparation of the selected normalized tensor" if decomposition is None
            else "selected layered MPS approximation; circuit fidelity not evaluated"),
        work_law="selected direct or layered MPS construction size",items=len(target),payload_bytes=known_bytes,work=work)
    # Each MPS layer has at most max(0, q-1) two-qubit unitaries,
    # each using at most three CX before routing.
    base = direct_preparation_cx_bound(q, complex_phases=bool(np.any(target.imag))) if decomposition is None else 3*layers*max(0,q-1)
    laws = [ResourceLaw(metric="cx", basis="cx", value=base,
        interpretation="upper_bound" if decomposition is None else "estimate",
        evidence=Evidence(kind="proved_relation" if decomposition is None else "numerical_estimate", source=METHOD),
        adjoint=adjoint, assumptions=("selected PREP recipe before routing/optimization",)) for adjoint in (False, True)]
    if decomposition is None:
        laws.extend(ResourceLaw(metric="cx", basis="cx", value=direct_preparation_controlled_cx_bound(q),
            interpretation="upper_bound", evidence=Evidence(kind="proved_relation", source=METHOD),
            controlled=True, adjoint=adjoint, assumptions=("one extra control on the selected direct PREP extension",))
            for adjoint in (False, True))
    record = SelectedDefinition(signature=signature, implementation=signature.target,
        semantics=BlockSemantics(kind="preparation", input=input_ref,
            basis=basis,preparation=preparation,
            relation="U|0> is the selected normalized tensor, or its selected layered MPS approximation",
            input_projector="all zero", output_projector="prepared tensor", success="deterministic PREP from zero input",
            workspace=0, restoration="no additional workspace", epsilon=0. if decomposition is None else None,
            approximation_metric="state L2 on zero input", approximation_evidence="direct ideal circuit relation; MPS circuit fidelity requires explicit validation",
            inverse_legal=True, control_legal=True, phase="the chosen full extension preserves physical phase under inverse/control"),
        choice="direct" if decomposition is None else "mps_circuit", decomposition=None, cost_law=None,
        coefficient_selection_id=preparation_identity(identity,decomposition,layers),
        construction_work=work, resource_laws=tuple(laws),
        cost_parameters=() if decomposition is None else (
            Binding(parameter="mps_layers",value=layers),
            Binding(parameter="mps_max_stored_bond",value=max(decomposition.bond_dimensions)),
            Binding(parameter="mps_core_bytes",value=decomposition.core_bytes)),
        cost_context="actual coefficient/input tensor and selected PREP, not a full-unitary error certificate",
        workspace=(Workspace(location="host", purpose="preparation", bytes=known_bytes, source=METHOD),))
    return SelectedBlock.bind(record, payload=(target, decomposition, layers, method.max_bytes, method.max_svd_work), constructor=construct_preparation)


def observable_terms(problem, output, *, max_bytes):
    """Return the observable's real Pauli terms in the encoded coordinates.

    Exact readout contracts the saved real Pauli table on the physical
    slice and sampled readout measures its qubit-wise commuting groups, so
    the observable must be a real Pauli sum. Pauli input is used as supplied. A dense
    observable must be exactly Hermitian, so that no symmetrization changes
    the requested quantity. Its zero-padded matrix is transformed once
    (operators._pauli.pauli_coefficient_masks), and labels are formed only
    for its nonzero words, in the order of ``pauli_coefficients``. The
    arrays of the padding and transform, the packing and the labels are
    each checked against ``max_bytes`` before that stage allocates them.
    Sampled readout groups the terms under ``LCHS.max_readout_work``.
    """
    if not isinstance(output, (NormalizedExpectation, QuadraticForm)):
        return ()
    observable = output.observable
    if observable.manifest.basis != problem.basis:
        raise ValueError("observable must use the original physical coordinates")
    if observable.reference.representation == "pauli":
        values = observable.pauli_terms().labels(max_bytes=max_bytes)
    elif observable.reference.representation == "dense":
        from nwqlib.operators._pauli import PauliTerms, _label_bytes, pauli_coefficient_masks
        n = problem.dimension
        d = 1 << max(1,(n-1).bit_length())
        q = d.bit_length()-1
        # The bytes alive at each stage beside the stored observable, checked
        # before the stage allocates. The padded copy, the two active
        # complex128 levels of the block transform and their half-sum
        # temporaries take 64 d**2, which depends on the dimension alone and
        # is checked before the observable is read. pauli_coefficient_masks
        # then checks the matrix, the coefficient vector and the masks of the
        # R nonzero words.
        _check_bytes(64*d*d, max_bytes, "dense observable Pauli transform")
        matrix = observable.dense_array()
        # Row by row, each test reads one row and one conjugated column of at
        # most n entries, so no conjugate of the whole observable is formed.
        if not all(np.array_equal(matrix[row, row:], matrix[row:, row].conj()) for row in range(n)):
            raise ValueError("observable must be exactly Hermitian; explicitly symmetrize if that changes the intended problem")
        if d != n:
            matrix = np.pad(matrix, ((0, d-n), (0, d-n)))
        x, z, coefficients = pauli_coefficient_masks(matrix, max_bytes=max_bytes)
        del matrix
        # The table holds those fresh arrays without a copy. They stay alive
        # beside the label tiles and the returned labels.
        table = PauliTerms._from_saved(q, x, z, coefficients)
        held = x.nbytes + z.nbytes + coefficients.nbytes
        del x, z, coefficients
        if len(table):
            _check_bytes(held + _label_bytes(len(table), q), max_bytes, "dense observable Pauli labels")
        values = table.labels(max_bytes=_bytes_left(max_bytes, held))
    else:
        raise ValueError("LCHS quantum observable needs finite real Pauli access or an explicit dense matrix")
    if any(value.imag != 0 for _,value in values):
        raise ValueError("Pauli observable coefficients must be real")
    return tuple((label,float(value.real)) for label,value in values)


def _readout_kind(problem, data, output, shots):
    """Return the coherent ranking model and readout of the output (``coherent_selection.CoherentObjective``).

    Accepted states are drawn by postselection, so their model is
    ``"rejection"``; exact readout and fixed-shot moments keep every shot and
    compare the cost of one circuit (``"single_circuit"``). Counts of an
    unpadded normalized expectation are success-conditional; the other
    moments, and a padded normalized expectation, use every returned shot.
    """
    if isinstance(output, Samples):
        return "rejection", "samples"
    if shots is None or isinstance(output, (Solution, StateVector)):
        return "single_circuit", "exact"
    padded = problem.dimension != data.generator.encoded_dimension
    conditional = isinstance(output, NormalizedExpectation) and not padded
    return "single_circuit", "conditional_counts" if conditional else "unconditional_counts"


def sampling_readout(problem, output, accuracy, *, num_qubits, method):
    """Return the ``coherent_selection.SamplingTarget`` of an accuracy request and the readout grouping it uses.

    A quadratic form is measured in the qubit-wise commuting groups of
    `lchs_readout` on the ``num_qubits`` system qubits, and each group's
    score range ``A_g`` is ``analysis.score_ranges`` of those groups; the
    norm squared has its one physical-mass setting and no grouping. The
    grouping comparisons are checked against ``max_readout_work`` as in
    planning. The grouping reads only the observable terms, the width and
    the output, not the circuit's other registers, so the coherent Plan of
    this pass emits it unchanged (``finish_quantum_plan``).

    Returns:
        ``(target, readout)``: the SamplingTarget, and ``(terms, settings,
        groups, comparisons)`` of a quadratic form, else None.

    Raises:
        ApplicabilityError: For a quadratic form on more than 64 system
            qubits. Every route measures at least those qubits, and its counts
            analysis decodes one 64-bit index per outcome, as
            ``finish_quantum_plan`` refuses for shots.
    """
    from nwqlib.algorithms.protocol import ApplicabilityError
    from .analysis import score_ranges
    from .coherent_selection import SamplingTarget
    common = dict(tolerance=float(accuracy.absolute_tolerance), confidence=float(accuracy.confidence))
    if isinstance(output, NormSquared):
        return SamplingTarget(output="norm_squared", ranges=(), **common), None
    if num_qubits > 64:
        raise ApplicabilityError(f"sampled LCHS readout of {type(output).__name__} needs at least {num_qubits} "
                                 "classical bits, more than its 64-bit count decoder supports")
    terms = observable_terms(problem, output, max_bytes=method.max_bytes)
    settings, groups, comparisons = lchs_readout(output, terms, tuple(range(num_qubits)), dimension=problem.dimension,
                                                 shots=1, method=method)
    return (SamplingTarget(output="quadratic_form", ranges=score_ranges(terms, groups), **common),
            (terms, settings, groups, comparisons))


class CoherentChoice(NamedTuple):
    """The coherent choice of one planning pass: the winning grid with its selection, facts and pass context.

    ``choose_provider`` returns it; the accuracy hook reads the selected
    row's count and ``plan_coherent_lcu`` prepares that row once.

    Attributes:
        data: The chosen grid's ``LCHSData``, with ``selected_select`` and
            ``provider_selection`` set.
        raw: Its quadrature summary (``lchs_quadrature_summary``).
        selection: Its ``CoherentSelection``.
        facts: The application facts of each alternative of that selection.
        context: The pass's ``parameters.CoherentContext``.
        readout: The accuracy grouping ``(terms, settings, groups,
            comparisons)`` of `sampling_readout`, or None.
        smallest_recovery: Under fixed shots, ``(Gamma, label)`` of the
            admitted row of smallest recovery over the grids of the pass that
            a Method can name, with its grid in the label when the pass
            compares several (`coherent_selection.smallest_recovery_row`),
            otherwise None.
    """

    data: object
    raw: object
    selection: object
    facts: object
    context: object
    readout: object = None
    smallest_recovery: object = None


def _candidate(method, problem, common, entry, *, context, enumeration, objective):
    """Build one provider entry's grid and phase-free selection; return its record and ``(key, choice)`` or None.

    ``objective`` is ``(acquisition, readout, sampling, grouping)``. A
    concrete pair (no enumeration) lets every refusal propagate. Under
    enumeration a provider refusal (ValueError of the grid) or a selection
    without an admitted row (ApplicabilityError) becomes the entry's record.
    A limit refusal (ResourceLimitError) always propagates, so a limit never
    removes an entry from the enumeration.
    """
    import math
    from nwqlib.algorithms.protocol import ApplicabilityError
    from .parameters import coherent_selection
    from .selection import with_providers
    from .time_independent_terms import lchs_quadrature_summary
    acquisition, readout, sampling, grouping = objective
    if entry.skip is not None:
        return _provider_record(entry, None, status=entry.skip[0], reason=entry.skip[1]), None
    ledger = {}
    try:
        data = with_providers(common, problem, entry, work_ledger=ledger)
    except ResourceLimitError:
        raise
    except ValueError as error:
        if not enumeration:
            raise
        return _provider_record(entry, None, status="refused", reason=f"provider grid: {error}"), None
    raw = lchs_quadrature_summary(data.quadrature, method=method, he_backend=data.backend,
                                  final_time=problem.elapsed_time)
    try:
        selection, facts = coherent_selection(method, problem, data, raw=raw, context=context, acquisition=acquisition,
                                              readout=readout, sampling=sampling)
    except ApplicabilityError as error:
        if not enumeration:
            raise
        return _provider_record(entry, data, status="refused", reason=str(error), ledger=ledger), None
    row = selection.selected
    value = row.cost.get("objective")
    key = None
    if value is not None:
        key = math.log2(value) if value > 0 else -math.inf
        if acquisition == "rejection":
            key += 2.0 * row.cost["gamma_log2"]
    record = _provider_record(entry, data, status="admitted", reason=selection.reason, ledger=ledger,
                              selection=selection, key=key)
    return record, (key, CoherentChoice(data, raw, selection, facts, context, grouping))


def _provider_record(entry, data, *, status, reason, ledger=None, selection=None, key=None):
    """Return the read-only ``LCHSData.provider_selection`` record of one frontier entry.

    An entry whose grid was built (``data`` not None) copies its scalars:
    ``range_k``, ``interval_count_each_side``, ``node_count`` (Q per panel
    for ``composite_gauss``, the total for other rules), ``total_node_count``
    (physical nodes without LCU padding), ``h1``, ``coefficient_l1_norm``
    (the uncompensated finite kernel-quadrature mass) and the unit-input
    component bounds ``approximate_lchs_error_bound`` and
    ``quadrature_error_bound`` before PSD growth (None when unavailable).
    Each is None for an entry without a grid.
    """
    from types import MappingProxyType
    parameters = dict(lchs_kernel=entry.lchs_kernel.implementation,
                      kernel_parameters=MappingProxyType(dict(entry.lchs_kernel.parameters)),
                      k_quadrature=entry.k_quadrature.implementation,
                      quadrature_parameters=MappingProxyType(dict(entry.k_quadrature.parameters)),
                      seams=MappingProxyType(dict(strip_margin=entry.strip_margin,
                                                  free_half_width=entry.free_half_width)))
    if data is not None:
        plan = data.quadrature.coefficient_plan
        parameters.update(kernel_parameters=plan.resolved_lchs_kernel.parameters,
                          quadrature_parameters=plan.resolved_k_quadrature.parameters,
                          derived=plan.derived_parameters)
    row = None if selection is None else selection.selected
    grid = None if data is None else data.quadrature
    bounds = None if grid is None else grid.coefficient_plan.quadrature
    scalars = {name: None if grid is None else getattr(grid, name) for name in (
        "range_k", "interval_count_each_side", "node_count", "total_node_count", "h1", "coefficient_l1_norm")}
    scalars.update({name: None if bounds is None else getattr(bounds, name)
                    for name in ("approximate_lchs_error_bound", "quadrature_error_bound")})
    return MappingProxyType(dict(
        label=entry.label, parameters=MappingProxyType(parameters), status=status, reason=reason, **scalars,
        row=None if row is None else row.label, key=key,
        quadrature_work=None if ledger is None else ledger.get("quadrature_work"),
        proposal=entry.proposal))


def choose_provider(method, problem, common, *, output, shots, accuracy=None):
    """Choose the provider grid and coherent row of one planning pass, streaming the candidates, without any preparation.

    ``common`` is the pass's ``selection.LCHSCommon``. The pass context
    (``parameters.coherent_context``) and, for an accuracy request, the
    readout grouping are built once. Each entry of
    ``selection.provider_frontier`` then builds its grid
    (``selection.with_providers``) and runs one phase-free selection
    (``parameters.coherent_selection``) with this output's acquisition, so
    every candidate is compared with its actual finite coefficient mass,
    not the continuous kernel beta. A concrete provider pair is the
    one-entry frontier of the same path, and its refusals propagate.

    Within one grid the selection drops the common factor gamma (application
    masses times the kernel one-norm); across grids the comparison restores
    it, as log2 values: ``C`` for single-circuit acquisition (exact readout,
    or the fixed-acquisition comparison of fixed shots),
    ``log2(zeta**2 C) + 2 log2 gamma`` for rejection, the ``(zeta gamma)**2 C``
    of KK7, and the sufficient work ``N G C`` (AH3, already from the row's
    own recovery zeta gamma) for an accuracy request. A candidate without an
    admitted row, or with an unknown objective, ranks after every known
    one, and ties keep the frontier order, so the default pair wins ties.

    Only the incumbent winner and the current candidate are alive: a losing
    candidate's grid, coefficient plan and selection are released before
    the next grid is built, and only its scalar record and, under fixed
    shots, the smallest recovery of its rows are kept. An
    enumeration pass therefore builds at most 12 grids and runs at most 12
    phase-free selections, each within the Method's per-construction and
    per-row limits, which bound one candidate and not their sum.

    Returns:
        The `CoherentChoice` of the winner, whose ``data`` carries
        ``provider_selection`` (one record per visited entry under
        enumeration, empty for a concrete pair). Nothing is prepared.

    Raises:
        ApplicabilityError: If no enumerated candidate has an admitted row.
    """
    from dataclasses import replace
    from types import MappingProxyType
    from nwqlib.algorithms.protocol import ApplicabilityError
    from .coherent_selection import smallest_recovery_row
    from .native import resolve_lcu_select_implementation
    from .parameters import coherent_context
    from .selection import provider_frontier
    selected_select = resolve_lcu_select_implementation(method, backend=common.backend)
    grouping = sampling = None
    if accuracy is None:
        acquisition, readout = _readout_kind(problem, common, output, shots)
    else:
        acquisition, readout = "accuracy", "unconditional_counts"
        sampling, grouping = sampling_readout(problem, output, accuracy, num_qubits=common.generator.num_qubits,
                                              method=method)
    context = coherent_context(method, problem, common)
    enumeration = method.lchs_kernel.implementation == "auto"
    records, incumbent, winner, smallest = [], None, None, None
    for entry in provider_frontier(method, problem, common, acquisition=acquisition, context=context,
                                   records=records):
        record, current = _candidate(method, problem, common, entry, context=context, enumeration=enumeration,
                                     objective=(acquisition, readout, sampling, grouping))
        records.append(record)
        # A Method cannot name a private seam of a grid, so a route on such a grid is left out of what the record offers.
        if (current is not None and acquisition == "single_circuit" and readout != "exact"
                and entry.free_half_width is None and entry.strip_margin is None):
            smallest = smallest_recovery_row(current[1].selection, smallest,
                                             grid=entry.label if enumeration and common.l_norm != 0 else None)
        if current is not None and (incumbent is None or current[0] is not None and (
                incumbent[0] is None or current[0] < incumbent[0])):
            incumbent, winner = current, len(records) - 1
        # Release a losing candidate before the next grid is built.
        current = None
    if incumbent is None:
        raise ApplicabilityError("no provider candidate of lchs_kernel='auto' has an admitted coherent row: "
                                 + "; ".join(f"{item['label']}: {item['reason']}" for item in records))
    if enumeration:
        records[winner] = MappingProxyType({**records[winner], "status": "selected"})
    choice = incumbent[1]
    return choice._replace(smallest_recovery=smallest, data=replace(choice.data, selected_select=selected_select,
                                                                    provider_selection=tuple(records) if enumeration else ()))


def _fixed_shot_precision(choice, recovery):
    """Return the ``LCHSFixedShotPrecision`` of a fixed-shot choice with recovery ``recovery``, or None.

    None when the choice is not under fixed shots or no compared row has a
    smaller recovery. The ranking stays as it is, so this only records the
    consequence. A selected row of unknown cost has the fewest original-U calls
    within its alternative on its grid (`coherent_selection._choose`). Between
    alternatives and between grids no cost remains to compare, so the first in
    their fixed order is taken (`coherent_selection._choose_alternative`,
    `choose_provider`), and the record says which of the two cases applies.
    """
    from .coherent_selection import _squared_ratio, row_on_grid
    selection = choice.selection
    if choice.smallest_recovery is None:
        return None
    smallest, label = choice.smallest_recovery
    selected = selection.selected
    grids = choice.data.provider_selection
    selected_row = row_on_grid(selected.label, next(item["label"] for item in grids if item["status"] == "selected")
                               if len(grids) > 1 else None)
    if label == selected_row or (recovery.exponent, recovery.mantissa) <= (smallest.exponent, smallest.mantissa):
        return None
    alternatives = {row.alternative for row in selection.candidates if row.status == "admitted" and row.requested}
    ranking = ("cx_per_circuit" if selected.cost.get("objective") is not None else
               "original_u_calls_and_order" if len(alternatives) > 1 or len(grids) > 1 else "original_u_calls")
    return LCHSFixedShotPrecision(readout=selection.objective.readout, ranking=ranking, selected_row=selected_row,
                                  smallest_row=label, smallest_recovery=smallest, factor=_squared_ratio(recovery, smallest))


def plan_quantum(method, problem, data, *, output, shots, rng):
    """Select the phase-weighted LCU body and recovery for the original physical dynamics.

    The success block of PREP, SELECT and inverse PREP is sum_j c_j U_j/alpha
    with alpha = sum_j |c_j| (ACL arXiv:2312.03916v2, Lemma 24, Eq. (178)),
    acting on the unit initial direction. The physical solution is therefore
    alpha times ||u0|| times the success-projected system amplitudes. With a
    constant source the input norms are already inside the branch
    coefficients, so ||u0|| is not applied again. Qubits are ordered as
    coefficient address, then system, and every non-system qubit must read
    zero for success.

    This plans the dense_exact and product-formula backends. The
    qsp_block_encoding backend plans the coherent LCU body instead
    (`choose_provider`, `plan_coherent_lcu`).
    """
    from .parameters import select_parameters, selected_error_facts, selected_identity, construction_work
    from .time_independent_terms import lchs_quadrature_summary
    from .native import construct_select
    raw = lchs_quadrature_summary(data.quadrature,method=data.method,
        he_backend=data.backend,final_time=problem.elapsed_time)
    data, extra = select_parameters(method, problem, data)
    layout = data.source_layout
    slots = len(data.coefficient_plan.coefficients) if layout is None else len(layout.phases)
    a = (slots-1).bit_length()
    q = data.generator.num_qubits
    width = a+q
    # Recover the coefficient normalization once. Source
    # branches already carry their own input amplitudes and scale. The
    # homogeneous coefficients hold exp(s*T) for the signed shift s divided
    # by 2**recovery_exponent, and a source layout holds its magnitudes
    # relative to 2**scale_exponent, so the recovery carries that 2**e.
    alpha, exponent = ((data.coefficient_plan.coefficient_l1_norm, data.coefficient_plan.recovery_exponent)
                       if layout is None else (layout.coefficient_l1_norm, layout.scale_exponent))
    growth = psd_recovery_scale(exponent)
    recovery = compose_recovery(alpha,
        *(() if data.source_state is not None else (problem.initial_state.preparation.physical_scale,)),
        *(() if growth is None else (growth,)))
    coordinates, success = tuple(range(a,width)), tuple(range(a))
    terms = observable_terms(problem, output, max_bytes=method.max_bytes)
    evidence = data.quadrature.conversion["psd_premise_evidence"]
    rec = LCHSReconstruction(elapsed_time=problem.elapsed_time, dimension=problem.dimension, encoded_dimension=1<<q,
        mode='unitary' if data.quadrature.l_norm==0 else 'quadrature',
        has_source=data.source_state is not None, initial_scale=problem.initial_state.preparation.physical_scale,
        source_scale=None if problem.source is None else problem.source.preparation.physical_scale,
        recovery=recovery, success_bits=success, system_bits=coordinates, terms=terms,
        coefficient_l1_norm=alpha, physical_branches=extra["physical_branches"], padded_branches=1<<a,
        selected_backend=data.backend, selected_select=data.selected_select,
        psd_premise=evidence if data.quadrature.psd_premise_satisfied else "unavailable",
        psd_shift=data.quadrature.conversion["psd_shift"], scalar_decay=data.quadrature.conversion["scalar_decay"],
        l_norm=data.quadrature.l_norm,
        kernel_approximation_bound=data.quadrature.coefficient_plan.quadrature.approximate_lchs_error_bound,
        quadrature_bound=data.quadrature.coefficient_plan.quadrature.quadrature_error_bound,
        source_nodes=extra.get("source_nodes",()), source_weights=extra.get("source_weights",()), step_counts=extra.get("step_counts",()))
    identity = selected_identity(data, problem, method)
    work, counts = construction_work(data, extra, method=method, elapsed=problem.elapsed_time)
    # The outer QSP gate's classification and the product-formula
    # decomposition and selection draw on the same ledger.
    select_work = _checked_select_work(work, method, charged=extra.get("select_work_charged", 0))
    signature = _signature("lchs_select", "lchs.select", tuple((f"bit_{j}",1) for j in range(width)))
    select_record = SelectedDefinition(signature=signature, implementation=signature.target,
        semantics=BlockSemantics(kind="unknown", input=InputRef(identity=identity, representation="selected_lchs_parameters", source=METHOD),
            basis=problem.basis, relation="selected phase-weighted evolution branches; source branches include their own input PREP",
            input_projector="SELECT ancillas zero; arbitrary coefficient address", output_projector="SELECT ancillas zero",
            success="algorithm SELECT block; LCU success also requires inverse coefficient PREP",
            workspace=0, restoration="selected source and SELECT workspace obligations; coefficient address is preserved",
            epsilon=None, approximation_metric="selected evolution operator norm", approximation_evidence="component laws with their stated premises",
            inverse_legal=True, control_legal=True, phase="complex coefficient and identity evolution phases kept"),
        choice=data.selected_select, coefficient_selection_id=identity, decomposition=None, cost_law=None,
        construction_work=work, cost_context="actual SELECT recipe; no coefficient PREP pair in this leaf's cost",
        # A supplied source circuit has an unknown controlled cost, so its
        # SELECT declares no CX law rather than an incomplete count.
        resource_laws=() if counts["select_cx"] is None else tuple(ResourceLaw(metric="cx", basis="cx",
            value=counts["select_cx"], interpretation="estimate", evidence=Evidence(kind="numerical_estimate", source=METHOD),
            adjoint=adjoint, assumptions=("selected unoptimized SELECT projection; routing/native synthesis may differ",))
            for adjoint in (False,True)),
        workspace=(Workspace(location="host", purpose="workspace", bytes=counts["known_peak_bytes"], source=METHOD),
                   Workspace(location="host", purpose="workspace", bytes=None, source=METHOD)))
    # Homogeneous evolution prepares u0 outside SELECT. Source evolution
    # includes each branch's own input preparation inside the selected action.
    blocks = []
    definitions, calls = [], []
    if data.source_state is None:
        if data.dense is not None:
            # A direct PREP takes the default layer count of select_vector_preparation, which its identity hashes.
            mps = {} if data.initial_mps is None else dict(decomposition=data.initial_mps,
                                                           layers=method.init_prep.num_layers)
            initial = select_vector_preparation("initial_prep",data.dense.initial_direction,method=method,**mps)
        else:
            # A compact generator uses the state's own construction (product,
            # occupation, vector or supplied circuit) and its cost law, as the
            # periodic route does, without forming a padded direction.
            spec = problem.initial_state.preparation
            _check_bytes(spec.payload_bytes,method.max_bytes,'LCHS input PREP')
            if exceeds(spec.work, method.max_select_work):
                raise ResourceLimitError(f'LCHS input PREP exceeds max_select_work: it needs {spec.work} work units, '
                                         f'max_select_work={method.max_select_work}')
            initial = select_preparation('initial_prep',problem.initial_state,per_bit=True)
        blocks.append(initial)
        definitions.append(Definition(id="initial_prep",node=BlockCall(signature="initial_prep",
            ports=tuple(PortMap(port=f"system_{j}",wire=f"bit_{bit}") for j,bit in enumerate(coordinates)))))
        calls.append("initial_prep")
    # PREP and inverse PREP surround SELECT on the coefficient address only.
    address = tuple(range(a))
    if a:
        mps = {} if data.coefficient_mps is None else dict(decomposition=data.coefficient_mps,
                                                           layers=method.lcu_prep.num_layers)
        prep = select_vector_preparation("coefficient_prep",extra["amplitudes"],method=method,**mps)
        inverse = transform_block("coefficient_inverse",prep,adjoint=True)
        blocks.extend((prep,inverse))
        for name in ("coefficient_prep","coefficient_inverse"):
            definitions.append(Definition(id=name,node=BlockCall(signature=name,
                ports=tuple(PortMap(port=f"system_{j}",wire=f"bit_{bit}") for j,bit in enumerate(address)))))
        calls.append("coefficient_prep")
    blocks.append(SelectedBlock.bind(select_record,payload=(data,problem.elapsed_time),constructor=construct_select))
    definitions.append(Definition(id="select",node=BlockCall(signature=signature.name,
        ports=tuple(PortMap(port=f"bit_{j}",wire=f"bit_{j}") for j in range(width)))))
    calls.append("select")
    if a:
        calls.append("coefficient_inverse")
    facts = selected_error_facts(data,problem=problem,raw=raw)
    return finish_quantum_plan(method,problem,rec,output=output,shots=shots,rng=rng,
        blocks=blocks,definitions=definitions,calls=calls,facts=facts,select_work=select_work,native_data=data,
        coefficient_plan=data.coefficient_plan if data.source_state is None else None)


def plan_coherent_lcu(method, problem, choice, *, output, shots, rng, accuracy=None):
    """Prepare the chosen coherent row once and plan the one coherent LCU block of the qsp_block_encoding backend and its readout.

    ``choice`` is the `CoherentChoice` of this planning pass
    (`choose_provider`); `parameters._select_coherent` checks an accuracy
    request's ``shots`` against its selected row and prepares that row,
    without selecting again. An accuracy pass's readout grouping is emitted
    unchanged (``finish_quantum_plan``).

    The block is ``native.construct_coherent_lcu``: coefficient PREP, the
    input preparation R, the selected evolution, the slot phases and the
    inverse PREP, so no separate input or coefficient PREP block exists on
    this route. Qubits are ordered as evolution ancillas, the analytic kernel
    PREP work register W (empty for any other PREP), kernel register,
    application register, then system, and every non-system qubit must read
    zero for success. The success block maps ``|0>`` to
    ``u_fin / (zeta gamma)``, so the physical recovery is gamma (the
    factorized PREP's application masses times the kernel one-norm, which
    already include every input norm and exp(s t_a)) times the route's zeta.
    """
    from math import fsum
    from nwqlib.problems.inputs import PhysicalScale
    from .parameters import _select_coherent, selected_error_facts, selected_identity, construction_work
    from .native import construct_coherent_lcu
    data, extra = _select_coherent(method, problem, choice, shots=shots)
    raw = choice.raw
    terms = observable_terms(problem, output, max_bytes=method.max_bytes) if choice.readout is None else choice.readout[0]
    prepared = data.qsp_plan
    preparation = prepared.preparation
    address = preparation.kernel_qubits+preparation.application_qubits
    ancillas = extra["evolution_ancillas"]+extra["preparation_work_qubits"]
    q = data.generator.num_qubits
    width = ancillas+address+q
    recovery = compose_recovery(PhysicalScale(mantissa=prepared.physical_recovery[0],
                                              exponent=prepared.physical_recovery[1]), prepared.recovery)
    coordinates, success = tuple(range(ancillas+address,width)), tuple(range(ancillas+address))
    evidence = data.quadrature.conversion["psd_premise_evidence"]
    # The imaginary-time filter alternative has one kernel node k = 0 with coefficient 1 and no kernel stages.
    kernel = prepared.candidate.alternative == "lchs"
    rule = data.quadrature.coefficient_plan.quadrature
    rec = LCHSReconstruction(elapsed_time=problem.elapsed_time, dimension=problem.dimension, encoded_dimension=1<<q,
        mode='unitary' if data.quadrature.l_norm==0 else 'quadrature',
        has_source=data.source_state is not None, initial_scale=problem.initial_state.preparation.physical_scale,
        source_scale=None if problem.source is None else problem.source.preparation.physical_scale,
        recovery=recovery, success_bits=success, system_bits=coordinates, terms=terms,
        # The kernel factor's one-norm; gamma in recovery carries the application masses.
        coefficient_l1_norm=fsum(abs(complex(c)) for c in data.quadrature.coefficients) if kernel else 1.,
        physical_branches=extra["physical_branches"], padded_branches=1<<address,
        selected_backend=data.backend, selected_select=data.selected_select,
        psd_premise=evidence if data.quadrature.psd_premise_satisfied else "unavailable",
        psd_shift=data.quadrature.conversion["psd_shift"], scalar_decay=data.quadrature.conversion["scalar_decay"],
        l_norm=data.quadrature.l_norm,
        kernel_approximation_bound=rule.approximate_lchs_error_bound if kernel else 0.,
        quadrature_bound=rule.quadrature_error_bound if kernel else 0.,
        source_nodes=extra["source_nodes"], source_weights=extra["source_weights"],
        fixed_shot_precision=_fixed_shot_precision(choice, recovery))
    identity = selected_identity(data, problem, method)
    work, counts = construction_work(data, extra, method=method, elapsed=problem.elapsed_time)
    # The child encodings' planning drew on the same max_select_work ledger.
    select_work = _checked_select_work(work, method, charged=extra["select_work_charged"],
                                       unresolved=counts["unresolved"])
    signature = _signature("lchs_coherent_lcu", "lchs.coherent_lcu", tuple((f"bit_{j}",1) for j in range(width)))
    row = prepared.candidate
    record = SelectedDefinition(signature=signature, implementation=signature.target,
        semantics=BlockSemantics(kind="unknown", input=InputRef(identity=identity, representation="selected_lchs_parameters", source=METHOD),
            basis=problem.basis, relation="factorized coefficient PREP, input PREP, selected coherent evolution, slot phases "
                                           "and inverse coefficient PREP; the success block maps |0> to u_fin/(zeta gamma)",
            input_projector="all zero", output_projector="evolution ancillas, PREP work, kernel and application registers zero",
            success="algorithm success projector of the coherent LCU", workspace=0,
            restoration="input preparation is not undone; it is the output register",
            epsilon=None, approximation_metric="recovered evolution operator norm",
            approximation_evidence=f"selected row {row.label}: {prepared.selection.reason}",
            inverse_legal=True, control_legal=True, phase="coefficient, scalar and input phases kept"),
        choice=data.selected_select, coefficient_selection_id=identity, decomposition=None, cost_law=None,
        construction_work=work, cost_context="selected coherent row: PREP pair, slot phases, input PREP and evolution"
            + ("" if counts["unresolved"] is None else f"; construction work unresolved: {counts['unresolved']}; "
               "the known workspace bytes exclude the synthesis of that definition"),
        # An unknown child or input-preparation cost leaves no CX law rather than an incomplete count.
        resource_laws=() if counts["select_cx"] is None else tuple(ResourceLaw(metric="cx", basis="cx",
            value=counts["select_cx"], interpretation="estimate", evidence=Evidence(kind="numerical_estimate", source=METHOD),
            adjoint=adjoint, assumptions=("selected coherent row's CX law before routing; child laws and Qiskit "
                                          "multi-controlled X counts without ancillas",))
            for adjoint in (False,True)),
        workspace=(Workspace(location="host", purpose="workspace", bytes=counts["known_peak_bytes"], source=METHOD),
                   Workspace(location="host", purpose="workspace", bytes=None, source=METHOD)))
    block = SelectedBlock.bind(record, payload=data, constructor=construct_coherent_lcu)
    definitions = [Definition(id="coherent_lcu", node=BlockCall(signature=signature.name,
        ports=tuple(PortMap(port=f"bit_{j}", wire=f"bit_{j}") for j in range(width))))]
    facts = selected_error_facts(data, problem=problem, raw=raw)
    return finish_quantum_plan(method, problem, rec, output=output, shots=shots, rng=rng, blocks=[block],
        definitions=definitions, calls=["coherent_lcu"], facts=facts, select_work=select_work, native_data=data,
        selection_accuracy=accuracy, readout=None if choice.readout is None else choice.readout[1:],
        coefficient_plan=data.coefficient_plan if data.source_state is None else None)


def lchs_readout(output, terms, coordinates, *, dimension, shots, method):
    """Choose the LCHS readout settings and count the qubit-wise commuting (QWC) grouping against max_readout_work.

    Returns ``(settings, groups, comparisons)``. A vector output reads its
    amplitudes and Samples one counts setting. With ``shots=None`` a scalar
    output has one exact ``projected_moments`` reduction. With shots,
    NormSquared measures its physical mass. A Pauli observable measures one
    counts setting per first-fit qubit-wise commuting group of its nonzero
    nonidentity labels, whose label is the group's accumulated basis, and a
    padded normalized output adds the unrotated ``physical_mass`` setting.
    ``groups`` lists each group setting's member labels. The grouping
    comparisons are checked tile by tile against max_readout_work, and
    comparisons is the number evaluated.
    """
    from nwqlib._quantum_readout import PROJECTED_MOMENTS, ReadoutSetting
    identity = "I"*len(coordinates) or "I"
    projection = dimension != 1 << len(coordinates)
    if isinstance(output,(Solution,StateVector,Samples)):
        return (ReadoutSetting(name="setting_0",label=identity),),(),0
    if shots is None:
        return (ReadoutSetting(name=PROJECTED_MOMENTS,label=identity),),(),0
    if isinstance(output,NormSquared):
        return (ReadoutSetting(name="physical_mass",label=identity,physical_projection=projection),),(),0
    groups, bases, comparisons = qwc_groups(
        terms, len(coordinates), max_comparisons=method.max_readout_work,
        max_bytes=method.max_bytes, limit_name="LCHS.max_readout_work")
    if not groups:
        groups,bases = ((),),(identity,)
    settings = tuple(ReadoutSetting(name=f"group_{index}",label=basis) for index,basis in enumerate(bases))
    if isinstance(output,NormalizedExpectation) and projection:
        settings += (ReadoutSetting(name="physical_mass",label=identity,physical_projection=True),)
    return settings,groups,comparisons


def observed_bits(rec):
    """Return the qubits that every counts setting measures, in classical-bit order.

    Every counts setting observes all success and system qubits, since the
    Program's classical layout is shared by all counts batches. A group
    setting decodes each label's parity from the coordinates of its support
    and ignores the other measured coordinates. The physical-mass setting
    and Samples use every coordinate.
    """
    return rec.success_bits+rec.system_bits


def _rotations(rec, setting, encoded_basis, blocks, definitions, rotations):
    """Return the one-site basis changes of a group setting, building each once per Plan.

    A group basis rotates each X coordinate by H and each Y coordinate by S†
    then H, once, with no parity network.
    """
    q = len(rec.system_bits)
    names = []
    for bit,axis in enumerate(reversed(setting.label)):
        if axis not in "XY":
            continue
        name = f"rotate_{axis.lower()}_{bit}"
        if name not in rotations:
            label = "".join(axis if j==bit else "I" for j in reversed(range(q)))
            block = select_pauli_parity(name,label,basis=encoded_basis)
            blocks.append(block)
            definitions.append(Definition(id=name,node=BlockCall(signature=block.record.signature.name,
                ports=tuple(PortMap(port=f"system_{j}",wire=f"bit_{wire}") for j,wire in enumerate(rec.system_bits)))))
            rotations.add(name)
        names.append(name)
    return names


def _checked_select_work(work, method, *, charged, unresolved=None):
    """Count child planning and construction against the same SELECT budget.

    Child plans report the work they charged even when the budget is unlimited.
    An uncounted construction leaves the total unknown and precludes an explicit limit.
    """
    if work is None:
        _refuse_uncounted_select(method, unresolved)
        return None
    total = work+charged
    if exceeds(total, method.max_select_work):
        raise ResourceLimitError(f"selected LCHS native construction needs {total} work units of max_select_work, {work} "
                                 f"to build it and {charged} to plan its parts, which exceeds "
                                 f"max_select_work={method.max_select_work}")
    return total


def _refuse_uncounted_select(method, reason):
    """Refuse an explicit max_select_work when part of the SELECT construction cannot be counted at planning.

    A supplied operation whose definition is not built has no planning
    count for its controlled query synthesis, so a requested limit cannot
    be checked, and the Plan is refused before any later work. Without a
    limit the Plan records the work as unresolved.
    """
    if method.max_select_work is not None:
        raise ResourceLimitError(f"{reason}, so the SELECT construction cannot be counted against "
                                 f"LCHS.max_select_work={method.max_select_work}. Expand the supplied circuit before "
                                 "ingest_block_encoding, for example with decompose(), so that its definitions are built, "
                                 "or set max_select_work to None")


def finish_quantum_plan(method,problem,rec,*,output,shots,rng,blocks,definitions,calls,facts=(),select_work=0,
                        selection_accuracy=None,readout=None,**native):
    """Share the planned coherent circuit body across the planned readout settings.

    A vector output reads its amplitudes. With ``shots=None`` a scalar output
    has one experiment: the coherent body and one ``projected_moments``
    reduction at its end, tied to the success bits, coordinates, original
    dimension and saved observable. Before the circuit runs, the Run checks
    the reduction's workspace with the saved state against its
    ``simulator_memory_mb`` (``Reducer.workspace``). With shots,
    each counts setting rotates its group basis and measures all success
    and system qubits. ``select_work`` is the max_select_work already
    counted by the construction. ``readout`` is ``(settings, groups,
    comparisons)`` that the planning pass already grouped for ``rec.terms``
    on the same number of system qubits (`sampling_readout`); None groups
    here (`lchs_readout`).

    Raises:
        ApplicabilityError: When a counts analysis that decodes one 64-bit
            index per outcome (``analysis._sampled``,
            ``reduce_sample_arrays``) would read more than 64 measured bits.
            NormSquared reads outcomes of any width
            (``physical_shot_count``).
    """
    from nwqlib._quantum_readout import PROJECTED_MOMENTS, projected_parameters
    from nwqlib.algorithms.protocol import ApplicabilityError
    from nwqlib.core.planning import ObservationPoint
    from .analysis import error_model
    width = len(rec.success_bits)+len(rec.system_bits)
    if shots is not None and width > 64 and not isinstance(output,NormSquared):
        raise ApplicabilityError(f"sampled LCHS readout of {type(output).__name__} needs {width} classical bits, "
                                 "more than its 64-bit count decoder supports")
    settings,groups,comparisons = readout if readout is not None else lchs_readout(output,rec.terms,rec.system_bits,
        dimension=rec.dimension,shots=shots,method=method)
    rec = rec.revise(settings=settings, groups=groups,
        grouping_comparisons=comparisons, select_work=select_work)
    vector = isinstance(output,(Solution,StateVector))
    reduction = shots is None and not vector
    declarations = [Register(name=f"bit_{j}",width=1,role="clean_ancilla" if j in rec.success_bits else "system") for j in range(width)]
    allocate,release = [],[]
    for j in range(width):
        allocate.append(f"allocate_{j}")
        release.append(f"release_{j}")
        definitions.extend((Definition(id=allocate[-1],node=Allocate(wire=f"bit_{j}")),Definition(id=release[-1],node=Release(wire=f"bit_{j}"))))
    classical,measure = (),[]
    if shots is not None:
        # Every counts setting measures all success and system qubits into
        # the shared classical layout, so one measurement per bit serves all.
        classical = tuple(ClassicalValue(name=f"readout_{j}",dtype="bits",width=1) for j in range(width))
        for j,bit in enumerate(rec.success_bits+rec.system_bits):
            measure.append(f"measure_{j}")
            definitions.append(Definition(id=measure[-1],node=Measure(wire=f"bit_{bit}",result=f"readout_{j}")))
    experiments,batches = [],[]
    encoded_basis = Basis(identity="lchs-encoded",dimension=rec.encoded_dimension,ordering="little-endian original index then zero dummy coordinates")
    rotations = set()
    for index,setting in enumerate(rec.settings):
        body = [*allocate,*calls]
        if shots is not None:
            body.extend(_rotations(rec,setting,encoded_basis,blocks,definitions,rotations))
            body.extend(measure)
        if not vector and not reduction:
            body.extend(release)
        definitions.append(Definition(id=f"body_{index}",node=Sequence(children=tuple(body))))
        if vector or reduction:
            continue
        name = f"batch_{index}"
        batches.append(name)
        definitions.append(Definition(id=name,node=MeasurementBatch(body=f"body_{index}",
            settings=(Setting(label=setting.name,metadata=MetadataRef(format=METHOD,data=InputRef(identity=problem.content_id,representation="scientific_input",source=METHOD))),),
            repetitions=shots,observation_kind="counts")))
        experiments.append(Experiment(name=setting.name,batch=name,setting_index=0,readout=ReadoutDetails()))
    root = "body_0" if vector or reduction else "settings"
    if root == "settings":
        definitions.append(Definition(id=root,node=Sequence(children=tuple(batches))))
    program = Program(root=root,definitions=tuple(definitions),registers=tuple(declarations),classical=classical,
        signatures=tuple(block.record.signature for block in blocks))
    construction = SelectedConstruction(program=program,selections=tuple(block.record for block in blocks))
    if vector:
        array = ArrayOutput(name="solution" if isinstance(output,Solution) else "state_vector",kind="vector",basis=problem.basis,
            frame="physical" if isinstance(output,Solution) else output.normalization,
            global_phase="physical" if isinstance(output,Solution) else output.global_phase)
        readout = AmplitudeReadout(construction_id=construction.content_id,source=METHOD,width=width,coordinates=rec.system_bits,
            success=tuple((bit,0) for bit in rec.success_bits),output=array,recovery=rec.recovery)
        experiments = [Experiment(name=rec.settings[0].name,setting=rec.settings[0].name,observation=ObservationSpec(kind="amplitudes",amplitudes=readout))]
    elif reduction:
        parameters = projected_parameters(coordinates=rec.system_bits,success=tuple((bit,0) for bit in rec.success_bits),
            dimension=rec.dimension,terms=rec.terms,moment=not isinstance(output,NormSquared))
        point = ObservationPoint(id="end",kind="reduction",reducer=PROJECTED_MOMENTS,parameters=parameters)
        experiments = [Experiment(name=PROJECTED_MOMENTS,setting=PROJECTED_MOMENTS,
            observation=ObservationSpec(kind="trajectory",positions=(point,)))]
    plan = Plan(problem=problem,method=method,output=output,execution="quantum",shots=shots,randomness=rng.snapshot(),
        selection_accuracy=selection_accuracy,construction=construction,experiments=tuple(experiments),reconstruction=rec,facts=tuple(facts),
        error_model=error_model(problem,output,construction,facts,shots=shots),
        assumptions=("construction tolerance is not a total physical output error bound",
                     *((ASSERTED_L_BOUNDS,) if rec.psd_premise == "user_assertion" else ())))
    return plan._bind(blocks=tuple(blocks),**native)
