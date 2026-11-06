"""Choose LCHS numerical data once, before any circuit or classical evolution."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import NamedTuple

import numpy as np

from nwqlib.algorithms.protocol import ApplicabilityError
from nwqlib.operators.access import _check_bytes
from nwqlib.operators.inputs import _freeze_array
from .provider_config import ProviderConfig
from .providers import _homogeneous_coefficient_plan

# Generator representations that LCHS reads through operators.cartesian,
# without a matrix of the system dimension.
COMPACT_REPRESENTATIONS = frozenset(("pauli", "cartesian", "band", "block_encoding"))


def resolve_evolution_backend(method, problem):
    """Return the concrete Hamiltonian-evolution backend of one planning, from the Method and how A is given.

    An explicit backend is returned unchanged. ``"auto"`` gives
    ``"qsp_block_encoding"`` for a compact generator
    (``COMPACT_REPRESENTATIONS``), which never forms an array of the system
    dimension, and ``"dense_exact"`` for a supplied dense A. ``plan_lchs``
    calls this once per planning, after the initial-condition and
    PeriodicStencil dispatches, which read no backend, and the selected data
    keep the result (``LCHSData.backend``), so loading, estimating and
    running a Plan never resolve again. An explicit
    ``lcu_select_implementation`` that the resolved backend cannot use is a
    representation-dependent conflict of an ``"auto"`` Method, so it raises
    ApplicabilityError here, which ``scientist.compare`` records as a
    rejected row; with an explicit backend ``native.resolve_lcu_select_implementation``
    keeps its ValueError.
    """
    backend = method.hamiltonian_evolution_backend
    if backend != "auto":
        return backend
    compact = problem.A.reference.representation in COMPACT_REPRESENTATIONS
    backend = "qsp_block_encoding" if compact else "dense_exact"
    usable = ("auto",) if compact else ("auto", "branch_controlled")
    if method.lcu_select_implementation not in usable:
        if not compact:
            remedy = "name the backend explicitly to use that SELECT"
        else:
            # Every other backend refuses a compact A ("a compact LCHS generator currently uses ...").
            remedy = "supply a dense A to use that SELECT"
            if method.lcu_select_implementation in ("multiplexor", "structured"):
                remedy += (", or for a scalar output from finite shots on a Pauli A set "
                           "observable_estimator='sampled_pairs' and hamiltonian_evolution_backend='trotter'")
        raise ApplicabilityError(
            f"hamiltonian_evolution_backend='auto' resolves to {backend!r} for this "
            f"{'compact' if compact else 'dense'} A, which accepts lcu_select_implementation "
            f"{' or '.join(map(repr, usable))}, not {method.lcu_select_implementation!r}; {remedy}")
    return backend


@dataclass(frozen=True)
class LCHSGenerator:
    """The generator A = L + iH as the selected recipe reads it.

    It records how A is accessed, the width of the encoded system register
    and the scalars that the recipe takes out of its evolved children. The
    dense recipe evolves the padded arrays L + s*I and H of
    ``DenseLCHSInputs`` and extracts nothing. The compact recipe evolves the
    centered children L_c = L - c_L*I and H_c = H - c_H*I of the operator
    inputs. Since I commutes with every operator,
    exp(-i t (k (L + s*I) + H)) = exp(-i t (k (c_L + s) + c_H)) exp(-i t (k L_c + H_c)),
    and the scalar phase moves into the coefficients
    (providers._homogeneous_coefficient_plan,
    source_selection.select_source_branches).

    Attributes:
        num_qubits: Qubits q of the encoded system register. The dense
            recipe pads a non-power-of-two dimension, so 2**q can exceed the
            Problem's dimension.
        access: ``"dense"`` or ``"compact"``.
        dissipative: Operator input of L for compact access, or None for
            dense access or an exactly zero L.
        hamiltonian: Operator input of H, with the same convention.
        dissipative_facts: ``operators.cartesian.PartFacts`` of L for compact
            access, None for dense access, whose facts are the eigensolve
            results in the quadrature's conversion record.
        hamiltonian_facts: ``PartFacts`` of H, with the same convention.
        dissipative_extracted: The scalar a = c_L + s multiplying k in the
            extracted phase, 0 for dense access.
        hamiltonian_extracted: The scalar b = c_H of the extracted phase, 0
            for dense access.
        zero_evolution: Whether both evolved children are exactly zero, so
            every branch is a scalar phase.
        exact_zero: Whether A is exactly zero and no shift or decay applies,
            so the propagator is the identity.
        generator_norm: Upper bound B_A on ||A|| of the original physical
            generator, which the source-quadrature remainder bound needs, or
            None when unknown at planning (the dense recipe, whose explicit
            refinement computes it).
        parts_commute: ``CartesianAccess.parts_commute`` of compact access,
            the provenance of an exact [L, H] = 0 fact of the representation,
            or None (dense access, or no such fact).
        single_a: The operator input of a general A given only as one
            supplied block encoding (``CartesianAccess.single_a``), whose L
            and H are not separate operators; their facts are the bounds
            ``-||A|| <= L, H <= ||A||``. None otherwise.
    """

    num_qubits: int
    access: str
    dissipative: object
    hamiltonian: object
    dissipative_facts: object
    hamiltonian_facts: object
    dissipative_extracted: float
    hamiltonian_extracted: float
    zero_evolution: bool
    exact_zero: bool
    generator_norm: float | None = None
    parts_commute: str | None = None
    single_a: object = None

    @property
    def encoded_dimension(self):
        """The encoded system dimension 2**q."""
        return 1 << self.num_qubits


@dataclass(frozen=True)
class DenseLCHSInputs:
    """The padded arrays of the dense recipe, which only explicit dense access reads.

    Vectors and matrices are zero-padded to the power-of-two encoded
    dimension (A ⊕ 0). The encoded A is not kept. Readers form the shifted
    matrix ``l_part + 1j*h_part`` where they need it.

    Attributes:
        l_part: Padded Hermitian L after the PSD shift, used in k*L + H.
        h_part: Padded Hermitian H.
        initial: Padded physical initial vector.
        source: Padded physical constant source, or None.
        initial_direction: Padded unit initial direction used by PREP.
        source_direction: Padded unit source direction, or None.
    """

    l_part: object
    h_part: object
    initial: object
    source: object
    initial_direction: object
    source_direction: object


@dataclass(frozen=True)
class LCHSData:
    """Numerical data that the chosen circuit or classical construction uses.

    One instance is shared by every circuit block, classical routine, saved
    archive and explicit check of a Plan. Its content hash
    (parameters.selected_identity) ties it to the planned SELECT record or
    classical routine. It holds the operator and state inputs and scalar
    facts, which every recipe reads, and the padded arrays of the dense
    recipe only when that recipe was selected. A compact generator therefore
    plans, estimates and saves without arrays of the system dimension.

    Attributes:
        generator: The ``LCHSGenerator`` with the encoded width, access,
            operator inputs and scalar facts.
        initial_state: The Problem's initial StateInput.
        source_state: The Problem's source StateInput, or None.
        quadrature: LCHSQuadratureData with the PSD facts, the k-nodes and the
            uncompensated coefficients.
        coefficient_plan: Homogeneous LCHSCoefficientPlan whose coefficients
            include exp(s*T) for the signed shift s and any extracted scalar
            phase, divided by 2**recovery_exponent, which the recovery
            carries.
        method: LCHS Method that chose these data.
        backend: The concrete Hamiltonian-evolution backend of these data,
            ``resolve_evolution_backend`` of the Method and the Problem,
            chosen once at planning. Every later reader of the Plan reads
            it here or in ``LCHSReconstruction.selected_backend``, which
            planning writes from it, never the Method's ``"auto"``.
        dense: ``DenseLCHSInputs`` of the dense recipe, or None.
        select_data: Frozen product-formula SELECT data, or None.
        select_records: Per-address-slot product-formula records of a
            constant-source layout, in address-slot order.
        source_layout: SourceBranchLayout for a constant source on the
            dense_exact and product-formula routes, or None.
        qsp_plan: For ``selected_select == "coherent_lcu"`` (the
            qsp_block_encoding backend), the frozen
            ``coherent_selection.PreparedCoherentRoute``: every coherent row
            with its reasons and forecast, the selected joint generator plan,
            factorized PREP, phases and evolution plan. Otherwise None.
        initial_mps: MPS decomposition of the initial direction chosen at planning, or None.
        coefficient_mps: MPS decomposition of the coefficient PREP chosen at planning, or None. On
            the coherent route, the (kernel, application) pair of factor decompositions in the
            selected label order, each None for a factor without qubits.
        selected_select: Resolved SELECT implementation, ``"coherent_lcu"``
            or an identity route.
        host_actions: Planned classical Pauli actions for classical execution, or
            None. It is a mapping with ``nodes`` (the saved node table,
            time_independent_terms.LCHSProductFormulaNodes) and ``applications``
            (per application its start, weight, source role and per-position
            step record and route).
        applications: ``source_selection.LCHSApplicationFacts`` of the
            coherent route: the initial and Duhamel applications with their
            masses, elapsed times, weights and input norms. None elsewhere.
        provider_selection: For ``lchs_kernel="auto"`` (provider
            enumeration, ``quantum.choose_provider``), one read-only mapping
            per entry of ``PROVIDER_FRONTIER`` that the pass visited, in
            frontier order: ``label``; ``parameters`` (kernel and quadrature
            implementations, their resolved parameters, the derived profile
            parameters such as J, R and its radius rule, and the private
            ``seams`` strip_margin and free_half_width); ``status``
            (``"selected"``, ``"admitted"``, ``"refused"``,
            ``"not_generated"`` or ``"duplicate"``); ``reason``; the selected
            coherent ``row`` and its comparison ``key``; ``quadrature_work``,
            the admitted ``max_quadrature_work`` units of the grid (None for
            a refused grid, whose reason names the units it needed); for
            every entry whose grid was built (selected,
            admitted, or refused after the grid), that grid's ``range_k``,
            ``interval_count_each_side``, ``node_count`` (Q per panel for
            ``composite_gauss``, the total for other rules),
            ``total_node_count`` (physical nodes without LCU padding),
            ``h1``, ``coefficient_l1_norm`` (the uncompensated finite
            kernel-quadrature mass, not gamma) and the unit-input component
            bounds ``approximate_lchs_error_bound`` and
            ``quadrature_error_bound`` before PSD growth; and,
            for the KK6 entry, its ``proposal`` (``providers.kk6_radius``).
            Values that do not exist for an entry are None. Empty for a
            concrete provider pair.
    """

    generator: object
    initial_state: object
    source_state: object
    quadrature: object
    coefficient_plan: object
    method: object
    backend: str
    dense: object = None
    select_data: object = None
    select_records: tuple = ()
    source_layout: object = None
    qsp_plan: object = None
    initial_mps: object = None
    coefficient_mps: object = None
    selected_select: str = "branch_controlled"
    host_actions: object = None
    applications: object = None
    provider_selection: tuple = ()

    def dense_inputs(self):
        """Return the ``DenseLCHSInputs`` of the dense recipe, refusing a compact generator.

        A reference, refinement or classical routine that needs the padded
        arrays calls this. A compact generator has no such arrays, and none
        is formed for it.
        """
        if self.dense is None:
            raise ApplicabilityError("this LCHS Plan has a compact generator and no dense arrays; "
                                     "dense references and classical evaluation need a supplied dense A")
        return self.dense


def _pad_vector(values, dimension):
    """Return values zero-padded to the encoded dimension, frozen read-only.

    The dummy coordinates of the A ⊕ 0 encoding start at zero, so the padded
    initial and source vectors evolve the original block unchanged.
    """
    if values is None:
        return None
    if len(values) == dimension:
        return values
    output = np.zeros(dimension, dtype=complex)
    output[:len(values)] = values
    return _freeze_array(output)


@dataclass(frozen=True)
class LCHSCommon:
    """Facts of one planning pass that every provider grid shares, before any grid exists.

    ``compact_common`` and ``dense_common`` build it from the Problem and
    Method; ``with_providers`` adds one kernel-quadrature grid to make the
    ``LCHSData`` of a candidate. Nothing here depends on the kernel, the
    k-quadrature or their parameters, so a refusal of one provider leaves
    the others available. It is not saved; the selected ``LCHSData`` is.

    Attributes:
        generator: The ``LCHSGenerator``.
        initial_state: The Problem's initial StateInput.
        source_state: The Problem's source StateInput, or None.
        method: The LCHS Method of this planning.
        backend: The resolved Hamiltonian-evolution backend.
        l_norm: The bound B_Ls on ||L + s*I|| that selects every grid: the
            dense recipe's numerical spectral norm of the shifted L, or the
            compact recipe's structural ``processed_norm``.
        conversion: Read-only PSD metadata (``LCHSQuadratureData.conversion``).
        psd_premise_satisfied: Whether L + s*I is known to be PSD, by the
            evidence that ``conversion["psd_premise_evidence"]`` names.
        dense: ``DenseLCHSInputs`` of a supplied dense A, else None.
    """

    generator: object
    initial_state: object
    source_state: object
    method: object
    backend: str
    l_norm: float
    conversion: object
    psd_premise_satisfied: bool
    dense: object = None


def dense_common(method, problem, *, backend):
    """Return the ``LCHSCommon`` of a supplied dense A on the padded encoding, with one eigensolve and no grid.

    ``backend`` is the resolved backend (``resolve_evolution_backend``).
    Only supplied dense A and explicit vectors are accepted, so no compact
    input is densified implicitly. Missing vector data is rejected before the
    eigensolve, and the byte limit is checked before the Cartesian arrays are
    formed. A non-power-of-two dimension is encoded as A ⊕ 0 with zero-padded
    inputs. The analytic zero endpoint of the dummy block joins the PSD
    decision without a second eigensolve (``_prepare_decomposition``).
    """
    from .time_independent_terms import _prepare_decomposition
    if problem.A.reference.representation != "dense":
        raise ApplicabilityError("this LCHS realization needs a supplied dense A or PeriodicStencil; no automatic densification")
    states = (problem.initial_state,) + (() if problem.source is None else (problem.source,))
    if any(state.reference.representation != "vector" for state in states):
        raise ApplicabilityError("dense LCHS needs explicit physical initial/source vectors")
    # These accesses return the existing immutable vectors. Reject missing
    # native data before the unrelated spectral/quadrature computation.
    for state in states:
        state.physical_vector()
    dimension = 1 << max(1, (problem.dimension-1).bit_length())
    # The enclosing selection law 64*D**2 + 128*D: four complex128 D-square
    # arrays for A, L, H and the padding or eigensolver working copy of
    # _prepare_decomposition's allocation schedule, and eight complex128
    # D-vectors for the padded physical and direction vectors of u0 and b and
    # their copies. The O(D) diagonal and spectral temporaries run before
    # those vectors are formed. The queried eigensolver workspace is not
    # charged (a declared known-array workspace, not a complete cap).
    _check_bytes(64*dimension**2 + 128*dimension, method.max_bytes, "LCHS encoded inputs and Cartesian arrays")
    prepared = _prepare_decomposition(matrix=problem.A.dense_array(), method=method, padded_dimension=dimension,
                                      max_spectral_work=method.max_spectral_work)
    # L and H are private arrays already adopted read-only by
    # _prepare_decomposition, so they are kept without another copy.
    l_part, h_part = prepared.l_part, prepared.h_evolution
    conversion = MappingProxyType(dict(prepared.conversion))
    source = source_direction = None
    if problem.source is not None:
        source = _pad_vector(problem.source.physical_vector(), dimension)
        source_direction = _pad_vector(problem.source._direction, dimension)
    dense = DenseLCHSInputs(l_part=l_part, h_part=h_part,
        initial=_pad_vector(problem.initial_state.physical_vector(), dimension), source=source,
        initial_direction=_pad_vector(problem.initial_state._direction, dimension), source_direction=source_direction)
    zero = not (np.any(l_part) or np.any(h_part))
    generator = LCHSGenerator(num_qubits=dimension.bit_length()-1, access="dense", dissipative=None,
        hamiltonian=None, dissipative_facts=None, hamiltonian_facts=None, dissipative_extracted=0.0,
        hamiltonian_extracted=0.0, zero_evolution=zero, exact_zero=zero and conversion["psd_shift"] == 0)
    return LCHSCommon(generator, problem.initial_state, problem.source, method, backend, prepared.l_norm, conversion,
                      prepared.numerical_psd_premise_satisfied, dense)


def select_dense(method, problem, *, backend, dense_select_slot_limit=None):
    """Select the homogeneous coefficient grid of the Method's pair for dense A (`dense_common`, `with_providers`)."""
    return with_providers(dense_common(method, problem, backend=backend), problem,
                          dense_select_slot_limit=dense_select_slot_limit)


def _original_generator_norm(access, compensation):
    """Return the bound B_A on ||A|| of the original generator for the source-quadrature remainder, or None.

    The remainder of ``inhomogeneous_theory.duhamel_log_bound`` comes from
    derivatives of f(r) = exp(-A (T - r)) b, so it needs the original A,
    including any known decay, not L + s*I or a centered child. The
    structural bound of the parts is used when it exists, also under a declared
    lower bound of L, so this norm factor does not rest on the declaration. For
    structural lower and upper bounds ell <= u of L it is never looser than the
    fallback: max(|ell|, |u|) <= (u - ell) + |ell|. Otherwise L = (L + s*I) - s*I
    gives ||L|| <= B_Ls + |s| for the signed shift s, which is negative
    under known decay, and ||A|| <= ||L|| + ||H||. For L = 2I, s = -2 and
    L + s*I = 0, the absolute value keeps the bound at 2.
    """
    from fractions import Fraction
    from nwqlib.operators.cartesian import _outward
    if access.generator_norm is not None:
        return access.generator_norm
    if access.hamiltonian_facts.norm is None:
        return None
    return _outward(Fraction(compensation.processed_norm) + abs(Fraction(compensation.signed_shift))
                    + Fraction(access.hamiltonian_facts.norm), upward=True)


def compact_common(method, problem, *, backend=None):
    """Return the ``LCHSCommon`` of a compact generator from its parts and structural facts, with no grid.

    ``backend`` is the resolved backend that the facts keep; None resolves
    it here (``resolve_evolution_backend``), for a caller such as the
    sampled-pair planner whose Method names its backend.

    The generator is a Pauli sum, an ``ingest_cartesian`` input, a Hermitian
    band operator or a supplied block encoding. ``operators.cartesian``
    gives its parts L and H and their exact structural bounds, with work
    proportional to the number of terms or bands.
    ``operators.cartesian.scalar_compensation`` turns the lower bound ell of
    L into the signed shift s = -ell (the nonnegative PSD shift minus the
    known decay), and the kernel and quadrature use the
    bound ``processed_norm`` on ||L + s*I||. No vector or matrix of the
    system dimension and no eigensolve is formed. A lower bound declared to
    ``ingest_cartesian`` enters ell as a premise, and
    ``conversion["psd_premise_evidence"]`` keeps it as ``"user_assertion"``
    for every consumer of the shift. A negative ell is an
    enclosure, not a finding that L has a negative eigenvalue, so
    ``make_l_psd=False`` refuses it with ApplicabilityError, because the PSD
    premise is not established rather than violated, and
    ``scientist.compare`` records a rejected row. That option excludes a
    positive PSD shift, not the extraction of known decay from a positive ell.
    A dense L with a computed
    negative eigenvalue raises ApplicabilityError too, and an access that exceeds
    ``max_bytes`` keeps its ResourceLimitError.
    """
    from nwqlib._limits import ResourceLimitError
    from nwqlib.operators.cartesian import cartesian_access, scalar_compensation
    try:
        access = cartesian_access(problem.A, max_bytes=method.max_bytes)
    except ResourceLimitError:
        raise
    except ValueError as error:
        raise ApplicabilityError(str(error)) from error
    l_facts, h_facts = access.dissipative_facts, access.hamiltonian_facts
    try:
        compensation = scalar_compensation(l_facts)
    except ValueError as error:
        raise ApplicabilityError(f"compact LCHS cannot establish the PSD premise of L: {error}") from error
    if compensation.psd_shift > 0 and not method.make_l_psd:
        raise ApplicabilityError(
            f"the lower bound {l_facts.lower!r} of L ({l_facts.evidence}) is negative, so it does not establish L >= 0, "
            "and make_l_psd=False forbids the positive shift it needs; this is an enclosure, not a negative eigenvalue. "
            "If you know a larger lower bound, state it with ingest_cartesian(L, H, dissipative_lower_bound=...), or "
            "lower_bound of ingest_block_encoding for a supplied Hermitian encoding of L. A periodic L given through "
            "ingest_band can have a tighter structural bound. make_l_psd=True shifts by whatever negative bound remains")
    shift = compensation.signed_shift
    conversion = MappingProxyType({"psd_shift": compensation.psd_shift, "scalar_decay": compensation.scalar_decay,
        "psd_premise_evidence": l_facts.evidence, "l_lower_bound": l_facts.lower, "l_upper_bound": l_facts.upper,
        "make_l_psd": method.make_l_psd})
    # A Pauli or band part keeps its identity coefficient c_X outside its
    # centered child. A supplied encoding gives none, so c_X = 0 there.
    centered_zero = [facts.access == "zero" or facts.access in ("pauli", "band") and facts.centered_alpha == 0
                     for facts in (l_facts, h_facts)]
    generator = LCHSGenerator(num_qubits=problem.dimension.bit_length()-1, access="compact",
        dissipative=access.dissipative, hamiltonian=access.hamiltonian, dissipative_facts=l_facts,
        hamiltonian_facts=h_facts, dissipative_extracted=l_facts.identity+shift,
        hamiltonian_extracted=h_facts.identity, zero_evolution=all(centered_zero),
        exact_zero=l_facts.access == h_facts.access == "zero",
        generator_norm=_original_generator_norm(access, compensation), parts_commute=access.parts_commute,
        single_a=access.single_a)
    return LCHSCommon(generator, problem.initial_state, problem.source, method,
                      resolve_evolution_backend(method, problem) if backend is None else backend,
                      compensation.processed_norm, conversion, True)


def select_compact(method, problem, *, backend=None):
    """Select the coefficient grid of the Method's pair for a compact generator (`compact_common`, `with_providers`)."""
    return with_providers(compact_common(method, problem, backend=backend), problem)


def with_providers(common, problem, entry=None, *, work_ledger=None, dense_select_slot_limit=None):
    """Return the ``LCHSData`` of one kernel-quadrature grid on the shared facts of a planning pass.

    ``entry`` is a ``ProviderEntry`` of the provider frontier; None uses the
    Method's pair. Only the quadrature data and the homogeneous coefficient
    plan depend on it: the generator, inputs and PSD facts are those of
    ``common``, so no spectral or structural work repeats. The grid is a
    scalar computation of its node count, bounded by ``max_quadrature_work``
    and ``max_bytes`` for this one construction, with no phase synthesis, circuit
    or spectral work. A supplied ``work_ledger`` receives the admitted
    quadrature work, and ``dense_select_slot_limit`` is the slot limit of a
    quantum dense_exact route (``providers._build_composite_gauss``).
    """
    from .time_independent_terms import _quadrature_data_from_plan
    method, generator = common.method, common.generator
    quadrature = _quadrature_data_from_plan(l_norm=common.l_norm, conversion=common.conversion,
        psd_premise_satisfied=common.psd_premise_satisfied, final_time=problem.elapsed_time, method=method,
        max_bytes=method.max_bytes, max_quadrature_work=method.max_quadrature_work, providers=entry,
        work_ledger=work_ledger, dense_select_slot_limit=dense_select_slot_limit)
    quadrature = replace(quadrature, k_nodes=_freeze_array(quadrature.k_nodes),
                         coefficients=_freeze_array(quadrature.coefficients))
    coefficients = _homogeneous_coefficient_plan(quadrature, problem.elapsed_time,
        scalar_phase=(generator.dissipative_extracted, generator.hamiltonian_extracted), max_bytes=method.max_bytes)
    return LCHSData(generator, common.initial_state, common.source_state, quadrature, coefficients, method,
                    common.backend, dense=common.dense)


class ProviderEntry(NamedTuple):
    """One candidate of the coherent provider choice: its kernel, k-quadrature and private seams.

    Attributes:
        label: Name in ``LCHSData.provider_selection``.
        lchs_kernel: Kernel ``ProviderConfig``, with only public parameters.
        k_quadrature: Quadrature ``ProviderConfig``.
        free_half_width: J' of the Low-Somma free uniform grid, else None.
        strip_margin: Gauss panel strip margin d other than the default, else None.
        proposal: The ``providers.kk6_radius`` record of the KK6 entry, else None.
        skip: ``(status, reason)`` of an entry that has no grid
            (``"not_generated"`` or ``"duplicate"``); it is recorded, never built.
    """

    label: str
    lchs_kernel: ProviderConfig
    k_quadrature: ProviderConfig
    free_half_width: int | None = None
    strip_margin: float | None = None
    proposal: Mapping | None = None
    skip: tuple | None = None


def _entry(label, kernel, quadrature, **kernel_parameters):
    return ProviderEntry(label, ProviderConfig(implementation=kernel, parameters=kernel_parameters),
                         ProviderConfig(implementation=quadrature))


# The finite provider frontier of lchs_kernel="auto" with k_quadrature="auto" (PLAN M2/M3, kernel-source
# supplement section 2.2), in tie-break order: equal comparison keys keep the earlier entry, so the default
# pair wins ties. ACL beta 0.75 (default), 0.70 and 0.80 span the range ACL arXiv:2312.03916v2 section 2.2
# reports as numerically best for truncation; d = 0.5 is a second strip margin of the Gauss panel bound
# (providers._ellipse_rule, valid for every d in (0, 1)). Each ACL beta and d uses the one least-node panel
# rule of _ellipse_rule and truncation_multiplier = 1; other panel rules at the same beta and d are not
# compared. Low-Somma c = 1 (default), 1/2 and 2 lie in the domain c > 0 of Theorem 2, and each is followed by
# its free uniform grid when J' != J (provider_frontier). The Apel-Berntson paper root closes the fixed list;
# provider_frontier adds the one KK6 radius for Samples and accuracy requests. These are engineering values
# without a dominance or optimality claim: another panel rule, multiplier or parameter can have a lower
# complete cost, and every entry is ranked by its actual finite grid and coherent row.
# ponytail: finite heuristic list; add an entry only when the selection-only keeper table shows it winning on a
# representative input.
PROVIDER_FRONTIER = (
    _entry("acl", "near_optimal_eq7", "composite_gauss"),
    _entry("acl_beta_0.7", "near_optimal_eq7", "composite_gauss", beta=.7),
    _entry("acl_beta_0.8", "near_optimal_eq7", "composite_gauss", beta=.8),
    _entry("acl_strip_0.5", "near_optimal_eq7", "composite_gauss")._replace(strip_margin=.5),
    _entry("low_somma", "low_somma_f2", "symmetric_uniform_trapezoid"),
    _entry("low_somma_c_0.5", "low_somma_f2", "symmetric_uniform_trapezoid", c=.5),
    _entry("low_somma_c_2", "low_somma_f2", "symmetric_uniform_trapezoid", c=2.),
    _entry("apel_berntson", "apel_berntson", "tanh_sinh"),
)


def provider_frontier(method, problem, common, *, acquisition, context, records):
    """Yield the provider entries of one coherent planning pass, one at a time, in frontier order.

    A concrete pair is the one entry of the Method's pair. A zero
    ||L + s I|| has the unitary reduction for every pair, so it is the one
    entry ``unitary_reduction``. An enumeration request yields
    ``PROVIDER_FRONTIER`` with two derived kinds of entry, which read the
    caller's ``records`` (one per yielded entry, appended before the next
    is requested):

    - After each Low-Somma entry, its free uniform grid (PLAN M3) with
      J' = 2**p - 1, the smallest such value at least the built profile's
      J, and h = R/J', when J' != J. Then h <= R/J <= h_max, so Theorem 3's
      step condition and the component caps still hold with the same R and
      gamma (``providers._resolve_coefficient_context``).
    - For ``"rejection"`` (Samples) and ``"accuracy"`` acquisitions, one
      Apel-Berntson radius from ``providers.kk6_radius`` (`_kk6_entry`).
      A fixed-shot or exact single-circuit objective, in which the kernel
      mass does not enter the ranked cost, generates none; this is a
      finite policy, not a proof that no other radius lowers C.

    So an enumeration pass visits at most 12 entries: 4 ACL, at most 6
    Low-Somma and 2 Apel-Berntson.
    """
    if common.l_norm == 0:
        yield ProviderEntry("unitary_reduction", method.lchs_kernel, method.k_quadrature)
        return
    if method.lchs_kernel.implementation != "auto":
        yield ProviderEntry(method.lchs_kernel.implementation, method.lchs_kernel, method.k_quadrature)
        return
    for entry in PROVIDER_FRONTIER:
        yield entry
        if entry.lchs_kernel.implementation == "low_somma_f2":
            free = _free_grid_entry(entry, records[-1])
            if free is not None:
                yield free
    if acquisition in ("rejection", "accuracy"):
        yield _kk6_entry(method, problem, common, context, records[-1], accuracy=acquisition == "accuracy")


def _free_grid_entry(entry, record):
    """Return the free uniform grid entry after a Low-Somma entry, None when J' = J."""
    label = entry.label + "_free_grid"
    derived = record["parameters"].get("derived", {})
    if "J" not in derived:
        return entry._replace(label=label, skip=("not_generated", f"{entry.label} built no grid, so its J is "
                                                                   "unknown"))
    J = int(derived["J"])
    free = (1 << J.bit_length()) - 1
    return None if free == J else entry._replace(label=label, free_half_width=free)


def _kk6_entry(method, problem, common, context, paper, *, accuracy):
    """Return the KK6 entry: one Apel-Berntson radius from the access form of this pass (KK6 policy K4).

    The form follows the actual access: an explicit
    ``qsp_route.parameters["joint"]``, else ``"single_a"`` when the
    generator is one supplied encoding of A and ``"separate_parts"``
    otherwise. Its query normalizations are those of the actual compiled
    children (``joint.access_summary``), a = alpha_L (0 for an absent L
    child) and b = alpha_H (0 for an absent H child), or alpha_A of the
    supplied A. A requested transformation without a kernel radius
    (``imaginary_gqet``, ``commuting_product``, a refused route or the
    ``convex_contraction`` joint form), a node encoding other than
    ``"table"``, which the Apel-Berntson nonuniform grid cannot meet, or an
    absent access form generates nothing. The continuous allowance
    epsilon_c is the AB pair's ``kernel_approximation`` allowance for this
    Problem. A radius exactly equal to the built paper root is recorded as a
    duplicate, without a closeness threshold.
    """
    from nwqlib.subroutines.block_encoding.joint import access_summary
    from .coherent_selection import _REAL_TIME_ROUTES
    from .providers import LCHSProblemContext, _component_allowances, kk6_radius
    label, kernel, quadrature = "apel_berntson_kk6", ProviderConfig(implementation="apel_berntson"), \
        ProviderConfig(implementation="tanh_sinh")
    request = dict(method.qsp_route.parameters)
    route = None if method.qsp_route.implementation == "auto" else method.qsp_route.implementation
    joint = request.get("joint") or ("single_a" if common.generator.single_a is not None else "separate_parts")
    reason = None
    if route is not None and route not in _REAL_TIME_ROUTES:
        reason = f"the requested qsp_route {route!r} has no kernel radius to propose"
    elif request.get("node", "table") != "table":
        reason = f"the Apel-Berntson nonuniform grid cannot meet the requested node encoding {request['node']!r}"
    elif joint == "convex_contraction":
        reason = "the requested convex_contraction joint form has no kernel radius to propose"
    elif joint == "single_a" and common.generator.single_a is None:
        reason = "single_a was requested, but this generator is not one supplied encoding of A"
    elif joint == "separate_parts" and common.generator.single_a is not None:
        reason = "separate_parts was requested, but this generator has no separate compiled children"
    if reason is not None:
        return ProviderEntry(label, kernel, quadrature, skip=("not_generated", reason))
    allowances = _component_allowances(LCHSProblemContext(final_time=float(problem.elapsed_time),
        epsilon=float(method.approximation_tolerance), l_norm=common.l_norm), ("apel_berntson", "tanh_sinh"),
        "'apel_berntson' + 'tanh_sinh'")
    if joint == "single_a":
        inputs = dict(alpha_a=access_summary(common.generator.single_a).alpha)
    else:
        l_child, h_child = context.children
        inputs = dict(a=0.0 if l_child is None else access_summary(l_child).alpha,
                      b=0.0 if h_child is None else access_summary(h_child).alpha)
    proposal = kk6_radius(allowances["kernel_approximation"], form=joint, max_quadrature_work=method.max_quadrature_work,
                          **inputs)
    proposal = MappingProxyType({**proposal, "role": "p = 2 rejection-surrogate proposal, ranked by its actual "
                                 "N G C" if accuracy else "p = 2 stationary proposal of the rejection surrogate"})
    if proposal["radius"] is None:
        return ProviderEntry(label, kernel, quadrature, proposal=proposal, skip=("not_generated", proposal["reason"]))
    entry = ProviderEntry(label, ProviderConfig(implementation="apel_berntson",
                                                parameters={"radius": proposal["radius"]}), quadrature,
                          proposal=proposal)
    derived = paper["parameters"].get("derived", {})
    if derived.get("R") == proposal["radius"]:
        return entry._replace(skip=("duplicate", f"the proposed radius equals the {paper['label']} paper root"))
    return entry


_NO_ACCURACY = ("LCHS sampling accuracy (LCHS.sampling_shots) is implemented for the coherent "
                "hamiltonian_evolution_backend='qsp_block_encoding' route and the no-evolution identities; give "
                "explicit shots for this backend, whose counts report their sampling radius")


def _route(method, problem, *, output, execution, accuracy=None):
    """Check the inputs that every LCHS planning shares and return its route, before any route-specific choice.

    Zero elapsed time, or a zero initial vector with no nonzero source, is
    the initial-condition identity with no evolution, ``("initial", zero)``.
    A PeriodicStencil A is the compact quantum Strang route,
    ``("periodic", None)``, which needs the fixed second-order trotter
    backend, auto/structured SELECT, direct PREP and no source; another
    choice raises ApplicabilityError. A compact generator (Pauli sum,
    Cartesian parts, Hermitian band or supplied encoding) and a dense A resolve their
    Hamiltonian-evolution backend once (``resolve_evolution_backend``). The
    coherent qsp_block_encoding backend returns the provider-independent
    facts, ``("coherent", common)`` (``compact_common`` or ``dense_common``),
    which build no provider grid, for ``quantum.choose_provider``. A dense
    A on another backend selects its numerical data, ``("selected", data)``,
    with the backend in ``data.backend``. A provider-enumeration request
    (``lchs_kernel="auto"`` with ``k_quadrature="auto"``) is refused with
    ApplicabilityError off the coherent route or without quantum execution.
    Input scales, observable coordinates and supported PREP choices are
    checked before any spectral or quadrature work, and an
    ``accuracy`` request on a backend without its connector is refused
    before that work as well, so an unsupported request never spends it.
    """
    states=(problem.initial_state,)+(() if problem.source is None else (problem.source,))
    if states[0].preparation.physical_scale is None or (problem.elapsed_time>0 and
            any(state.preparation.physical_scale is None for state in states[1:])):
        raise ApplicabilityError("LCHS initial/source input needs its actual ingested physical scale")
    observable=getattr(output,'observable',None)
    if observable is not None:
        observable._require_data()
        if observable.manifest.basis != problem.basis:
            raise ValueError("observable must use the original physical coordinates")
        # The quantum readout reads Pauli terms, the supplied ones or those
        # of a dense matrix (quantum.observable_terms).
        if execution=='quantum' and observable.reference.representation not in ('pauli', 'dense'):
            raise ApplicabilityError("LCHS quantum observable needs finite real Pauli access or an explicit dense matrix")
    # The synthesis tolerance inverts the weighted Strang bound of the
    # periodic route. Fixed-step dense routes publish no synthesis bound at
    # planning, so there is nothing to invert.
    if method.trotter_synthesis_tolerance is not None and problem.A.reference.representation != "periodic_stencil":
        raise ApplicabilityError("trotter_synthesis_tolerance selects steps only for a PeriodicStencil A. " + (
            "This compact A plans with hamiltonian_evolution_backend='qsp_block_encoding' or 'auto', without a step "
            "count or tolerance" if problem.A.reference.representation in COMPACT_REPRESENTATIONS else
            "Give trotter_steps, or use trotter_error_budgeted for per-node step selection"))
    zero = (problem.initial_state.preparation.physical_scale.mantissa == 0 and
            (problem.elapsed_time==0 or problem.source is None or problem.source.preparation.physical_scale.mantissa == 0))
    if problem.elapsed_time == 0 or zero:
        return "initial", zero
    if problem.A.reference.representation == "periodic_stencil":
        if execution != "quantum":
            raise ApplicabilityError("PeriodicStencil LCHS currently has a quantum Strang realization; no implicit dense host model")
        if method.hamiltonian_evolution_backend == "auto":
            # The periodic route is a product formula, whose step count "auto" does not choose; the default step
            # count 1 would leave its synthesis error uncontrolled.
            raise ApplicabilityError("a PeriodicStencil A uses the second-order product formula; set "
                                     "hamiltonian_evolution_backend='trotter' with trotter_steps or "
                                     "trotter_steps=None and trotter_synthesis_tolerance")
        # The periodic Strang construction is the only one for this A, so another choice is a legal request
        # that no construction supports, refused before any grid.
        if (method.hamiltonian_evolution_backend != "trotter" or method.trotter_order != 2
                or method.lcu_select_implementation not in {"auto", "structured"}):
            raise ApplicabilityError("periodic LCHS requires fixed second-order trotter and auto/structured SELECT")
        if method.initial_state_preparation != "direct" or method.lcu_state_preparation != "direct":
            raise ApplicabilityError("periodic LCHS uses direct basis/product/input and coefficient PREP")
        if problem.source is not None:
            raise ApplicabilityError("PeriodicStencil LCHS currently has a homogeneous Strang realization")
        if accuracy is not None:
            raise ApplicabilityError(_NO_ACCURACY)
        return "periodic", None
    backend = resolve_evolution_backend(method, problem)
    if accuracy is not None and backend != "qsp_block_encoding":
        raise ApplicabilityError(f"{_NO_ACCURACY}; this Plan's backend is {backend!r}")
    enumeration = method.lchs_kernel.implementation == "auto"
    if enumeration and (backend != "qsp_block_encoding" or execution != "quantum"):
        raise ApplicabilityError("lchs_kernel='auto' with k_quadrature='auto' compares the providers by the "
                                 "rows of the coherent qsp_block_encoding route with quantum execution; this request "
                                 f"has backend {backend!r} and execution {execution!r}, so give a concrete pair")
    if method.lcu_state_preparation == "analytic" and backend != "qsp_block_encoding":
        raise ApplicabilityError("lcu_state_preparation='analytic' prepares the kernel factor of the coherent "
                                 f"qsp_block_encoding LCU; hamiltonian_evolution_backend='auto' chose {backend!r} "
                                 "for this A")
    # The coherent route prepares the application and kernel factors of its coefficient state separately, so an
    # MPS coefficient PREP extends to a source there, one MPS circuit per factor; the branch layouts of the other
    # backends have no such construction.
    if execution == "quantum" and problem.source is not None and (
            method.initial_state_preparation != "direct" or (method.lcu_state_preparation != "direct"
                                                             and backend != "qsp_block_encoding")):
        raise ApplicabilityError("constant-source LCHS currently has direct PREP only, except an MPS coefficient "
                                 "PREP with hamiltonian_evolution_backend='qsp_block_encoding'")
    if problem.A.reference.representation in COMPACT_REPRESENTATIONS:
        if execution != "quantum":
            raise ApplicabilityError("a compact LCHS generator has no classical evaluation; it would need the dense matrix")
        if backend != "qsp_block_encoding":
            raise ApplicabilityError("a compact LCHS generator currently uses hamiltonian_evolution_backend="
                                     "'qsp_block_encoding' (or 'auto'); the other backends need a supplied dense A")
        if method.initial_state_preparation == "mps_circuit":
            raise ApplicabilityError("MPS initial-state preparation needs the dense initial vector; "
                                     "a compact generator uses the state's own preparation")
        return "coherent", compact_common(method, problem, backend=backend)
    if backend == "qsp_block_encoding":
        return "coherent", dense_common(method, problem, backend=backend)
    # Only the quantum dense_exact SELECT synthesizes a controlled dense exponential per address slot, which this limit
    # bounds. The same grid evaluated classically, or with product-formula branches, has no such limit.
    slot_limit = method.max_dense_select_slots if execution == "quantum" and backend == "dense_exact" else None
    return "selected", select_dense(method, problem, backend=backend, dense_select_slot_limit=slot_limit)


def plan_lchs(method, problem, *, output, execution, shots, rng, accuracy=None):
    """Send one LinearDynamics to the one LCHS planning function that handles it.

    `_route` checks the shared inputs and chooses the route: the
    initial-condition identity, the PeriodicStencil Strang route, or the
    selected numerical data of a compact or dense A, which go to classical
    (dense only) or quantum planning. A legal input with a combination of
    choices that no current construction supports raises
    ApplicabilityError, which ``scientist.compare`` records as a rejected
    row. An ``accuracy`` request (``LCHS.sampling_shots``) reaches only the
    no-evolution identity and the coherent route, which check that
    ``shots`` is the count it selects and record the request in the Plan.
    The coherent route chooses its provider grid and row once in this pass
    (``quantum.choose_provider``) and prepares only that row
    (``quantum.plan_coherent_lcu``).
    """
    if accuracy is not None and execution == "quantum":
        route, value = _accuracy_route(method, problem, output=output, accuracy=accuracy)
    else:
        route, value = _route(method, problem, output=output, execution=execution, accuracy=accuracy)
    if route == "initial":
        from .host import plan_initial
        return plan_initial(method, problem, output=output, execution=execution, shots=shots, rng=rng, zero=value,
                            accuracy=accuracy)
    if route == "periodic":
        from .periodic import plan_periodic
        return plan_periodic(method, problem, output=output, shots=shots, rng=rng)
    if route == "coherent":
        from .quantum import plan_coherent_lcu
        if accuracy is None:
            from .quantum import choose_provider
            choice = choose_provider(method, problem, value, output=output, shots=shots)
        else:
            choice = _accuracy_choice(method, problem, value, output=output, accuracy=accuracy)
        return plan_coherent_lcu(method, problem, choice, output=output, shots=shots, rng=rng, accuracy=accuracy)
    if execution == "classical":
        from .host import plan_classical
        return plan_classical(method, problem, value, output=output, rng=rng)
    from .quantum import plan_quantum
    return plan_quantum(method, problem, value, output=output, shots=shots, rng=rng)


def _accuracy_route(method, problem, *, output, accuracy):
    """Return `_route` for a quantum accuracy request, once per planning call (`core.planning.planning_memo`)."""
    from nwqlib.core.planning import planning_memo
    return planning_memo("lchs_route", (method, problem, output, accuracy),
                         lambda: _route(method, problem, output=output, execution="quantum", accuracy=accuracy))


def _accuracy_choice(method, problem, common, *, output, accuracy):
    """Return the coherent choice of an accuracy request, once per planning call.

    With ``accuracy``, ``quantum.choose_provider`` reads no shots, so the
    choice that `accuracy_shots` makes is the one that planning makes.
    """
    from nwqlib.core.planning import planning_memo
    from .quantum import choose_provider
    return planning_memo("lchs_coherent_choice", (method, problem, common, output, accuracy),
                         lambda: choose_provider(method, problem, common, output=output, shots=None,
                                                 accuracy=accuracy))


def accuracy_shots(method, problem, *, output, accuracy):
    """Return the shots per counts setting that planning selects for a sampling-accuracy request, by scalar work only.

    The same route as `plan_lchs` (`_route`), then the no-evolution count
    (``host.initial_accuracy_shots``) or the selected row's count of the
    coherent choice (``quantum.choose_provider``). It fits no phase,
    decomposes no MPS, builds no circuit and draws nothing. The route and
    the coherent choice are computed once per planning call
    (`core.planning.planning_memo`), so ``plan`` reads them instead of
    repeating them and checks that its shots are this count. None means the
    Plan acquires nothing. Sampled pairs (``observable_estimator="sampled_pairs"``)
    branch here to ``sampled_plan.sampled_accuracy_shots``, which returns the
    B*m shots of B independent rows.
    """
    if method.observable_estimator == "sampled_pairs":
        from .sampled_plan import sampled_accuracy_shots
        return sampled_accuracy_shots(method, problem, output=output, accuracy=accuracy)
    route, value = _accuracy_route(method, problem, output=output, accuracy=accuracy)
    if route == "initial":
        from .host import initial_accuracy_shots
        return initial_accuracy_shots(method, problem, output=output, accuracy=accuracy)
    choice = _accuracy_choice(method, problem, value, output=output, accuracy=accuracy)
    return choice.selection.selected.cost["shots"]
