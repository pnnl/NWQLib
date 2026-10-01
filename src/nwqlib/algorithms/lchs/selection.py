"""Select LCHS numerical data once before any circuit or host evolution."""

from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np

from nwqlib.algorithms._eigen_inputs import AUTO_DENSE_PAULI_DIMENSION
from nwqlib.algorithms.protocol import ApplicabilityError
from nwqlib.operators.access import _check_bytes
from nwqlib.operators.inputs import _freeze_array
from .providers import _homogeneous_coefficient_plan


@dataclass(frozen=True)
class LCHSData:
    """Numerical products consumed by the chosen native/host realization.

    One instance is shared by every native leaf, host kernel, archive and
    explicit check of a Plan. Its content hash (parameters.selected_identity)
    binds it to the selected SELECT record or host kernel. Vectors and
    matrices are zero-padded to the power-of-two encoded dimension (A ⊕ 0).
    The encoded A is not kept: readers take shapes from the prepared L,
    test for a zero generator with ``np.any(l_part) or np.any(h_part)`` and
    form the shifted matrix ``l_part + 1j*h_part`` where they need it.

    Attributes:
        initial: Padded physical initial vector.
        source: Padded physical constant source, or None.
        initial_direction: Padded unit initial direction used by PREP.
        source_direction: Padded unit source direction, or None.
        quadrature: LCHSQuadratureData with the prepared L (after any PSD
            shift), H, the k-nodes and the uncompensated coefficients.
        coefficient_plan: Homogeneous LCHSCoefficientPlan whose coefficients
            include the exp(shift*T) compensation, divided by a power of two
            that the recovery carries when the factor exceeds binary64.
        method: LCHS Method that selected these products.
        select_data: Frozen product-formula SELECT data, or None.
        select_records: Per-slot product-formula records of a constant-source
            layout, in slot order.
        source_layout: SourceBranchLayout for a constant source, or None.
        qsp_plan: Frozen joint-generator QSP selection, or None.
        initial_mps: Selected MPS decomposition of the initial direction, or None.
        coefficient_mps: Selected MPS decomposition of the coefficient PREP, or None.
        selected_select: Resolved SELECT implementation or identity route.
        host_actions: Selected classical Pauli actions for host execution, or None:
            a mapping with ``nodes`` (the stored node table,
            time_independent_terms.LCHSProductFormulaNodes) and ``applications``
            (per application its start, weight, source role and per-position
            step record and route).
    """

    initial: object
    source: object
    initial_direction: object
    source_direction: object
    quadrature: object
    coefficient_plan: object
    method: object
    select_data: object = None
    select_records: tuple = ()
    source_layout: object = None
    qsp_plan: object = None
    initial_mps: object = None
    coefficient_mps: object = None
    selected_select: str = "branch_controlled"
    host_actions: object = None


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


def select_dense(method, problem):
    """Select the homogeneous coefficient grid for dense A on the padded encoding.

    Only supplied dense A and explicit vectors are accepted, so no compact
    input is densified implicitly. Missing vector data is rejected before the
    eigensolve and quadrature selection, and the byte cap is checked before
    the Cartesian arrays are formed. A non-power-of-two dimension is encoded
    as A ⊕ 0 with zero-padded inputs. The analytic zero endpoint of the dummy
    block joins the PSD decision without a second eigensolve.
    """
    from .time_independent_terms import generate_lchs_quadrature
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
    # generate_lchs_quadrature's allocation schedule, and eight complex128
    # D-vectors for the padded physical and direction vectors of u0 and b and
    # their copies. The O(D) diagonal and spectral temporaries run before
    # those vectors are formed. The queried eigensolver workspace is not
    # charged (a declared known-array workspace, not a complete cap).
    _check_bytes(64*dimension**2 + 128*dimension, method.max_bytes, "LCHS encoded inputs and Cartesian arrays")
    original = problem.A.dense_array()
    quadrature = generate_lchs_quadrature(matrix=original, final_time=problem.elapsed_time,
        method=method, padded_dimension=dimension, max_bytes=method.max_bytes,
        max_spectral_work=method.max_spectral_work,max_quadrature_work=method.max_quadrature_work)
    # L and H are private arrays already adopted read-only by
    # _prepare_decomposition, so they are kept without another copy.
    quadrature = replace(quadrature,
        k_nodes=_freeze_array(quadrature.k_nodes), coefficients=_freeze_array(quadrature.coefficients),
        conversion=MappingProxyType(dict(quadrature.conversion)))
    initial = _pad_vector(problem.initial_state.physical_vector(), dimension)
    initial_direction = _pad_vector(problem.initial_state._direction, dimension)
    source = source_direction = None
    if problem.source is not None:
        source = _pad_vector(problem.source.physical_vector(), dimension)
        source_direction = _pad_vector(problem.source._direction, dimension)
    coefficients = _homogeneous_coefficient_plan(quadrature, problem.elapsed_time)
    return LCHSData(initial, source, initial_direction, source_direction,
        quadrature, coefficients, method)


def plan_lchs(method, problem, *, output, execution, shots, rng):
    """Route one LinearDynamics to its single LCHS planning owner.

    Zero elapsed time, or a zero initial vector with no nonzero source, uses
    the initial-condition identity with no evolution. PeriodicStencil A uses the compact quantum
    Strang route. Dense A selects its numerical products once and hands them
    to the classical or quantum owner. Input scales, observable coordinates
    and supported PREP choices are checked before any spectral or quadrature
    work, so an unsupported request never spends that work.
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
        # Same dense-observable limit as quantum.observable_terms.
        if execution=='quantum' and not (observable.reference.representation=='pauli' or
                observable.reference.representation=='dense' and problem.dimension<=AUTO_DENSE_PAULI_DIMENSION):
            raise ApplicabilityError("LCHS quantum observable needs finite real Pauli access or an explicitly dense "
                                     f"dimension <={AUTO_DENSE_PAULI_DIMENSION}")
    # The synthesis tolerance inverts the weighted Strang bound of the
    # periodic route. Fixed-step dense routes publish no synthesis bound at
    # planning, so there is nothing to invert.
    if method.trotter_synthesis_tolerance is not None and problem.A.reference.representation != "periodic_stencil":
        raise ApplicabilityError("trotter_synthesis_tolerance selects steps only for a PeriodicStencil A. "
                                 "Give trotter_steps, or use trotter_error_budgeted for per-node step selection")
    zero = (problem.initial_state.preparation.physical_scale.mantissa == 0 and
            (problem.elapsed_time==0 or problem.source is None or problem.source.preparation.physical_scale.mantissa == 0))
    if problem.elapsed_time == 0 or zero:
        from .host import plan_initial
        return plan_initial(method, problem, output=output, execution=execution, shots=shots, rng=rng, zero=zero)
    if problem.A.reference.representation == "periodic_stencil":
        if execution != "quantum":
            raise ApplicabilityError("PeriodicStencil LCHS currently has a quantum Strang realization; no implicit dense host model")
        from .periodic import plan_periodic
        return plan_periodic(method, problem, output=output, shots=shots, rng=rng)
    if execution == "quantum" and problem.source is not None and (
        method.initial_state_preparation != "direct" or method.lcu_state_preparation != "direct"):
        raise ValueError("constant-source LCHS currently has direct PREP only")
    data = select_dense(method, problem)
    if execution == "classical":
        from .host import plan_classical
        return plan_classical(method, problem, data, output=output, rng=rng)
    from .quantum import plan_quantum
    return plan_quantum(method, problem, data, output=output, shots=shots, rng=rng)
