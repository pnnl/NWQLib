"""The QFD reconstruction of a Plan, the Result of its projected solve and the record of a chemistry generator."""

from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_validator

from nwqlib.algorithms.gcim.fixed_basis import (
    FixedGCIMReconstruction,
    GroupMoment,
    PairEstimate,
    PauliArrays,
    ProjectedPencil,
    SampledPairVariance,
    ScalarEstimate,
    ShotAllocation,
    _check_shot_allocation,
)
from nwqlib.core.analysis import Result
from nwqlib.core.planning import Plan
from nwqlib.core.records import ContentID, FrozenArray, Nonnegative, Real, Record, Text
from nwqlib.operators.access import Count


class QFDReconstruction(Record):
    """The fixed step V and the Pauli tables that turn the observations of a QFD Plan back into ``H`` and ``S``.

    `plan.reconstruction` of a [`QFD`][nwqlib.algorithms.qfd.method.QFD]
    Plan holds it. V applies, for each of its r substeps, the rotations
    ``R_P(theta) = exp(-i theta P/2)`` of the rows `step_rows` of
    `evolution_rows` with the angles `step_angles`, in this order. Basis
    state j is V repeated ``powers[j]`` times, or its adjoint repeated
    ``-powers[j]`` times, applied to the reference. With
    `QFD.evolution="pauli_terms"` the rotations come from the Pauli table
    of the target in `pencil`. With `"fermionic_blocks"` they come from the
    generator B of `QFD.evolution_terms`, whose table is `evolution`, and
    `step_blocks` marks its complete blocks. H and S always belong to the
    target in `pencil`.

    Attributes:
        schema_version: Record layout version.
        pencil: The Pauli table of the target A, basis size, groups and
            measurement choices shared with FixedGCIM
            (`FixedGCIMReconstruction`). Its `acquisition.pairs` records the
            construction of every off-diagonal pair.
        substep: The emitted binary64 substep ``time_step/steps_per_time_step``
            in inverse energy units.
        step_rows: int64 array of the positions in `evolution_rows` of the
            factors of one substep, in execution order. Empty when the
            generator of V is a multiple of the identity, which makes V the
            identity.
        step_angles: float64 array of the emitted angle theta of each factor.
        reference_index: Basis index of the power 0, the reference state.
        evolution: None for `"pauli_terms"`. For `"fermionic_blocks"`, the
            Pauli table of B (`PauliArrays`): one row for each distinct word
            that a factor of V uses, with the coefficient of B summed over
            its blocks and rounded once, and the identity row, whose
            coefficient is c_B. A word whose coefficients cancel keeps its
            row with coefficient 0. The coefficients describe B. V uses only
            the words and `step_angles`.
        step_blocks: None for `"pauli_terms"`. For `"fermionic_blocks"`, the
            int64 positions in `step_rows` where each complete block of one
            substep starts, strictly increasing from 0, and empty when V has
            no factors. A block runs to the next start or to the end of the
            substep.
    """

    schema_version: Literal[1] = 1
    pencil: FixedGCIMReconstruction
    substep: Annotated[Real, Field(gt=0)]
    step_rows: FrozenArray
    step_angles: FrozenArray
    reference_index: Count
    evolution: PauliArrays | None = None
    step_blocks: FrozenArray | None = None

    @property
    def evolution_rows(self):
        """The nonidentity ``(label, coefficient)`` rows that `step_rows` index: those of `pencil`, or of `evolution` when it is set."""
        if self.evolution is None:
            return self.pencil.nonidentity_terms
        return tuple(row for row in self.evolution.rows() if row[0].count("I") < len(row[0]))

    @model_validator(mode="after")
    def _factors(self):
        """Require one angle per factor, factors that name rows of the generator of V, and block starts within one substep."""
        rows, angles = self.step_rows.array, self.step_angles.array
        if (rows.dtype != np.int64 or angles.dtype != np.float64 or rows.ndim != 1 or angles.shape != rows.shape
                or not np.isfinite(angles).all()):
            raise ValueError("QFD step needs int64 rows and finite float64 angles of one shape")
        count = len(self.evolution_rows)
        if rows.size and (rows.min() < 0 or rows.max() >= count) or bool(rows.size) != bool(count):
            raise ValueError("QFD step rows must name the nonidentity rows of the Pauli table of its generator")
        if (self.evolution is None) != (self.step_blocks is None):
            raise ValueError("QFD evolution table and block starts are recorded together")
        if self.step_blocks is not None:
            starts = self.step_blocks.array
            if (starts.dtype != np.int64 or starts.ndim != 1 or bool(starts.size) != bool(rows.size)
                    or starts.size and (starts[0] != 0 or starts[-1] >= rows.size or (np.diff(starts) <= 0).any())):
                raise ValueError("QFD block starts must increase strictly from 0 within one substep")
        if self.reference_index >= self.pencil.basis_size:
            raise ValueError("QFD reference index lies outside its basis")
        return self


class QFDEvolutionDiagnostics(Record):
    """How `chemistry_evolution_terms` derived a fermionic evolution generator B from the integrals of chemistry data.

    [`chemistry_evolution_terms`][nwqlib.algorithms.qfd.chemistry.chemistry_evolution_terms]
    returns it with B. Pass both to `QFD`, as
    `QFD(..., evolution="fermionic_blocks", evolution_terms=terms,
    evolution_diagnostics=diagnostics)`, and the record is saved with the
    Method, the Plan and the Result, where `result.plan.method.evolution_diagnostics`
    reads it. `QFD` requires `evolution_input` to equal the content hash of
    its `evolution_terms`.

    The helper builds the table C_raw of the integrals, with one coefficient
    g_T per normal-ordered ladder monomial T, and returns
    ``B = sum_T b^gamma_T T`` after one Hermitian projection and a cutoff
    gamma. C_raw is the operator of that table, formed from the binary64
    integrals with the factor 1/2 of the two-body terms and the summation
    of equal monomials, and may be non-Hermitian. With q the unrounded
    Hermitian table, b the projected table before the cutoff and b^gamma
    the returned table, summed over every monomial, the two sides of a pair
    counted separately,

        ||B - C_raw|| <= sum_T |b^gamma_T - g_T| <= Delta_sym + Delta_round + Delta_drop,

    since every nonzero normal-ordered monomial has norm at most 1, with
    ``Delta_sym = sum_T |q_T - g_T|``, ``Delta_round = sum_T |b_T - q_T|``
    and ``Delta_drop`` the sum of ``|b_T|`` over the dropped monomials.
    Every bound below is computed in exact rationals from the binary64
    values and rounded up once, and it is None when it overflows binary64.
    These bounds compare B with the table C_raw of the integrals. They do
    not bound the evolution, since C_raw need not be Hermitian, and neither
    they nor the one-body correction measures the accuracy of a molecular
    energy. Upper bounds on the distance between the target A and B are
    stated in the Plan's `assumptions`. The fields below are read-only.

    Attributes:
        schema_version: Record layout version.
        evolution_input: Content hash (`reference.identity`) of the B that
            the helper returned.
        source_name: The `"name"` entry of the chemistry data's `metadata`,
            or None when it has none.
        integral_convention: The `"integral_convention"` entry of that
            `metadata`, or None.
        one_body_shape: Shape of the spatial-orbital one-body array.
        two_body_shape: Shape of the spatial-orbital two-body array.
        transformation: Fixed name of the projection and cutoff.
        domain: Fixed statement of the inputs and mode order the helper accepts.
        evolution_coefficient_cutoff: The cutoff gamma in Hartree, at least 0.
        changed_pairs: Number of pairs of distinct adjoint monomials whose
            projected coefficients differ from those of C_raw.
        dropped_pairs: Number of pairs of distinct adjoint monomials, and of
            self-adjoint monomials, that the cutoff removed.
        symmetrization_correction: Upper bound on ``Delta_sym``.
        rounding_correction: Upper bound on ``Delta_round``.
        dropped_mass: Upper bound on ``Delta_drop``.
        raw_difference: Upper bound on ``sum_T |b^gamma_T - g_T|``, and hence
            on ``||B - C_raw||``.
        max_one_body_asymmetry: Upper bound on ``max_{p<q} |h_pq - h_qp|`` of
            the spatial one-body array h of the data.
        one_body_correction: Upper bound on ``sum_pq |dh_pq|``, with
            ``dh = h_B - h`` and h_B the spatial one-body coefficients of the
            returned B.
        one_body_correction_bound: Upper bound on ``2 sum_pq |dh_pq|``, which
            bounds the norm of the one-body correction ``sum_pq,sigma dh_pq
            a_{p sigma}^dagger a_{q sigma}`` on the Fock space, the factor 2
            for the two spin blocks.
        generator_shift: The identity coefficient c_B of the Jordan-Wigner
            image of B, rounded once. It is not added to any energy.
        spin_statement: How many pairs of adjoint monomials and self-adjoint
            monomials of B conserve the alpha and beta electron numbers, and
            so Sz, under the mode order 2p + sigma, with the particle number
            and Sz of the reference occupations.
        scope: Fixed statement of what the bounds above compare.
    """

    schema_version: Literal[1] = 1
    evolution_input: ContentID
    source_name: Text | None
    integral_convention: Text | None
    one_body_shape: tuple[Count, Count]
    two_body_shape: tuple[Count, Count, Count, Count]
    transformation: Literal["signed midpoint of each pair of adjoint canonical monomials, rounded once, then "
                            "removal of whole pairs at or below the cutoff"] = (
        "signed midpoint of each pair of adjoint canonical monomials, rounded once, then "
        "removal of whole pairs at or below the cutoff")
    domain: Literal["real finite spatial-orbital integrals h[p,q] and v[p,q,r,s] = (ps|qr), spin modes 2p + sigma "
                    "with alpha sigma = 0 and beta sigma = 1"] = (
        "real finite spatial-orbital integrals h[p,q] and v[p,q,r,s] = (ps|qr), spin modes 2p + sigma "
        "with alpha sigma = 0 and beta sigma = 1")
    evolution_coefficient_cutoff: Nonnegative
    changed_pairs: Count
    dropped_pairs: Count
    symmetrization_correction: Nonnegative | None
    rounding_correction: Nonnegative | None
    dropped_mass: Nonnegative | None
    raw_difference: Nonnegative | None
    max_one_body_asymmetry: Nonnegative | None
    one_body_correction: Nonnegative | None
    one_body_correction_bound: Nonnegative | None
    generator_shift: Real
    spin_statement: Text
    scope: Literal["the bounds compare B with the table C_raw of the integrals, which need not be Hermitian, so "
                   "they do not bound the evolution, and they do not measure the accuracy of a molecular energy"] = (
        "the bounds compare B with the table C_raw of the integrals, which need not be Hermitian, so "
        "they do not bound the evolution, and they do not measure the accuracy of a molecular energy")


class QFDResult(Result):
    """Lowest Ritz energy from a `QFD` run, with the reference energy and the projected matrices.

    [`solve`][nwqlib.scientist.solve] returns it for a
    [`QFD`][nwqlib.algorithms.qfd.method.QFD] method, and `load_result`
    reopens a saved one. `eigenvalue` is the lowest kept Ritz value of the
    projected pencil `H f = E S f` of the basis `V**p_j |psi>`, in the unit
    of the `Eigenproblem`. `reference_energy` is `H[i0, i0] = <psi|A|psi>`
    of the reference, at the basis index i0 of the power 0, which need not
    be the first index. `lowering` is `reference_energy - eigenvalue`,
    negative values included. With exact, consistent H and S the Ritz value
    is at most the reference energy when no overlap direction is cut, and
    at least the smallest eigenvalue of A, but sampled matrices carry
    neither guarantee. The value is a projected estimate in the chosen
    trial space, not a certified ground energy, and a lowering is not a
    recovered fraction of a correlation energy. `pencil` holds H, S and the
    solve diagnostics: kept rank, the complete overlap spectrum and the
    condition number of the kept directions. `print(result)` shows the
    values, the basis schedule and the scope, and
    `result.analyze(overlap_cutoff=...)` solves the same matrix elements at
    another cutoff without new measurement. The fields of
    [`Result`][nwqlib.core.analysis.Result] are present too.

    Attributes:
        eigenvalue: Lowest kept Ritz value in the problem's energy unit, or
            None when data are missing or the solver rejected the pencil.
        reference_energy: `H[i0, i0]`, the energy of the reference state in
            the same unit, from the same pencil, or None without a pencil.
            For an operator that is a multiple `c I` of the identity it is
            c, as is `eigenvalue`.
        lowering: `reference_energy - eigenvalue`, or None when either is
            None. It is not clipped at zero.
        missing: Names of planned measurements with no returned data. A
            nonempty tuple means no pencil was solved.
        pairs: Exact pair measurements: pooled `H0_ij` and `S_ij` with their
            reductions and entry error bounds. Empty for sampled Plans.
        estimates: For a sampled Plan, the pooled weighted group means
            ([`ScalarEstimate`][nwqlib.algorithms.gcim.fixed_basis.ScalarEstimate]).
        group_moments: For a sampled Plan, the moments of every count
            measurement, in measurement order.
        sampled_pair_variances: For a complete sampled pencil, the pooled
            sample-mean variances and overlap–H0 covariance of each pair.
        pencil: The [`ProjectedPencil`][nwqlib.algorithms.gcim.fixed_basis.ProjectedPencil]
            in basis order, or None for partial data or when the operator is
            a multiple of the identity, which needs no H or S. Under
            `PencilSampling` it is formed from the main-stage measurements
            only, and so is `reference_energy`.
        analysis_cutoff: Overlap-eigenvalue cutoff used by this analysis.
        target_identification: Fixed statement of what the value does not establish.
        shot_allocation: Under `PencilSampling`, the pilot decision
            ([`ShotAllocation`][nwqlib.algorithms.gcim.fixed_basis.ShotAllocation])
            as it was decided, or None before it. Then `estimates`,
            `group_moments`, `sampled_pair_variances`, `pencil`, `missing` and
            `contribution_ids` concern the main stage only, and a Result from
            before the decision lists every setting in `missing`. The pilot
            measurements stay in the Run's observations. None without
            `PencilSampling`.
    """

    schema_version: Literal[1] = 1
    eigenvalue: Real | None
    reference_energy: Real | None
    lowering: Real | None
    missing: tuple[Text, ...]
    estimates: tuple[ScalarEstimate, ...] = ()
    pairs: tuple[PairEstimate, ...] = ()
    group_moments: tuple[GroupMoment, ...] = ()
    sampled_pair_variances: tuple[SampledPairVariance, ...] = ()
    pencil: ProjectedPencil | None
    analysis_cutoff: Annotated[Real, Field(gt=0)]
    target_identification: Text = (
        "projected estimate in the fixed-V trial space; smallest full-space eigenvalue not established"
    )
    shot_allocation: ShotAllocation | None = None

    def _summary_lines(self):
        unit = self._unit_text()
        lines = [
            f"Ritz eigenvalue: {self._scalar_text(self.eigenvalue)}{unit}",
            f"Reference energy: {self._scalar_text(self.reference_energy)}{unit}; "
            f"lowering: {self._scalar_text(self.lowering)}{unit}",
        ]
        if self._plan is not None:
            method = self._plan.method
            lines.append(
                f"Basis: powers {method.powers} of one fixed V, order {method.formula_order} with "
                f"{method.steps_per_time_step} substeps of time step {method.time_step!r}")
        lines.append(self.target_identification)
        if self.missing:
            lines.append(f"Partial data: {len(self.missing)} missing transition measurements")
        elif self.pencil is None:
            lines.append("No H or S formed: the operator is a multiple of the identity")
        elif self.pencil.failure_reason:
            lines.append("Analysis unavailable: " + self.pencil.failure_reason)
        if self.pencil is not None:
            lines.extend(self.pencil._sampled_summary())
        if self.shot_allocation is not None:
            lines.append(self.shot_allocation._summary(self.pencil))
        return lines

    def validate_plan(self, plan: Plan) -> None:
        """Reject this Result unless its values and pencil follow from ``plan``.

        The checks, in order: the same Plan and construction, no value from
        partial observations, the settings of a sampled Plan with
        nonidentity terms equal to those of its recorded group partition
        and resolved quadratures, group statistics only on such a Plan and
        exact pair reductions only without shots, the identity coefficient
        as eigenvalue and reference energy with zero lowering when the
        operator is a multiple of the identity, a pencil of the Plan's
        basis size and analysis cutoff, an eigenvalue equal to the pencil's
        lowest Ritz value, a reference energy equal to its entry
        ``H[i0, i0]``, the lowering of these two values, the overlap filter
        of the Plan's measurements and sampled enclosure fields recomputed
        from the pencil. A `shot_allocation` is accepted exactly on a
        `PencilSampling` Plan, naming its sampling and one main count per
        experiment, and is required once any setting has main-stage data.
        This check follows the partial-data check. No measurement or
        eigensolve is repeated.

        Args:
            plan: The Plan this Result claims to analyze.

        Raises:
            ValueError: If any check fails.
        """
        self._validate_common_plan(plan, Plan)
        if self.construction_id != plan.construction.content_id:
            raise ValueError("QFD result differs from its actual selected construction")
        if self.missing and (self.eigenvalue is not None or self.pencil is not None):
            raise ValueError("partial observations cannot produce a projected estimate")
        _check_shot_allocation(self.shot_allocation, self.missing, plan, "QFD")
        rec = plan.reconstruction.pencil
        sampled = plan.shots is not None
        grouped = sampled and bool(rec.nonidentity_coefficients)
        if grouped and (not rec.groups or tuple(e.name for e in plan.experiments) != tuple(
                s[0] for s in rec.settings())):
            raise ValueError("QFD sampled Plan differs from its declared group settings")
        if not grouped and (rec.groups or self.group_moments or self.sampled_pair_variances or self.estimates):
            raise ValueError("QFD group statistics belong only to a sampled Plan with nonidentity terms")
        if sampled and self.pairs:
            raise ValueError("a sampled QFD Result holds no exact pair reductions")
        if not rec.nonidentity_coefficients:
            shift = rec.identity_shift
            if (self.eigenvalue, self.reference_energy, self.lowering, self.pencil) != (shift, shift, 0.0, None):
                raise ValueError("QFD identity values differ from the exact constant")
            return
        if self.pencil is None:
            if (self.eigenvalue, self.reference_energy, self.lowering) != (None, None, None):
                raise ValueError("QFD values need their pencil")
            return
        p = self.pencil
        if len(p.hamiltonian) != rec.basis_size or len(p.overlap) != rec.basis_size or (
                p.overlap_cutoff != self.analysis_cutoff):
            raise ValueError("QFD pencil differs from its basis or analysis settings")
        if self.eigenvalue != (p.eigenvalues[0] if p.eigenvalues else None):
            raise ValueError("QFD eigenvalue differs from its projected spectrum")
        i0 = plan.reconstruction.reference_index
        if self.reference_energy != p.hamiltonian[i0][i0].real:
            raise ValueError("QFD reference energy differs from the reference entry of its pencil")
        if self.lowering != (None if self.eigenvalue is None else self.reference_energy - self.eigenvalue):
            raise ValueError("QFD lowering differs from its reference energy and eigenvalue")
        p._validate_sampled_enclosure(rec, sampled=sampled)
