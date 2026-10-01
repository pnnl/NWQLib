"""LCHS choices for the original equation du/dt=-A u+b.

Input and coordinates belong to LinearDynamics. Numerical selection belongs to
this Method, acquisition to Run, and reconstruction to the selected readout.
"""

import json

from nwqlib._limits import DEFAULT_MAX_BYTES

from typing import Annotated, Any, ClassVar, Literal

from pydantic import Field, PrivateAttr, StrictBool, field_serializer, field_validator, model_validator

from nwqlib._quantum_readout import PROJECTED_MOMENTS, projected_requirements
from nwqlib.algorithms.protocol import Method, ApplicabilityError
from nwqlib.core.records import PositiveInt, Real
from nwqlib.problems.records import LinearDynamics, Solution, StateVector, Samples
from .primary_records import DESCRIPTOR, LCHSAnalysis
from .provider_config import ProviderConfig
from .providers import resolve_lchs_provider_requests
from .solution_error_budget import DEFAULT_PSD_TOLERANCE


def _projected_allowance(method, plan, point, *, observation, width, run,
                         limit, parameter, planning_work):
    """Check local bytes and cumulative projected work before native acquisition.

    planning_work is the charge assigned to the selected cap at planning.
    Completed projected points of other experiments spend that same cap.
    The current experiment is rechecked without a second charge when a
    preparation is reused. This matches LCHS's one exact scalar experiment.
    """
    def points(readout):
        return (item for item in readout.positions
                if item.kind == "reduction" and item.reducer == PROJECTED_MOMENTS)

    requested = 0
    for item in points(observation):
        size, work = projected_requirements(json.loads(item.parameters), width)
        if size > method.max_bytes:
            raise ValueError(
                f"LCHS projected reduction needs {size} bytes, max_bytes={method.max_bytes}. "
                f"Raise LCHS(max_bytes=...) to at least {size} "
                f"(increase {size - method.max_bytes}). Refused before acquisition.")
        requested += work
    spent = 0
    for chunk in run.observations.chunks:
        if chunk.point is None or chunk.experiment == point.experiment:
            continue
        for item in points(chunk.readout()):
            parameters = json.loads(item.parameters)
            old_width = sum(len(parameters[key]) for key in ("coordinates", "success", "conditions"))
            spent += projected_requirements(parameters, old_width)[1]
    required = planning_work + spent + requested
    if required > limit:
        raise ValueError(
            f"LCHS projected reduction needs {requested} work units, with {planning_work} "
            f"planning units and {spent} completed reduction units. {parameter}={limit}. "
            f"Raise {parameter} to at least {required} (increase {required - limit}). "
            "Refused before acquisition.")
    return int(limit - planning_work - spent)


class LCHS(Method):
    """Select a finite LCHS approximation; tolerance is not total output error.

    Default dense SELECT uses classically computed branch exponentials. The
    near-optimal kernel uses beta=.75 unless configured otherwise. Quadrature,
    truncation, evolution and state-preparation errors have distinct scopes.

    Attributes:
        approximation_tolerance: Combined tail and quadrature allowance for the default provider pair, in (0,1). QSP synthesis or each automatic Trotter branch separately receives 0.1 times this value. It is not a total physical-output tolerance.
        lchs_kernel: ProviderConfig selecting the integral kernel and explicit scalar parameters.
        k_quadrature: ProviderConfig selecting nodes/weights; defaults to composite Gauss panels.
        hamiltonian_evolution_backend: ``dense_exact``, fixed ``trotter``, ``trotter_error_budgeted`` or ``qsp_block_encoding`` branch evolution.
        lcu_select_implementation: ``auto``, ``multiplexor``, ``structured`` or ``branch_controlled`` SELECT wiring.
        dense_control_route: ``gatewise``, ``whole_matrix`` or ``auto`` route by which a construction controls a dense unitary on two or more qubits (``_dense_synthesis.select_dense_control_route``): a ``dense_exact`` branch on its address bits and a ``dense_dilation`` QSP child on the joint generator's combine qubit. ``gatewise`` synthesizes the unitary and lets Qiskit control each synthesized gate, ``whole_matrix`` synthesizes the controlled matrix, and ``auto`` takes the whole-matrix route for one control and the gate-wise route for more. On the whole-matrix route a constant-source branch is one unitary, its evolution times the matrix of its input preparation. The QSP pass that holds the generator is controlled gate-wise on every route.
        trotter_steps: Positive fixed step count of the ``trotter`` backend, or None when trotter_synthesis_tolerance selects it. ``trotter_error_budgeted`` selects its own counts and requires the default 1. ``dense_exact`` and ``qsp_block_encoding`` apply no product formula, so a value other than 1 selects the same approximation and circuit as the default, and planning warns about it.
        trotter_synthesis_tolerance: Positive allowance for the Strang synthesis bound of the compact periodic route, or None. The route selects the smallest step count whose weighted bound (``periodic.select_periodic_parameters``) is at most this value, in the same per-unit-input operator-norm frame as approximation_tolerance. It requires the fixed ``trotter`` backend and trotter_steps=None, and other routes refuse it.
        trotter_order: Lie order 1 or symmetric Suzuki order 2 of the fixed or budgeted product formula. dense_exact and qsp_block_encoding accept only 2.
        max_trotter_steps: Cap on automatically selected product-formula steps.
        duhamel_nodes: Positive constant-source time-quadrature node count; not automatically reduced to fit a cap.
        make_l_psd: Allow the documented Hermitian-part shift and physical recovery; False requires the admitted PSD premise.
        psd_tolerance: Numerical allowance in Hermitian-part PSD admission, not permission to ignore a negative mode.
        initial_state_preparation: ``direct`` or explicitly approximate ``mps_circuit`` initial-state PREP.
        initial_state_mps_max_bond_dim: Optional retained bond-dimension cap for initial-state compression.
        initial_state_mps_threshold: Positive singular-value truncation threshold for initial-state compression.
        initial_state_mps_num_layers: Positive number of initial-state MPS circuit layers.
        lcu_state_preparation: ``direct`` or explicitly approximate ``mps_circuit`` coefficient PREP.
        lcu_mps_max_bond_dim: Optional coefficient-state bond-dimension cap.
        lcu_mps_threshold: Positive singular-value truncation threshold for coefficient compression.
        mps_num_layers: Positive number of coefficient-state MPS circuit layers.
        max_dense_select_slots: Maximum padded slots in the dense SELECT validation construction.
            A larger selection is refused at planning, and the grid, Duhamel nodes
            and tolerance are never changed to fit.
        max_bytes: Cap on known input and numerical workspace bytes.
        max_svd_work: Cap on counted state/coefficient compression work, including the layered
            construction of an ``mps_circuit`` PREP (``mps.layered_construction_size``).
        max_spectral_work: Cap on counted spectral analysis work.
        max_quadrature_work: Cap on selected quadrature node/weight work. For the composite-Gauss
            provider it covers the completed panel search plus the selected rule's construction,
            not every scalar probe.
        max_select_work: Cap on counted SELECT construction work.
        max_readout_work: Inclusive cap on sampled-readout grouping comparisons and
            registered exact projected-reduction input visits. The default is
            2_000_000_000. Planning checks the sampled-readout grouping comparisons
            against this cap and records them. Preparation checks each exact
            projected reduction against the remaining allowance before
            acquisition, and completed projected reductions spend it.
        max_qsp_degree: Maximum polynomial degree for selected QSP branch evolution.
        max_qsp_evaluations: Maximum phase-solver residual evaluations.
        max_admission_steps: Admission ceiling of the Plan's Programs, their ``AdmissionLimits.max_steps``: the kept field slots and the structural and lifecycle work units of one admission check. The resource fold of a Program may use up to 24 times this value. The default, 1,000,000, is the QLS default, ten times the shared ``AdmissionLimits`` default. Raising it admits a larger metadata check and fold. It changes no selected quantum work.
    """

    schema_version: Literal[6] = 6
    approximation_tolerance: Annotated[Real, Field(gt=0, lt=1)] = .01
    lchs_kernel: Any = Field(default_factory=lambda: ProviderConfig(implementation="near_optimal_eq7"))
    k_quadrature: Any = Field(default_factory=lambda: ProviderConfig(implementation="composite_gauss"))
    hamiltonian_evolution_backend: Literal[
        "dense_exact", "trotter", "trotter_error_budgeted", "qsp_block_encoding"
    ] = "dense_exact"
    lcu_select_implementation: Literal["auto", "multiplexor", "structured", "branch_controlled"] = "auto"
    dense_control_route: Literal["gatewise", "whole_matrix", "auto"] = "auto"
    trotter_steps: PositiveInt | None = 1
    trotter_synthesis_tolerance: Annotated[Real, Field(gt=0)] | None = None
    trotter_order: Literal[1, 2] = 2
    max_trotter_steps: PositiveInt = 100_000
    duhamel_nodes: PositiveInt = 8
    make_l_psd: StrictBool = True
    psd_tolerance: Annotated[Real, Field(gt=0)] = DEFAULT_PSD_TOLERANCE
    initial_state_preparation: Literal["direct", "mps_circuit"] = "direct"
    initial_state_mps_max_bond_dim: PositiveInt | None = None
    initial_state_mps_threshold: Annotated[Real, Field(gt=0)] = 1e-14
    initial_state_mps_num_layers: PositiveInt = 2
    lcu_state_preparation: Literal["direct", "mps_circuit"] = "direct"
    lcu_mps_max_bond_dim: PositiveInt | None = None
    lcu_mps_threshold: Annotated[Real, Field(gt=0)] = 1e-14
    mps_num_layers: PositiveInt = 2
    max_dense_select_slots: PositiveInt = 256
    max_bytes: PositiveInt = DEFAULT_MAX_BYTES
    max_svd_work: PositiveInt = 100_000_000
    max_spectral_work: PositiveInt = 100_000_000
    max_quadrature_work: PositiveInt = 100_000_000
    max_select_work: PositiveInt = 100_000_000
    max_readout_work: PositiveInt = 2_000_000_000
    max_qsp_degree: PositiveInt = 256
    max_qsp_evaluations: PositiveInt = 20_000
    max_admission_steps: PositiveInt = 1_000_000
    _ignored_trotter_steps: int | None = PrivateAttr(default=None)

    descriptor: ClassVar = DESCRIPTOR
    result_type: ClassVar = LCHSAnalysis

    @field_validator("lchs_kernel", "k_quadrature", mode="before")
    @classmethod
    def _provider(cls, value):
        """Accept a registry name, a {"implementation", "parameters"} dict or a ProviderConfig."""
        if isinstance(value, str):
            return ProviderConfig(implementation=value)
        if isinstance(value, dict):
            return ProviderConfig(**value)
        if not isinstance(value, ProviderConfig):
            raise TypeError("LCHS provider choice must be a name or ProviderConfig")
        return value

    @field_serializer("lchs_kernel", "k_quadrature")
    def _provider_record(self, value):
        """Serialize a provider request as its explicit implementation and parameters."""
        return value.to_dict()

    @field_validator("trotter_order", mode="before")
    @classmethod
    def _order(cls, value):
        """Require an integer order before the Literal check, so a float 2.0 or a bool fails."""
        from nwqlib._validation import integer
        return integer(value, "trotter_order", 1)

    @model_validator(mode="after")
    def _choices(self):
        """Reject configurations whose choices conflict, before any problem is seen.

        The kernel-quadrature pair must be registered and allowed, and
        an active trotter_steps may not exceed max_trotter_steps. Both product-formula
        backends accept Lie order 1 and Suzuki order 2. ``trotter_error_budgeted``
        selects each node's step count from the bound of the chosen order,
        CSTWZ (Phys. Rev. X 11, 011020 (2021), doi:10.1103/PhysRevX.11.011020)
        Prop. 9, Eq. (120) for order 1 or Prop. 10, Eq. (121) for order 2.
        dense_exact and qsp_block_encoding apply no product formula, so they
        accept only the default order 2.

        Exactly one of trotter_steps and trotter_synthesis_tolerance is set,
        so a fixed step count and the count that a synthesis tolerance
        selects cannot disagree. The tolerance selects the step count of the
        fixed ``trotter`` backend only. ``trotter_error_budgeted`` selects
        every node's step count from its allowance and never reads
        trotter_steps, so any value other than the default 1 is refused
        rather than ignored. An explicit 1 equals the default, which the
        stored record cannot tell apart, so it stays admitted. ``dense_exact``
        and ``qsp_block_encoding`` also never read trotter_steps. They admit
        a value other than 1, normalize it to 1 before content identity is
        computed, and retain the supplied value only for ``plan``'s warning.
        """
        resolve_lchs_provider_requests(self.lchs_kernel, self.k_quadrature)
        if (self.trotter_steps is None) == (self.trotter_synthesis_tolerance is None):
            raise ValueError(
                "set exactly one of trotter_steps and trotter_synthesis_tolerance. "
                "The tolerance selects the periodic Strang step count, so pass trotter_steps=None with it"
            )
        if self.trotter_synthesis_tolerance is not None and self.hamiltonian_evolution_backend != "trotter":
            raise ValueError("trotter_synthesis_tolerance selects the step count of hamiltonian_evolution_backend='trotter'")
        if self.hamiltonian_evolution_backend == "trotter_error_budgeted" and self.trotter_steps != 1:
            raise ValueError(
                "hamiltonian_evolution_backend='trotter_error_budgeted' selects its own step count for each node "
                "from its allowance and does not use trotter_steps. Leave trotter_steps at its default, "
                "or use hamiltonian_evolution_backend='trotter' for a fixed step count"
            )
        if self.hamiltonian_evolution_backend in ("dense_exact", "qsp_block_encoding") and self.trotter_steps != 1:
            self._ignored_trotter_steps = self.trotter_steps
            object.__setattr__(self, "trotter_steps", 1)
        if self.trotter_steps is not None and self.trotter_steps > self.max_trotter_steps:
            raise ValueError("trotter_steps exceeds max_trotter_steps")
        if self.trotter_order != 2 and self.hamiltonian_evolution_backend not in ("trotter", "trotter_error_budgeted"):
            raise ValueError(
                "trotter_order=1 requires hamiltonian_evolution_backend 'trotter' or "
                "'trotter_error_budgeted', because dense_exact and qsp_block_encoding use no product formula"
            )
        return self

    def plan(self, problem, *, output, execution, shots, rng):
        """Check output, execution and shots against LCHS, then select the Plan once.

        A complex vector needs amplitude readout, so vector outputs reject
        shots. Samples need quantum execution with shots. QSP evolution has no
        classical acquisition. All numerical selection happens in plan_lchs.

        ``dense_exact`` and ``qsp_block_encoding`` apply no product formula
        and record 0 steps. An ignored trotter_steps is canonicalized to 1
        during Method validation, leaving the selected content identity
        unchanged. Planning warns using the privately retained supplied value.
        The warning skips frames inside the nwqlib package,
        so it names the caller's line whether the call came through
        ``plan``, ``solve`` or ``compare``. Constructing, copying or loading
        a Method or Plan does not warn.
        """
        if not isinstance(problem, LinearDynamics):
            raise ApplicabilityError("LCHS requires LinearDynamics(A, initial_state, time, source)")
        if output.kind not in self.descriptor.output_families:
            raise ApplicabilityError("LCHS supports a physical solution, state, norm, observable or samples")
        if isinstance(output, (Solution, StateVector)) and shots is not None:
            raise ApplicabilityError("counts do not recover a complex vector; omit shots or select a scalar/Samples output")
        if isinstance(output, Samples) and (shots is None or execution != "quantum"):
            raise ApplicabilityError("Samples requires quantum execution and finite shots")
        if execution == "classical" and shots is not None:
            raise ApplicabilityError("classical LCHS acquires deterministic numerical data; shots belong to quantum readout")
        if execution == "classical" and self.hamiltonian_evolution_backend == "qsp_block_encoding":
            raise ApplicabilityError("QSP LCHS has no selected classical acquisition; use quantum execution")
        if self._ignored_trotter_steps is not None:
            import os
            import warnings
            import nwqlib
            warnings.warn(
                f"trotter_steps={self._ignored_trotter_steps} has no effect with hamiltonian_evolution_backend="
                f"'{self.hamiltonian_evolution_backend}', which applies no product formula. trotter_steps "
                "applies only to the 'trotter' backend, so this Plan selects the same approximation and "
                "circuit as the default trotter_steps=1. To use "
                f"{self._ignored_trotter_steps} steps, set hamiltonian_evolution_backend='trotter' and "
                f"trotter_steps={self._ignored_trotter_steps} together. Revising only the backend keeps "
                "trotter_steps=1",
                UserWarning, skip_file_prefixes=(os.path.dirname(nwqlib.__file__) + os.sep,),
            )
        from .selection import plan_lchs
        return plan_lchs(self, problem, output=output, execution=execution, shots=shots, rng=rng)

    def analyze(self, plan, data, *, settings):
        """Reconstruct the requested output from the acquired data (analysis.analyze).

        LCHS fixes its recovery scale and coordinates in the Plan, so there
        are no reconstruction settings to choose after acquisition.
        """
        if settings:
            raise ValueError("LCHS has no post-hoc reconstruction settings")
        from .analysis import analyze
        return analyze(plan, data)

    def reduction_allowance(self, plan, point, *, observation, width, run):
        """Admit projected bytes and work left under max_readout_work."""
        return _projected_allowance(self, plan, point, observation=observation,
            width=width, run=run, limit=self.max_readout_work,
            parameter="LCHS.max_readout_work",
            planning_work=plan.reconstruction.grouping_comparisons)

    def save_archive(self, plan, files):
        """Save the Plan's selected numerical data and block bindings (archive.save)."""
        from .archive import save
        return save(self, plan, files)

    @classmethod
    def load_archive(cls, data, files):
        """Restore a saved Plan and check its selected-data identities (archive.load)."""
        from .archive import load
        return load(data, files)

    def verify(self, plan, result, *, checks):
        """Run one explicit LCHSVerification reference or LCHSRefinement request.

        Both are extra work requested by the caller. Neither is part of plan,
        solve or analysis.
        """
        if self != plan.method:
            raise ValueError("LCHS verification requires the Plan's original configured Method")
        from .verification import LCHSVerification, verify
        from .refinement import LCHSRefinement, refine
        if type(checks) is LCHSVerification:
            return verify(plan, result, checks=checks)
        if type(checks) is LCHSRefinement:
            return refine(plan, result, checks=checks)
        raise TypeError("checks requires one explicit LCHSVerification or LCHSRefinement")
