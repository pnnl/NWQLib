"""Public selection, saved PREP and caller-owned partial execution relations."""

from dataclasses import replace
from itertools import combinations
import re
from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp

import nwqlib
from nwqlib.algorithms import ADAPT, ExpectationMethod, FixedGCIM, Lanczos, QCELS, QFD, RFE
from nwqlib.algorithms.gcim import BoundedRitzGrowth
from nwqlib.algorithms.lchs import LCHS, PTSC, Richardson, SampledPairs, Trotter, TrotterErrorBudgeted
from nwqlib.algorithms.qhd import QHD
from nwqlib.algorithms.qls import QLS
from nwqlib.execution import ObservationView
from nwqlib.operators import PeriodicStencil, ingest_band, ingest_cartesian, ingest_pauli
from nwqlib.problems import NormalizedExpectation, NormSquared, QuadraticForm, Samples, StateVector, ingest_occupation
from test_run_lifecycle import InjectedBackend, clear_backend as clear_backend


def forbidden(*args, **kwargs):
    raise AssertionError(
        "existing selected data triggered planning, normalization or native replay"
    )


def partial_result(plan, data):
    """Use the actual Method producer and the common attachment boundary."""
    return plan.method.analyze(plan, data, settings={})._attach(plan, data)


def test_a_row_at_a_limit_keeps_the_other_rows_and_differs_from_an_inapplicable_method():
    """One Method beyond its memory or work limit costs that row only, and a capacity refusal is not inapplicability."""
    from nwqlib import Eigenproblem, LinearSystem
    from nwqlib._limits import ResourceLimitError
    from nwqlib.algorithms.protocol import ApplicabilityError

    problem = Eigenproblem(A=[[2.0, 1.0], [1.0, 2.0]])
    first = Lanczos(initial_state=[1, 0], krylov_dimension=2)
    last = Lanczos(initial_state=[0, 1], krylov_dimension=1)
    # Under the same order, the rows beside a row that reached a limit hold the Plans of a
    # comparison in which no limit was reached, because each row's seed comes from its position.
    free = nwqlib.compare(problem, methods=(first, first.revise(max_bytes=10**9), last), seed=7)
    assert all(row.plan is not None and not row.resource_limited for row in free.rows)
    limited = first.revise(max_bytes=1)
    inapplicable = QLS()
    compared = nwqlib.compare(problem, methods=(first, limited, last, inapplicable), seed=7)
    assert [row.plan is not None for row in compared.rows] == [True, False, True, False]
    assert [row.resource_limited for row in compared.rows] == [False, True, False, False]
    assert compared.rows[0].plan.content_id == free.rows[0].plan.content_id
    assert compared.rows[2].plan.content_id == free.rows[2].plan.content_id
    assert compared.rows[1].reason and compared.rows[3].reason
    # Running each row without a Plan raises its own kind of refusal, with its reason.
    with pytest.raises(ResourceLimitError) as raised:
        nwqlib.solve(compared.select(1))
    assert str(raised.value) == compared.rows[1].reason
    with pytest.raises(ApplicabilityError):
        nwqlib.prepare(compared.select(3))
    # plan keeps raising what compare records.
    with pytest.raises(ResourceLimitError):
        nwqlib.plan(problem, method=limited, seed=7)
    # Seeds follow positions, so swapping the Methods compares each Method's row state only.
    swapped = nwqlib.compare(problem, methods=(inapplicable, last, limited, first), seed=7)
    assert [(row.plan is not None, row.resource_limited) for row in swapped.rows] == [
        (False, False), (True, False), (False, True), (True, False)]
    # A work limit is a limit too. A general A with these entries needs p**3 = 8 units of dense
    # completion, whichever seed it gets.
    general = LinearSystem(A=[[1.1, .3], [.1, .9]], b=[1, .25])
    worked = nwqlib.compare(general, methods=(QLS(max_work=1), QLS(epsilon_inv=.1)), seed=7)
    assert worked.rows[0].resource_limited and worked.rows[1].plan is not None


_RECIPES = nwqlib.LinearDynamics(A=ingest_pauli([("I", .3+.1j), ("X", .25), ("Z", .2j), ("Y", -.15j)], num_qubits=1),
                                 initial_state=[2.4, 3.2j], time=.7)
_RECIPE_READOUT = dict(output=QuadraticForm(observable=ingest_pauli([("X", .7), ("Z", -.45), ("I", .2)], num_qubits=1)),
                       shots=64)
_PAIRS = dict(estimator=SampledPairs(repeats=4), ham_evo="trotter_error_budgeted")
_RICHARDSON = dict(_PAIRS, estimator=SampledPairs(repeats=4, product_formula=Richardson(pf_allowance=.01)))
_PTSC = dict(ham_evo=TrotterErrorBudgeted(order=1), estimator=SampledPairs(
    repeats=4, product_formula=PTSC(pf_allowance=.16, mu_cap=1.5, taylor_cutoff=8)))
_QSP_PAIRS = dict(estimator="sampled_pairs", ham_evo="qsp_block_encoding")
_BUDGETED = dict(ham_evo="trotter_error_budgeted")
_QSP = dict(ham_evo="qsp_block_encoding")
_AUTO = dict(lchs_kernel="auto", k_quadrature="auto")
_FIXED = dict(ham_evo=Trotter(steps=4))
# The first example of docs/algorithms/lchs.md.
_GUIDE_A = np.array([[.4, .15], [.05, .25]])
_DENSE = nwqlib.LinearDynamics(A=_GUIDE_A, initial_state=[.6, .8], time=.5)
_SOURCE = nwqlib.LinearDynamics(A=[[.6, .1], [.1, .3]], initial_state=[1., 0.], time=.1, source=[.2, -.1])
_COMPACT_A = ingest_pauli([("I", .4 + .2j), ("X", .2), ("Z", .1 + .3j)], num_qubits=1)
_COMPACT = nwqlib.LinearDynamics(A=_COMPACT_A, initial_state=[.6, .8j], time=.5)
_COMPACT_READOUT = dict(output=QuadraticForm(observable=ingest_pauli([("X", .7), ("Z", -.45)], num_qubits=1)), shots=256)
_EIGEN = nwqlib.Eigenproblem(A=[[.4, .1], [.1, -.7]])
_START = dict(initial_state=[1., 0.])
_X = sp.Symbol("x", real=True)
_LARGE_EIGEN = nwqlib.Eigenproblem(A=np.diag(np.arange(128.)))
_TWO_QUBITS = ingest_pauli((("ZZ", 1.), ("XI", .5), ("IX", .5)), num_qubits=2)
_TWO_STATES = dict(basis=([1, 0, 0, 0], [0, 1, 1, 0]))
_ADAPT_TWO_QUBITS = dict(initial_state=[.6, 0., 0., .8], pool=(ingest_pauli((("XY", 1j),), num_qubits=2),))
_GENERAL_SYSTEM = nwqlib.LinearSystem(A=[[1.1, .3], [.1, .9]], b=[1, .25])
# A band child beside a Pauli child, each order, so that a compact generator's second child holds what the first left.
_BAND_THEN_PAULI = nwqlib.LinearDynamics(A=ingest_cartesian(ingest_band([0, 1, -1], [2., -.5 + .1j, -.5 - .1j], num_qubits=2),
                                                            ingest_pauli([("ZI", .3)], num_qubits=2)),
                                         initial_state=np.array([.6, 0, .8j, 0]), time=.4)
# A dense A of few Pauli words takes the term route of a dense generator, each child planned from its labels.
_DENSE_FEW_WORDS = nwqlib.LinearDynamics(A=.7*np.eye(4) + .3*np.diag([1., -1., -1., 1.]) + .2j*np.fliplr(np.eye(4)),
                                         initial_state=[1., 0., 0., 0.], time=.3)
_PAULI_THEN_BAND = nwqlib.LinearDynamics(A=ingest_cartesian(ingest_pauli([("II", .7), ("ZZ", .3)], num_qubits=2),
                                                            ingest_band([-1, 1], [-.2j, .2j], num_qubits=2)),
                                         initial_state=np.array([.6, 0, .8j, 0]), time=.4)


# Each case holds the Method at its limit, a Method with enough capacity, and the compare options. The caps come from
# capacity sweeps on these inputs and sit below the need that each refusal states. The default caps are
# DenseExact.max_select_slots=4096 and max_basis_size=64.
_LIMITS = dict(
    sampled_step_cap=(_RECIPES, LCHS(**dict(_PAIRS, ham_evo=TrotterErrorBudgeted(max_steps=1))), LCHS(**_PAIRS), _RECIPE_READOUT),
    richardson_step_cap=(_RECIPES, LCHS(**dict(_RICHARDSON, ham_evo=TrotterErrorBudgeted(max_steps=4))),
                         LCHS(**_RICHARDSON), _RECIPE_READOUT),
    ptsc_step_cap=(_RECIPES, LCHS(**dict(_PTSC, ham_evo=TrotterErrorBudgeted(order=1, max_steps=4))),
                   LCHS(**_PTSC), _RECIPE_READOUT),
    ptsc_work_cap=(_RECIPES, LCHS(**_PTSC, max_select_work=4096), LCHS(**_PTSC), _RECIPE_READOUT),
    # At time 500 and this documented tolerance the guide's A needs 8192 slots.
    default_dense_select_slots=(nwqlib.LinearDynamics(A=_GUIDE_A, initial_state=[.6, .8], time=500.),
                                LCHS(approx_tol=1e-3),
                                LCHS(approx_tol=1e-3, **_FIXED), {}),
    eigensolve_work=(_DENSE, LCHS(max_spectral_work=1), LCHS(), {}),
    generator_classification=(_DENSE, LCHS(**_QSP, max_select_work=1), LCHS(**_QSP), {}),
    census_work=(_DENSE, LCHS(**_BUDGETED, max_select_work=16384), LCHS(**_BUDGETED), {}),
    source_branch_work=(_SOURCE, LCHS(max_quadrature_work=16384), LCHS(), {}),
    periodic_work=(nwqlib.LinearDynamics(A=PeriodicStencil(2, mass=0., diffusion=.25), initial_state=[1., 0., 0., 0.],
                                         time=.2),
                   LCHS(ham_evo=Trotter(steps=2), max_select_work=1),
                   LCHS(ham_evo=Trotter(steps=2)), {}),
    classical_kernel_work=(_DENSE, LCHS(max_select_work=1), LCHS(), dict(execution="classical")),
    provider_enumeration_work=(_COMPACT, LCHS(**_AUTO, max_select_work=4096), LCHS(**_AUTO), {}),
    block_encoding_query_bytes=(_COMPACT, LCHS(**_QSP_PAIRS, max_bytes=65536), LCHS(**_QSP_PAIRS), _COMPACT_READOUT),
    # The cosine fit of this long evolution needs an FFT above 1 MiB of arrays.
    coherent_phase_doubling=(nwqlib.LinearDynamics(A=_COMPACT_A, initial_state=[.6, .8j], time=20.),
                             LCHS(max_bytes=2**20), LCHS(), {}),
    block_encoding_work=(nwqlib.LinearSystem(A=[[1.1, .1], [.1, .9]], b=[1., .25]), QLS(epsilon_inv=.1, max_work=1),
                         QLS(epsilon_inv=.1), {}),
    dense_synthesis_work=(nwqlib.LinearSystem(A=[[1.1, .3], [.1, .9]], b=[1., .25]), QLS(epsilon_inv=.1, max_work=16),
                          QLS(epsilon_inv=.1), {}),
    qpe_bytes=(_EIGEN, QCELS(**_START, max_bytes=1), QCELS(**_START), {}),
    qcels_power=(_EIGEN, QCELS(**_START, max_power=1), QCELS(**_START), {}),
    rfe_power=(_EIGEN, RFE(**_START, max_power=1), RFE(**_START), {}),
    default_gcim_basis=(_LARGE_EIGEN, FixedGCIM(basis=tuple(np.eye(128)[i] for i in range(65))),
                        FixedGCIM(basis=tuple(np.eye(128)[i] for i in range(2))), dict(execution="classical")),
    gcim_basis_cap=(nwqlib.Eigenproblem(A=_TWO_QUBITS), FixedGCIM(**_TWO_STATES, max_basis_size=1),
                    FixedGCIM(**_TWO_STATES), {}),
    gcim_analysis_work=(nwqlib.Eigenproblem(A=_TWO_QUBITS), FixedGCIM(**_TWO_STATES, max_analysis_work=1),
                        FixedGCIM(**_TWO_STATES), {}),
    default_qfd_basis=(_EIGEN, QFD(**_START, time_step=.1, powers=tuple(range(65))), Lanczos(initial_state=[1, 0]), {}),
    raw_pauli_bytes=(nwqlib.Eigenproblem(A=[[1., 0.], [0., -1.]]),
                     ADAPT(initial_state=[1, 1], pool=(ingest_pauli((("Y", 1j),), num_qubits=1),), max_bytes=256),
                     ADAPT(initial_state=[1, 1], pool=(ingest_pauli((("Y", 1j),), num_qubits=1),)), {}),
    # A direct two-qubit reference has the CX bound 2*(2**2 - 2) = 4 of reference_cx_bound.
    adapt_reference_depth=(nwqlib.Eigenproblem(A=_TWO_QUBITS),
                           ADAPT(**_ADAPT_TWO_QUBITS, basis_growth=BoundedRitzGrowth(max_chain_length=2, basis_depth_limit=1)),
                           ADAPT(**_ADAPT_TWO_QUBITS, basis_growth=BoundedRitzGrowth(max_chain_length=2, basis_depth_limit=4)),
                           {}),
    expectation_products=(nwqlib.Expectation(state=[1., 0.], observable=[[1., 0.], [0., -1.]]),
                          ExpectationMethod(max_products=1), ExpectationMethod(), dict(execution="classical")),
    qhd_work=(nwqlib.Optimization(objective=(_X - .2)**2, variables=(_X,), bounds=((-1., 1.),)), QHD(max_work=1), QHD(), {}),
)


@pytest.mark.parametrize("problem, limited, enough, options", _LIMITS.values(), ids=_LIMITS.keys())
def test_a_capacity_refusal_of_any_family_gives_a_limited_row_beside_a_method_that_plans(problem, limited, enough,
                                                                                         options):
    from nwqlib._limits import ResourceLimitError

    with pytest.raises(ResourceLimitError) as raised:
        nwqlib.plan(problem, method=limited, seed=7, **options)
    rows = nwqlib.compare(problem, methods=(limited, enough), seed=7, **options).rows
    assert rows[0].plan is None and rows[0].resource_limited and rows[0].reason == str(raised.value)
    assert rows[1].plan is not None and not rows[1].resource_limited
    # The position decides the other row's seed, so it holds the Plan it holds beside a Method that plans.
    free = nwqlib.compare(problem, methods=(enough, enough), seed=7, **options).rows[1]
    assert rows[1].plan.content_id == free.plan.content_id
    with pytest.raises(ResourceLimitError):
        nwqlib.solve(rows[0])


# The problem, the Method at a limit and the compare options.
_NEEDS = dict(
    qls_general_work=(_GENERAL_SYSTEM, lambda limit: QLS(max_work=limit), dict(execution="classical")),
    qls_spectral_selection=(_GENERAL_SYSTEM,
                            lambda limit: QLS(solver="shortcut_native_svp", norm_estimate=1.5, max_work=limit),
                            dict(execution="classical",
                                 output=StateVector(normalization="unit", global_phase="modulo_global_phase"))),
    qls_encoding_construction=(nwqlib.LinearSystem(A=_TWO_QUBITS, b=[1., 0., 0., 0.]),
                               lambda limit: QLS(kappa=8., epsilon_inv=.1, max_work=limit), {}),
    qls_classical_action=(_GENERAL_SYSTEM, lambda limit: QLS(alpha=2., kappa=8., max_work=limit),
                          dict(execution="classical")),
    lchs_dense_generator_work=(_DENSE, lambda limit: LCHS(**_QSP, max_select_work=limit), {}),
    lchs_dense_generator_bytes=(_DENSE, lambda limit: LCHS(**_QSP, max_bytes=limit), {}),
    lchs_compact_generator_work=(_COMPACT, lambda limit: LCHS(max_select_work=limit), {}),
    lchs_dense_term_route_bytes=(_DENSE_FEW_WORDS, lambda limit: LCHS(**_QSP, max_bytes=limit), {}),
    lchs_band_then_pauli_bytes=(_BAND_THEN_PAULI, lambda limit: LCHS(max_bytes=limit), {}),
    lchs_pauli_then_band_work=(_PAULI_THEN_BAND, lambda limit: LCHS(max_select_work=limit), {}),
    raw_pauli_bytes=(nwqlib.Eigenproblem(A=np.diag([1., -1., .5, -.5])),
                     lambda limit: ADAPT(initial_state=[1, 1, 1, 1],
                                         pool=(ingest_pauli((("XY", 1j), ("YX", 1j)), num_qubits=2),), max_bytes=limit),
                     {}),
)


@pytest.mark.parametrize("problem, method, options", _NEEDS.values(), ids=_NEEDS.keys())
def test_following_each_stated_need_from_a_limit_of_one_reaches_a_plan(problem, method, options):
    """Doubling and bisecting a limit finds the least value past each refused stage, and the refusal states that value.

    A nested stage states its own requirement and what was already drawn from the limit, so the value is one stated
    number, written whole or as ``a * b**c``, or the sum of two. Each limit is evaluated through ``compare``, so a stated
    value that raises an option error fails here. The stage's text keeps its numbers, because different stages can
    share a sentence, and masks the limits themselves, which change with each trial.
    """
    def reason_at(limit):
        row = nwqlib.compare(problem, methods=(method(limit),), seed=7, **options).rows[0]
        assert row.plan is not None or row.resource_limited, row.reason
        return row.reason if row.plan is None else None

    def stage(limit):
        reason = reason_at(limit)
        return reason and re.sub(r"(max_\w+=|memory limit of )\d+", r"\1#", reason)

    cap = 1
    while (refused := stage(cap)) is not None:
        low, high = cap, 2*cap
        while stage(high) == refused:
            low, high = high, 2*high
            assert high <= 2**40, refused  # Every need of these small inputs lies below 2**20.
        while high - low > 1:
            middle = (low + high)//2
            low, high = (middle, high) if stage(middle) == refused else (low, middle)
        whole = re.sub(r"(\d+) \* (\d+)\*\*(\d+)", lambda power: str(int(power[1])*int(power[2])**int(power[3])), reason_at(cap))
        numbers = [int(number) for number in re.findall(r"\d+", whole)]
        assert high in numbers or any(first + second == high for first, second in combinations(numbers, 2)), whole
        cap = high


def test_a_phase_synthesis_stopped_by_a_limit_gives_a_row_and_one_stopped_by_its_rule_raises():
    """A next FFT that does not fit is capacity. A failed attempt at a length the ideal rule calls sufficient is numerical.

    The residual tolerance 1e-30 lies far below what binary64 attempts reach on this input, which steers the doubling
    to the ideal rule. It is no practical accuracy request. Under max_bytes=2**16 the doubling stops earlier, at a
    length that the rule has not called sufficient.
    """
    from nwqlib._limits import ResourceLimitError
    from nwqlib.subroutines.qsp.phases import _QSPPhaseConvergenceError

    problem = nwqlib.LinearSystem(A=[[1.1, .1], [.1, .9]], b=[1., .25])
    tight = dict(epsilon_inv=.01, phase_tol=1e-30)
    other = QLS(epsilon_inv=.01)
    with pytest.raises(_QSPPhaseConvergenceError) as ruled:
        nwqlib.compare(problem, methods=(QLS(**tight), other), seed=7)
    assert not isinstance(ruled.value, ResourceLimitError) and ruled.value.evidence.ideal_rule is not None
    limited = QLS(**tight, max_bytes=2**16)
    rows = nwqlib.compare(problem, methods=(limited, other), seed=7).rows
    assert rows[0].resource_limited and rows[0].plan is None and rows[1].plan is not None
    with pytest.raises(_QSPPhaseConvergenceError) as stopped:
        nwqlib.plan(problem, method=limited, seed=7)
    assert isinstance(stopped.value, ResourceLimitError) and rows[0].reason == str(stopped.value)
    evidence = stopped.value.evidence
    assert evidence.ideal_rule is None and evidence.fft_lengths == ruled.value.evidence.fft_lengths[:len(evidence.fft_lengths)]
    assert len(evidence.fft_lengths) < len(ruled.value.evidence.fft_lengths)


def test_a_sufficient_limit_keeps_the_selected_grid_route_and_payload():
    """Limits above every need select what no limit selects, read from the provider records, the row and the phases."""
    def selected(**limits):
        data = nwqlib.plan(_COMPACT, method=LCHS(**_AUTO, **limits), seed=7)._native["native_data"]
        prepared = data.qsp_plan
        providers = [(item["label"], item["status"], item["row"], item["key"]) for item in data.provider_selection]
        rows = [(row.label, row.status, row.reasons) for row in prepared.selection.candidates]
        return providers, prepared.candidate.label, rows, data.quadrature.k_nodes, prepared.phases

    free = selected()
    # 2**40 lies above every byte and work need of this two-dimensional input, which the sweeps put below 2**20.
    capped = selected(max_bytes=2**40, max_select_work=2**40, max_quadrature_work=2**40)
    assert capped[:3] == free[:3]
    np.testing.assert_array_equal(capped[3], free[3])
    np.testing.assert_array_equal(capped[4].cos_solution.phases, free[4].cos_solution.phases)
    np.testing.assert_array_equal(capped[4].sin_solution.phases, free[4].sin_solution.phases)


def _wide_pauli(q):
    return ingest_pauli((("Z" + "I"*(q - 1), 1.), ("X" + "I"*(q - 1), .5)), num_qubits=q)


def _wide_start(q):
    return ingest_occupation((0,)*q, num_qubits=q)


def _wide_dynamics(q):
    terms = (("I"*q, .4 + .2j), ("X" + "I"*(q - 1), .2), ("Z" + "I"*(q - 1), .1 + .3j))
    return nwqlib.LinearDynamics(A=ingest_pauli(terms, num_qubits=q), initial_state=_wide_start(q), time=.5)


def _wide_form(q):
    return QuadraticForm(observable=ingest_pauli((("Z" + "I"*(q - 1), .5), ("I"*(q - 1) + "Z", .25)), num_qubits=q))


def _wide_ratio(q):
    return NormalizedExpectation(observable=_wide_form(q).observable)


@pytest.mark.parametrize("problem, false, true, options", [
    # Eigenvalues 1 +/- sqrt(.02), so alpha/sigma_min is about 1.398 and kappa=1.05 is false.
    (nwqlib.LinearSystem(A=[[1.1, .1], [.1, .9]], b=[1, .25]),
     QLS(alpha=1.2, kappa=1.05), QLS(alpha=1.2, kappa=1.4), {}),
    # A Krylov space of dimension 3 does not exist in dimension 2.
    (nwqlib.Eigenproblem(A=[[2.0, 1.0], [1.0, 2.0]]),
     Lanczos(initial_state=[1, 0], krylov_dimension=3), Lanczos(initial_state=[1, 0], krylov_dimension=2), {}),
    # The dissipative part diag(-1, 1) has the eigenvalue -1, and make_l_psd=False forbids the shift.
    (nwqlib.LinearDynamics(A=np.diag([-1.0, 1.0]) + 0j, initial_state=[1.0, 0.0], time=.5),
     LCHS(ham_evo=Trotter(steps=2), make_l_psd=False),
     LCHS(ham_evo=Trotter(steps=2)), dict(execution="classical")),
    # The diagonals of this Hermitian A are not constant, so it is no periodic banded operator.
    (nwqlib.LinearSystem(A=[[1.2, .7], [.7, .6]], b=[1, .25]),
     QLS(block_encoding="banded", epsilon_inv=.1), QLS(epsilon_inv=.1), {}),
    # alpha=.1 lies below the largest entry 1.2 of this A, and so below its spectral norm.
    (nwqlib.LinearSystem(A=[[1.2, .7], [.7, .6]], b=[1, .25]), QLS(alpha=.1), QLS(epsilon_inv=.1), {}),
    # A one-hot register has a qubit per grid point, and 2**16384 has 4933 decimal digits, above the 4300 of a saved record.
    (nwqlib.Optimization(objective=(_X - .2)**2, variables=(_X,), bounds=((-1., 1.),)),
     QHD(num_grid_points=16384), QHD(num_grid_points=4), {}),
    # ADAPT's Hamiltonian action holds Pauli masks in one 64-bit word.
    (nwqlib.Eigenproblem(A=_wide_pauli(65)),
     ADAPT(initial_state=_wide_start(65), pool=(ingest_pauli((("Y" + "I"*64, 1j),), num_qubits=65),)),
     FixedGCIM(basis=(_wide_start(65),)), {}),
    # The coherent route measures 13 address and 52 system qubits, 65 bits, and sampled pairs measure 53.
    (_wide_dynamics(52), LCHS(), LCHS(**_PAIRS), dict(output=_wide_form(52), shots=16)),
    # QSP pairs measure the branch, 60 system qubits and 4 flags, 65 bits, and product-formula pairs measure 61.
    (_wide_dynamics(60), LCHS(**_QSP_PAIRS), LCHS(**_PAIRS), dict(output=NormSquared(), shots=16)),
    # Every route measures at least the 65 system qubits of this form, so no LCHS Method holds and the row stands alone.
    (_wide_dynamics(65), LCHS(), None,
     dict(output=_wide_form(65), accuracy=nwqlib.Accuracy(absolute_tolerance=.1, component="sampling", confidence=.95))),
    # The classical route of Lanczos reads dense, sparse or Pauli entries, and a band specification holds none of them.
    (nwqlib.Eigenproblem(A=ingest_band((0, 1, -1), (2., -.5, -.5), num_qubits=2)),
     Lanczos(initial_state=[1, 0, 0, 0], krylov_dimension=2), None, dict(execution="classical")),
], ids=("qls_kappa", "lanczos_krylov", "lchs_negative_l", "qls_banded_structure", "qls_alpha_below_entries",
        "qhd_register_record", "adapt_mask_word", "lchs_coherent_counts", "lchs_pair_counts", "lchs_accuracy_counts",
        "lanczos_band_classical"))
def test_a_false_method_premise_gives_a_row_beside_the_method_that_holds(problem, false, true, options):
    from nwqlib.algorithms.protocol import ApplicabilityError

    with pytest.raises(ApplicabilityError):
        nwqlib.plan(problem, method=false, seed=7, **options)
    rows = nwqlib.compare(problem, methods=(false,) if true is None else (false, true), seed=7, **options).rows
    assert rows[0].plan is None and rows[0].reason and not rows[0].resource_limited
    assert true is None or rows[1].plan is not None
    with pytest.raises(ApplicabilityError):
        nwqlib.solve(rows[0])


@pytest.mark.parametrize("problem, method", [
    # Past 64 qubits a Pauli table needs a second mask word, and a state vector of 2**65 amplitudes has no index.
    (nwqlib.Eigenproblem(A=_wide_pauli(65)), Lanczos(initial_state=_wide_start(65), krylov_dimension=2)),
    (nwqlib.Expectation(state=_wide_start(65), observable=_wide_pauli(65)), ExpectationMethod()),
    # ADAPT's packed action tables use a signed index for the 2**63 amplitudes of 63 qubits.
    (nwqlib.Eigenproblem(A=_wide_pauli(63)),
     ADAPT(initial_state=_wide_start(63), pool=(ingest_pauli((("Y" + "I"*62, 1j),), num_qubits=63),))),
], ids=("lanczos", "expectation", "adapt"))
def test_a_classical_action_past_the_full_state_index_gives_a_limited_row(problem, method):
    """The refusal at 62 qubits states the memory, and the state vector is larger still at these widths."""
    row = nwqlib.compare(problem, methods=(method,), execution="classical", seed=7).rows[0]
    assert row.plan is None and row.resource_limited and row.reason


def _analyzed_on_zero_counts(plan, directory, monkeypatch):
    """Run a Plan on the offline backend, every shot reading zero on every classical bit, and return its Result."""
    def zero_counts(self, native):
        return SimpleNamespace(raw_output={"counts": {"0"*native.circuit.num_clbits: native.observation.shots}},
                               metadata={"native_job_id": "original-job"})

    monkeypatch.setattr(InjectedBackend, "_result", zero_counts)
    monkeypatch.setattr(InjectedBackend, "supports_synchronous", True)
    prepared = nwqlib.prepare(plan, backend=InjectedBackend(), directory=directory)
    with prepared.run as run:
        nwqlib.submit(prepared)
        return run.wait(timeout=0)


@pytest.mark.parametrize("settings, output, last", [
    ({}, _wide_form, 51),
    ({}, _wide_ratio, 51),
    ({}, lambda q: Samples(), 51),
    (_QSP_PAIRS, lambda q: NormSquared(), 59),
    (_PAIRS, lambda q: NormSquared(), 63),
], ids=("coherent_form", "coherent_ratio", "coherent_samples", "qsp_pairs", "product_formula_pairs"))
def test_an_lchs_plan_at_64_measured_bits_is_analyzed_and_one_qubit_more_is_refused(settings, output, last, tmp_path,
                                                                                    monkeypatch):
    """The analysis of these counts decodes one 64-bit index per outcome, so the boundary is 64 bits on every route.

    The coherent circuit measures 13 address and q system qubits, QSP pairs the branch, q system qubits and 4 flags, and
    product-formula pairs the branch and q system qubits. The planned circuit's classical declarations state the count.
    """
    method = LCHS(**settings)
    plan = nwqlib.plan(_wide_dynamics(last), method=method, output=output(last), shots=16, seed=7)
    assert sum(item.width for item in plan.construction.program.classical) == 64
    assert _analyzed_on_zero_counts(plan, tmp_path / "run", monkeypatch) is not None
    past = nwqlib.compare(_wide_dynamics(last + 1), methods=(method,), output=output(last + 1), shots=16, seed=7).rows[0]
    assert past.plan is None and past.reason and not past.resource_limited


@pytest.mark.parametrize("q", (52, 64))
def test_the_analysis_of_a_coherent_norm_reads_counts_past_64_bits(q, tmp_path, monkeypatch):
    """The count of a NormSquared sums Python integers (physical_shot_count), so these Plans run and are analyzed.

    The circuit measures 13 address and q system qubits, 65 bits at q=52. At q=64 the dimension 2**64 of the LCU also
    passes through its exact system qubit count, which a floating-point logarithm cannot take.
    """
    plan = nwqlib.plan(_wide_dynamics(q), method=LCHS(), output=NormSquared(), shots=16, seed=7)
    assert sum(item.width for item in plan.construction.program.classical) == 13 + q
    assert _analyzed_on_zero_counts(plan, tmp_path / "run", monkeypatch) is not None


def test_sampled_qls_past_its_count_decoder_gives_a_premise_row():
    """Sampled readout of this system holds three bits beside its q qubits, and the count decoder has 64."""
    from nwqlib.algorithms.protocol import ApplicabilityError

    def row(q):
        system = nwqlib.LinearSystem(A=_wide_pauli(q), b=_wide_start(q))
        return nwqlib.compare(system, methods=(QLS(kappa=8., epsilon_inv=.1),), output=NormSquared(), shots=16, seed=7).rows[0]

    assert row(61).plan is not None
    beyond = row(62)
    assert beyond.plan is None and beyond.reason and not beyond.resource_limited
    with pytest.raises(ApplicabilityError):
        nwqlib.solve(beyond)


def test_supplied_preparation_archive_keeps_global_phase_and_original_parameters(
    tmp_path, monkeypatch
):
    """An analytic exp(i phi) Ry(theta)|0> state exposes lost supplied phase while later caller
    mutations stay isolated.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from nwqlib.core import InputRef, Source
    from nwqlib.problems import bind_preparation_circuit
    from nwqlib._prepared_execution import Run

    phi, theta = 0.375, 0.8125
    circuit = QuantumCircuit(1, global_phase=phi)
    circuit.ry(theta, 0)
    observable = ingest_pauli((("Z", 1.0),), num_qubits=1)
    state = bind_preparation_circuit(
        circuit,
        basis=observable.manifest.basis,
        reference=InputRef(
            identity="supplied-phase-preparation",
            representation="circuit",
            source=Source(
                name="supplied",
                version="1",
                domain="one-qubit unitary",
                reference="exp(i phi) Ry(theta)",
            ),
        ),
    )
    selected = nwqlib.plan(
        nwqlib.Expectation(state=state, observable=observable), method=ExpectationMethod(), seed=7
    )
    from _profile_fixtures import stored_inputs
    from nwqlib.backends import InstructionSupport, assess

    _, _, profile, allocation, context, runtime, at = stored_inputs()
    (block,) = selected.blocks
    configuration = profile.configuration.revise(
        target=profile.configuration.target.revise(
            instructions=(
                InstructionSupport(implementation=block.record.implementation, max_qubits=1),
            )
        )
    )
    allocation = allocation.revise(configuration_id=configuration.content_id)
    (model,) = profile.models
    model = model.revise(
        domain=model.domain.revise(
            configuration_id=configuration.content_id, allocation_id=allocation.content_id
        )
    )
    profile = profile.revise(configuration=configuration, models=(model,))
    assessed = assess(
        selected,
        selected.resolve("expectation"),
        profile=profile,
        allocation=allocation,
        context=context,
        runtime=runtime,
        assessed_at=at,
    )
    support = next(
        item for item in assessed.capability.details if item.scope == "selected instruction support"
    )
    assert support.status == "feasible"
    (prediction,) = assessed.predictions
    assert prediction.seconds is None
    assert any("outside the model operation mix" in reason for reason in prediction.reasons)
    run = Run(selected)
    folder = partial_result(selected, run.data).save(tmp_path / "phase")
    circuit.global_phase = -0.25
    circuit.ry(0.5, 0)
    with monkeypatch.context() as patch:
        patch.setattr(ExpectationMethod, "plan", forbidden)
        patch.setattr("nwqlib.blocks.selection.select_preparation", forbidden)
        restored = nwqlib.load_result(folder)
    assert restored.plan.problem.state._native.global_phase == phi
    assert restored.plan.problem.state._native.data[0].operation.params == [theta]
    assert len(restored.plan.problem.state._native.data) == 1
    prepared = nwqlib.prepare(restored.plan)
    native = prepared.circuits[0]
    assert native.data[-1].operation.name == "save_expval"
    native.data.pop()  # Omit only the exact readout on this detached circuit view.
    # Physical amplitudes, including global phase, follow the supplied Ry action.
    expected = np.exp(1j * phi) * np.array([np.cos(theta / 2), np.sin(theta / 2)])
    np.testing.assert_allclose(
        Statevector.from_instruction(native).data, expected, atol=3e-14, rtol=0
    )
    result = nwqlib.submit(prepared).wait()
    assert result.value == pytest.approx(np.cos(theta), abs=3e-14, rel=0)
    assert restored.value is None and restored.data.trace.events == ()
    prepared.run.close()
    run.close()


def test_failed_public_submission_keeps_partial_contributions_and_original_exposure(
    tmp_path, monkeypatch
):
    """Fail the second setting and preserve the first contribution alongside all charged,
    uncertain and forecast evidence.
    """
    from nwqlib.backends import align_telemetry
    from nwqlib.scientist import ComparisonRow
    from _profile_fixtures import original_forecast

    problem = nwqlib.Expectation(
        state=[1, 0], observable=ingest_pauli((("Z", 1.0), ("X", 1.0)), num_qubits=1)
    )
    selected = nwqlib.plan(problem, method=ExpectationMethod(), shots=4, seed=7)
    forecast, allocation = original_forecast(selected)
    row = ComparisonRow(selected.method, selected, estimate=forecast, allocation=allocation)
    InjectedBackend.supports_synchronous = True
    prepared = nwqlib.prepare(row, backend=InjectedBackend(), directory=tmp_path / "run")
    run = prepared.run
    assert len(prepared.circuits) == run.trace.preparations == 1
    initial = partial_result(selected, run.data)
    assert initial.value is None and initial.data.receipts == run.prepared_artifacts
    assert not initial.data.trace.events
    assert (
        nwqlib.load_result(initial.save(tmp_path / "unused")).data.receipts == initial.data.receipts
    )
    calls = []
    failure = RuntimeError("second setting failed after first immutable chunk")
    original_submit = InjectedBackend.submit

    def acquire(self, native, *, submission_id, run):
        calls.append(native)
        assert run.trace.events[-1].assessment_id in {a.content_id for a in forecast.assessments}
        if len(calls) == 2:
            raise failure
        return original_submit(self, native, submission_id=submission_id, run=run)

    monkeypatch.setattr(InjectedBackend, "submit", acquire)
    with pytest.raises(RuntimeError) as raised:
        nwqlib.submit(prepared)
    assert raised.value is failure and prepared.run is run
    assert [event.status for event in run.trace.events] == ["completed", "uncertain"]
    assert sum(event.shots for event in run.trace.events) == 8 and len(run.observations.chunks) == 1
    assert len(run.prepared_artifacts) == 2 and run.result is None
    snapshot, rng = run.data, run.rng.snapshot()
    monkeypatch.setattr(InjectedBackend, "submit", forbidden)
    monkeypatch.setattr(InjectedBackend, "refresh", forbidden)
    monkeypatch.setattr(ExpectationMethod, "plan", forbidden)
    partial = partial_result(selected, snapshot)
    assert partial.value is None and partial.missing
    assert partial.contribution_ids == (snapshot.observations.chunks[0].content_id,)
    assert partial.plan.problem is problem and partial.data is snapshot
    with monkeypatch.context() as patch:
        patch.setattr(ExpectationMethod, "analyze", forbidden)
        report = partial.report()
    assert "unavailable" in report["summary"] and "Partial data" in report["summary"]
    assert "1/1 chunks used" in report["summary"] and "2 attempts, 1 uncertain" in report["summary"]
    assert ObservationView.model_validate(report["observations"]) == snapshot.observations
    assert report["receipts"] == [receipt.model_dump(mode="json") for receipt in snapshot.receipts]
    assert report["forecast"] == forecast.model_dump(mode="json")
    assert report["allocation"] == allocation.model_dump(mode="json")
    assert [event["status"] for event in report["trace"]["events"]] == ["completed", "uncertain"]
    ledger = align_telemetry(partial)
    assert len(ledger.rows) == 2 and ledger.rows[0].contribution_ids == partial.contribution_ids
    assert not ledger.rows[1].contribution_ids and ledger.rows[1].event.failure.startswith(
        "RuntimeError:"
    )
    folder = partial.save(tmp_path / "partial")
    restored = nwqlib.load_result(folder)
    assert restored.data.trace == snapshot.trace
    assert restored.data.forecast == forecast
    assert (
        restored.data.allocation == allocation
        and restored.contribution_ids == partial.contribution_ids
    )
    again = restored.analyze()
    assert again.value is None and again.data is restored.data
    assert len(calls) == 2 and run.rng.snapshot() == rng
    assert nwqlib.load_result(folder).data.trace.events == snapshot.trace.events
    assert run.trace.events == snapshot.trace.events
    # Collected data and charged exposure are different populations: an empty
    # reduction keeps both attempted settings and all receipts in its snapshot.
    empty_data = replace(snapshot, observations=ObservationView())
    empty = partial_result(selected, empty_data)
    assert empty.contribution_ids == () and empty.data.trace.events == snapshot.trace.events
    assert (
        nwqlib.load_result(empty.save(tmp_path / "empty")).data.trace.events
        == snapshot.trace.events
    )
    run.close()
