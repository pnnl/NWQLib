"""The coherent source LCU preserves its independent physical grid."""

import math

import numpy as np
import pytest
from scipy.linalg import expm

from nwqlib import LinearDynamics, plan, solve
from nwqlib.algorithms.lchs import LCHS, ProviderConfig


def test_actual_qsp_source_binds_elapsed_diagonals_recovery_and_physical_phase(monkeypatch):
    """Each application keeps its own elapsed time, mass and input in the coherent source LCU.

    The expected values come from an independent signed-address grid, the
    Cauchy coefficients c_j = .5/(pi (1 + k_j**2)), NumPy's two-point Gauss
    rule and explicit matrix exponentials: label a of the application
    register carries t_a, the time factor q_a = t_a/T of the joint
    generator and the PREP amplitude sqrt(m_a / sum m) with
    m_a = w_a ||z_a|| exp(s t_a); gamma = sum_a m_a sum_j |c_j| and the
    recovery is gamma times the selected route's zeta.
    """
    matrix = np.array([[-.2, .15+.1j], [.04-.06j, .3]])
    initial, source = np.array([.8, .6j]), np.array([.3, -.2j])
    elapsed = .5
    method = LCHS(hamiltonian_evolution_backend='qsp_block_encoding', duhamel_nodes=2,
        lchs_kernel=ProviderConfig(implementation='cauchy_density'),
        k_quadrature=ProviderConfig(implementation='signed_binary_uniform',
            parameters={'num_qubits':2, 'lsb_position':-1}))
    chosen = plan(LinearDynamics(A=matrix, initial_state=initial, source=source, time=elapsed), method=method)
    data = chosen._native['native_data']
    prepared = data.qsp_plan
    preparation = prepared.preparation
    nodes = np.array([0., .5, -1., -.5])
    coefficients = .5/(np.pi*(1+nodes**2))
    source_times = elapsed/2*(1+np.array([-1., 1.])/np.sqrt(3))
    l_part, h_part = (matrix+matrix.conj().T)/2, (matrix-matrix.conj().T)/(2j)
    assert np.linalg.norm(l_part@h_part-h_part@l_part)>.01
    radius = np.sqrt(((l_part[0,0].real-l_part[1,1].real)/2)**2+abs(l_part[0,1])**2)
    minimum = (l_part[0,0].real+l_part[1,1].real)/2 - radius
    maximum = (l_part[0,0].real+l_part[1,1].real)/2 + radius
    shift = -minimum + method.psd_tolerance*max(abs(minimum), abs(maximum))
    assert minimum<0
    assert chosen.reconstruction.psd_shift==pytest.approx(shift, abs=2e-16)
    np.testing.assert_allclose(chosen.reconstruction.source_nodes, source_times, rtol=0, atol=1e-16)
    np.testing.assert_allclose(chosen.reconstruction.source_weights, [elapsed/2]*2, rtol=0, atol=1e-16)
    # Application labels: the initial state on 0 and the two Duhamel nodes on
    # 2 and 3, above the kind bit 1; label 1 is padding.
    times, masses = np.zeros(4), np.zeros(4)
    expected = np.zeros(2, dtype=complex)
    applications = [(0, elapsed, 1., initial),
        (2, elapsed-source_times[0], elapsed/2, source),
        (3, elapsed-source_times[1], elapsed/2, source)]
    for label, time, weight, vector in applications:
        times[label] = time
        masses[label] = weight*np.linalg.norm(vector)*np.exp(shift*time)
        for node, coefficient in zip(nodes, coefficients, strict=True):
            expected += weight*coefficient*np.exp(shift*time)*(expm(
                -1j*time*(node*(l_part+shift*np.eye(2))+h_part))@vector)
    assert preparation.kind_qubit == 1
    assert preparation.application_kinds == ("initial", None, "source", "source")
    np.testing.assert_allclose(preparation.application_times, times, rtol=0, atol=2e-16)
    np.testing.assert_allclose(prepared.generator.components['time'].values, times/elapsed, rtol=0, atol=2e-16)
    # The PREP amplitudes sqrt(m_a/sum m) and sqrt(|c_j|/sum|c|). A relative
    # error of a few 1e-16 in m_a or c_j changes an amplitude by at most half
    # of it, far below 1.5e-14. The c_j are real and positive and the dense
    # recipe extracts no scalar, so every slot phase is zero.
    np.testing.assert_allclose(preparation.application_amplitudes, np.sqrt(masses/np.sum(masses)),
                               rtol=1.5e-14, atol=0)
    np.testing.assert_allclose(preparation.kernel_amplitudes, np.sqrt(coefficients/np.sum(coefficients)),
                               rtol=1.5e-14, atol=0)
    assert preparation.phase_recipe == "none"
    assert not any(preparation.kernel_phases) and not any(preparation.application_phases)
    gamma = np.sum(masses)*np.sum(coefficients)
    assert math.ldexp(*prepared.physical_recovery) == pytest.approx(gamma, rel=2e-15, abs=0)
    # zeta of the selected route from its own synthesized target: 2s (shared
    # raw), 1/(3a - 4a**3) with a = 1/(2s) (shared OAA), or the qubitized scale.
    route, phases = prepared.candidate.route, prepared.phases
    amplitude = .5/phases.scale
    zeta = {"shared_raw": 2*phases.scale, "shared_oaa": 1/(3*amplitude-4*amplitude**3)}.get(route, phases.scale)
    assert prepared.recovery == pytest.approx(zeta, rel=2e-15, abs=0)
    assert chosen.reconstruction.recovery.as_float() == pytest.approx(gamma*zeta, rel=4e-15, abs=0)
    homogeneous = plan(LinearDynamics(A=matrix, initial_state=initial, time=elapsed), method=method)
    homogeneous_route = homogeneous._native['native_data'].qsp_plan
    assert homogeneous_route.preparation.application_qubits == 0
    assert homogeneous_route.generator.components['time'] is None
    assert sum(register.width for register in chosen.construction.program.registers)<=13
    from nwqlib.algorithms.lchs import coherent_selection
    from nwqlib.subroutines.qsp import evolution, phases as phase_owner
    for module, name in ((evolution, 'prepare_qsp_evolution'), (evolution, 'prepare_jacobi_anger_laurent'),
                         (evolution, 'prepare_jacobi_anger_gqet'), (phase_owner, 'solve_laurent_qsp_phases'),
                         (coherent_selection, 'select_coherent_route')):
        monkeypatch.setattr(module, name, lambda *a, **kw: pytest.fail('selected phases fitted again'))
    actual = solve(chosen)
    synthesis = next(item.fact.value.value for item in chosen.facts if item.fact.quantity == "qsp_synthesis")
    # This is the same finite grid. The deviation is the evolution's own
    # approximation, which the qsp_synthesis component gamma*(algorithmic +
    # child) bounds, plus binary64 simulation rounding. It is not a claim that
    # this small Cauchy grid solves the exact ODE.
    assert np.linalg.norm(actual.solution-expected) <= synthesis+1e-12


GRID = dict(hamiltonian_evolution_backend='qsp_block_encoding', duhamel_nodes=2,
            lchs_kernel=ProviderConfig(implementation='cauchy_density'),
            k_quadrature=ProviderConfig(implementation='signed_binary_uniform',
                                        parameters={'num_qubits': 2, 'lsb_position': -1}))


def test_compact_source_applies_decay_and_scalar_phase_at_each_elapsed_time():
    """A general complex Pauli A with known decay matches the finite sum of each application.

    A = (1 + .3i)I + .25X - .1Y + .4iZ has noncommuting parts L = I + .25X - .1Y
    and H = .3I + .4Z. The Pauli enclosure of L gives ell = 1 - .35 = .65 > 0, so
    the plan extracts the known decay s = -.65. Application a, with elapsed time
    t_a and weight w_a, contributes
    w_a*c_j*exp(s*t_a)*exp(-i t_a (k_j (L + s I) + H)) applied to its own input,
    written here from explicit matrices with NumPy's Gauss-Legendre rule.
    Evaluating the scalar phase or the decay at T instead of t_a moves this sum
    by 9.1e-3 and 2.1e-2, and preparing b with a different global phase by .14,
    all far above the QSP tolerance 1e-4 of the dense case above.

    This is also the public check that the PREP labels, the joint generator's
    registers and the evolution emitter stay associated: the emitter applies
    the generator on [node, application, physical] wires, and a query that
    reaches them in another order (for example reversed) pairs nodes, times
    and inputs that the sum does not contain. The case has 10 qubits.

    The same Plan's formula breakdown gives the whole source body one call
    with the selected row's attempt-CX law, the no-ancilla upper bound of its
    PREP pair, coupled phase, input preparations and evolution
    (coherent_selection module docstring), and the built body transpiled at
    fixed options has at most that many CX.
    """
    from types import SimpleNamespace
    from qiskit import transpile
    from qiskit.quantum_info import SparsePauliOp
    from nwqlib.algorithms.lchs.native import construct_coherent_lcu
    from nwqlib.algorithms.lchs.quantum_resources import resource_breakdown
    from nwqlib.operators import ingest_pauli
    A = ingest_pauli([("I", 1.+.3j), ("X", .25), ("Z", .4j), ("Y", -.1)], num_qubits=1)
    matrix = SparsePauliOp.from_list(A.pauli_terms().labels()).to_matrix()
    l_part, h_part = (matrix+matrix.conj().T)/2, (matrix-matrix.conj().T)/(2j)
    assert np.linalg.norm(l_part@h_part-h_part@l_part) > .3
    initial, source, elapsed = np.array([.6, .8j]), np.array([.3j, -.2]), .7
    chosen = plan(LinearDynamics(A=A, initial_state=initial, source=source, time=elapsed), method=LCHS(**GRID))
    assert chosen.reconstruction.psd_shift == 0
    assert chosen.reconstruction.scalar_decay == pytest.approx(.65, rel=1e-15)
    nodes = np.array([0., .5, -1., -.5])
    coefficients = .5/(np.pi*(1+nodes**2))
    points, weights = np.polynomial.legendre.leggauss(2)
    applications = [(elapsed, 1., initial)] + [(elapsed-elapsed/2*(1+x), elapsed/2*w, source)
                                               for x, w in zip(points, weights, strict=True)]
    expected = sum(weight*c*np.exp(-.65*time)*(expm(-1j*time*(k*(l_part-.65*np.eye(2))+h_part))@vector)
                   for time, weight, vector in applications for k, c in zip(nodes, coefficients, strict=True))
    assert sum(register.width for register in chosen.construction.program.registers) <= 10
    result = solve(chosen)
    np.testing.assert_allclose(result.solution, expected, rtol=0, atol=1e-4)
    # Planning bounds the Duhamel remainder of this compact A with the
    # structural bound on ||A||. Refinement has no dense A to tighten it and
    # reports the same positive value, never zero.
    from nwqlib.algorithms.lchs import LCHSRefinement
    planned = {item.fact.quantity: item.fact for item in chosen.facts}["duhamel_quadrature"].value.value
    _, refined = result.verify(checks=LCHSRefinement(components=("duhamel",)))
    (refined,) = (item.fact for item in refined if item.fact.quantity == "duhamel_quadrature")
    assert planned > 0 and refined.value.value == planned
    options = dict(basis_gates=['cx', 'u'], optimization_level=0, seed_transpiler=7)
    data = chosen._native['native_data']
    candidate = data.qsp_plan.candidate
    (entry,) = resource_breakdown(chosen)
    (body,) = [row for row in entry['blocks'] if row['block'] == 'lchs_coherent_lcu']
    assert body['calls'] == 1 and body['laws'][0].value == candidate.cost['attempt_cx']
    built = construct_coherent_lcu(SimpleNamespace(_payload=data), (), None)
    assert 0 < transpile(built, **options).count_ops().get('cx', 0) <= candidate.cost['attempt_cx']


def test_zero_evolution_scales_outside_binary64_keep_amplitudes_phase_and_positive_bounds():
    """Input and decay scales beyond binary64 keep the finite-sum meaning of a constant source.

    With A = c I every branch is the scalar exp(-c t), so the finite sum is
    exp(-c T) u0 + sum_q w_q exp(-c (T - r_q)) b on the Gauss-Legendre rule,
    with no phase solve. Each application mass w_a*||z_a||*exp(s*t_a) is one
    PhysicalScale product, integer exponents subtract exactly, a PREP
    amplitude sqrt(m_a/sum m) is formed without its probability, and a
    positive bound below binary64 is unknown, not zero.
    """
    from math import exp, expm1, ldexp, log, log2, sqrt
    from qiskit import QuantumCircuit
    from nwqlib import StateVector
    from nwqlib.algorithms.lchs.source_selection import application_facts, select_source_branches
    from nwqlib.algorithms.protocol import ApplicabilityError
    from nwqlib.core.records import Basis, InputRef, Source
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems.inputs import PhysicalScale, bind_preparation_circuit

    def scalar(c):
        return ingest_pauli([("I", c)], num_qubits=1)

    def facts(chosen):
        return {item.fact.quantity: item.fact for item in chosen.facts}

    unit = StateVector(normalization="unit")
    # Supplied circuits with equal norms 2**(2**64 + 2048) and phases .9 and -.4:
    # the relative amplitudes are O(1), so the unit state shows the relative
    # phase and the physical scale keeps the exact exponent.
    exponent, ordering = 2**64+2048, scalar(.5).basis.ordering

    def supplied(label, angle, phase):
        circuit = QuantumCircuit(1)
        circuit.ry(angle, 0)
        reference = InputRef(identity=f"test:{label}", representation="circuit",
                             source=Source(name="test", version="1", domain="test", reference="declared"))
        return bind_preparation_circuit(circuit, reference=reference, basis=Basis(identity="computational",
            dimension=2, ordering=ordering), norm=PhysicalScale(mantissa=.5, exponent=exponent+1), phase=phase)

    result = solve(LinearDynamics(A=scalar(.5), initial_state=supplied("u", .7, .9), source=supplied("b", 2.1, -.4),
                                  time=1.), method=LCHS(hamiltonian_evolution_backend='qsp_block_encoding',
                                                        duhamel_nodes=2), output=unit)
    points, weights = np.polynomial.legendre.leggauss(2)
    relative = (np.exp(-.5+.9j)*np.array([np.cos(.35), np.sin(.35)])
                + sum(w/2*np.exp(-.5*(1-(1+x)/2)) for x, w in zip(points, weights, strict=True))
                * np.exp(-.4j)*np.array([np.cos(1.05), np.sin(1.05)]))
    # Rounding of the amplitude and phase synthesis only; a lost scale or
    # phase changes these O(1) amplitudes by O(1).
    np.testing.assert_allclose(result.state_vector, relative/np.linalg.norm(relative), rtol=0, atol=1e-12)
    scale = result.physical_scale
    assert scale.exponent-exponent+log2(scale.mantissa) == pytest.approx(log2(np.linalg.norm(relative)), abs=1e-12)

    # Two inputs of norm 1e308 with A = 0: their coefficient mass exceeds
    # binary64, but the PREP amplitudes are (1, 1)/sqrt(2) and u(T) = u0 + T*b
    # has representable coordinates.
    method = LCHS(hamiltonian_evolution_backend='qsp_block_encoding', duhamel_nodes=1)
    result = solve(LinearDynamics(A=scalar(0.), initial_state=[1e308, 0.], source=[0., 1e308], time=1.),
                   method=method, output=unit)
    np.testing.assert_allclose(result.state_vector, [2**-.5, 2**-.5], rtol=0, atol=1e-15)
    scale = result.physical_scale
    assert scale.exponent+log2(scale.mantissa) == pytest.approx(log2(1e308)+.5, abs=1e-12)
    # Source mass w*||b||*exp(s*T/2) with T = 2**-600, ||b|| = .6*2**-470 and
    # s*T/2 = 100*ln 2: the product w*||b|| alone is subnormal, the mass is
    # .6*2**-970. gamma = coefficient_l1_norm*2**scale_exponent carries it;
    # the rounding of s*T/2 changes it by a relative 1e-14 at most.
    final = 2.0**-600
    layout = select_source_branches(coefficients=(1+0j,), k_nodes=(0.,), initial_scale=PhysicalScale(mantissa=0,
        exponent=0), source_scale=PhysicalScale(mantissa=.6, exponent=-470), final_time=final,
        shift=200*log(2.)/final, duhamel_nodes=1, compact=True)
    assert ldexp(layout.coefficient_l1_norm, layout.scale_exponent) == pytest.approx(ldexp(.6, -970), rel=1e-13, abs=0)

    # A = 2000 I: the initial application has relative mass exp(-1000), below
    # binary64 as a probability, but the normal relative amplitude exp(-500).
    # Each mass carries the rounding of x/ln 2 in exp(x) = 2**(x/ln 2), of
    # relative order u*|x|/ln 2 <= 1.1e-16*2886, so 1e-12 bounds the ratio.
    problem = dict(initial_state=[1., 0.], source=[0., 1.], time=1.)
    chosen = plan(LinearDynamics(A=scalar(2000.), **problem), method=method, output=unit)
    # Label 0 is the initial application and label 1 the one source node.
    amplitudes = chosen._native["native_data"].qsp_plan.preparation.application_amplitudes
    assert amplitudes[0]/amplitudes[1] == pytest.approx(exp(-500.), rel=1e-12, abs=0)
    result = solve(chosen)
    np.testing.assert_allclose(result.state_vector, [0., 1.], rtol=0, atol=1e-15)
    scale = result.physical_scale
    assert scale.exponent+log2(scale.mantissa) == pytest.approx(-1000/log(2.), abs=1e-11)
    # A = 3000 I: the initial amplitude exp(-750) underflows, so it is omitted
    # only under an explicit allowance that covers its bound exp(-3000),
    # reported as the binary64 upper bound 2**-1074.
    with pytest.raises(ApplicabilityError, match="source_drop_allowance"):
        plan(LinearDynamics(A=scalar(3000.), **problem), method=method)
    allowed = plan(LinearDynamics(A=scalar(3000.), **problem),
                   method=LCHS(hamiltonian_evolution_backend='qsp_block_encoding', duhamel_nodes=1,
                               source_drop_allowance=1e-300))
    assert facts(allowed)["source_range_omission"].value.value == 5e-324
    preparation = allowed._native["native_data"].qsp_plan.preparation
    assert preparation.omitted_applications == (0,)
    assert {kind for kind in preparation.application_kinds if kind} == {"source"}
    # Dense A = -1e10 I: the SELECT layout's source mass lies about 2**7.2e9 below the initial one, an exponent
    # beyond a C int. Its slots are omitted only under an allowance that covers their bound, which no binary64
    # allowance does.
    # Initial norm 2**-1 and source norm .99*2**E at T = 1, one kernel node: the source row is
    # sqrt(1.98/Z)*2**h with Z = 1 + 2**-2150 or less. h = -1075 (E = -2150) rounds it up to the least
    # subnormal and keeps it; h = -1076 (E = -2152) gives exactly zero, which is omitted with the exact
    # bound .99*2**-2152 under an allowance and refused without one.
    kept = dict(coefficients=(1+0j,), k_nodes=(0.,), initial_scale=PhysicalScale(mantissa=.5, exponent=0),
                final_time=1., shift=0., duhamel_nodes=1)
    edge = select_source_branches(source_scale=PhysicalScale(mantissa=.99, exponent=-2150), **kept)
    assert edge.dropped_bound is None and edge.amplitudes[edge.application_slots[1][0]] == 5e-324
    with pytest.raises(ApplicabilityError, match="source_drop_allowance"):
        select_source_branches(source_scale=PhysicalScale(mantissa=.99, exponent=-2152), **kept)
    edge = select_source_branches(source_scale=PhysicalScale(mantissa=.99, exponent=-2152), drop_allowance=1e-300,
                                  **kept)
    assert len(edge.applications) == 1 and (edge.dropped_bound.mantissa, edge.dropped_bound.exponent) == (.99, -2152)
    with pytest.raises(ApplicabilityError, match=r"2\*\*7213475205 to omit them"):
        plan(LinearDynamics(A=-1e10*np.eye(2), **problem), method=LCHS(duhamel_nodes=1, source_drop_allowance=1e300),
             execution="quantum")

    # The registered grid signed_binary_uniform(2, lsb -1072) with the Cauchy
    # kernel stores four c_j = 2**-1074. For A = (.2 + .6i)I + .2Z the compact
    # selection keeps L = .2I + .2Z (s = 0) and extracts the scalar phase
    # exp(-i*t*(.2*k + .6)), which is exp(-.6i*t) at k_j near 2**-1072. The
    # homogeneous finite sum 2**E*sum_j d_j is then 4*2**-1074*exp(-.6i), and
    # the source layout with input norms 2**1074, orthogonal inputs and one
    # Duhamel node at T/2 of weight T = 1 is (4*exp(-.6i), 4*exp(-.3i)). A
    # phase taken from a rotated subnormal coefficient is -pi/4 or 0 instead.
    # A few roundings of binary64 magnitudes and phases bound each error.
    from nwqlib.algorithms.lchs.selection import select_compact
    tiny = LCHS(hamiltonian_evolution_backend='qsp_block_encoding', duhamel_nodes=1,
                lchs_kernel=ProviderConfig(implementation='cauchy_density'),
                k_quadrature=ProviderConfig(implementation='signed_binary_uniform',
                                            parameters={'num_qubits': 2, 'lsb_position': -1072}))
    data = select_compact(tiny, LinearDynamics(A=ingest_pauli([("I", .2+.6j), ("Z", .2)], num_qubits=1),
                                               initial_state=[1., 0.], time=1.))
    assert data.quadrature.coefficients.tolist() == [2.0**-1074]*4
    homogeneous = data.coefficient_plan
    total = sum(complex(ldexp(d.real, homogeneous.recovery_exponent+1074), ldexp(d.imag, homogeneous.recovery_exponent+1074))
                for d in homogeneous.coefficients)
    assert abs(total-4*np.exp(-.6j)) <= 1e-14
    big = PhysicalScale(mantissa=.5, exponent=1075)
    layout = select_source_branches(coefficients=data.quadrature.coefficients, k_nodes=data.quadrature.k_nodes,
        initial_scale=big, source_scale=big, final_time=1., shift=0., duhamel_nodes=1, compact=True,
        scalar_phase=(data.generator.dissipative_extracted, data.generator.hamiltonian_extracted))
    vector = ldexp(layout.coefficient_l1_norm, layout.scale_exponent)*np.array(
        [np.sum(layout.amplitudes[rows]**2*np.exp(1j*layout.phases[rows])) for rows in (slice(0, 4), slice(4, 8))])
    np.testing.assert_allclose(vector, 4*np.exp(-1j*np.array([.6, .3])), rtol=0, atol=1e-14)
    # (1+i)*2**-1074 has magnitude sqrt(2)*2**-1074, which abs rounds to
    # 2**-1074; with input norm 2**1074 the physical coefficient mass is
    # sqrt(2), up to one rounding of hypot. The mixed table
    # (1e14 + i*2**-1074, (1+i)*2**-1074) keeps the positive amplitude
    # sqrt(sqrt(2)*2**-1074/1e14) = 2**.25*2**-537/1e7, up to a few roundings.
    eta, none = 2.0**-1074, PhysicalScale(mantissa=0, exponent=0)
    layout = select_source_branches(coefficients=(complex(eta, eta),), k_nodes=(0.,), final_time=1., shift=0.,
        initial_scale=PhysicalScale(mantissa=.5, exponent=1075), source_scale=none, duhamel_nodes=1, compact=True)
    assert ldexp(layout.coefficient_l1_norm, layout.scale_exponent) == pytest.approx(sqrt(2.), rel=2**-52, abs=0)
    layout = select_source_branches(coefficients=(complex(1e14, eta), complex(eta, eta)), k_nodes=(0., 1.),
        initial_scale=PhysicalScale(mantissa=.5, exponent=1), source_scale=none, final_time=1., shift=0.,
        duhamel_nodes=1, compact=True)
    assert layout.amplitudes[1] == pytest.approx(2**.25*2.0**-537/1e7, rel=1e-15, abs=0)

    # A = 1e-200 I: the source-quadrature bound a**2*T**3*||b||/24 is positive
    # and below binary64, so it is reported as the upper bound 2**-1074. A = 0
    # keeps its exact zero.
    duhamel = facts(plan(LinearDynamics(A=scalar(1e-200), **problem), method=method))["duhamel_quadrature"]
    assert duhamel.availability == "concrete" and duhamel.value.value == np.nextafter(0., 1.)
    exact = facts(plan(LinearDynamics(A=scalar(0.), **problem), method=method))["duhamel_quadrature"]
    assert exact.availability == "concrete" and exact.value.value == 0.0

    # Continuous source mass nu_b*integral over [a, b] of exp(s*t) dt, which
    # equals nu_b*(exp(s*b) - exp(s*a))/s, on each branch of its evaluation.
    zero = PhysicalScale(mantissa=0, exponent=0)

    def continuous(norm, shift, interval, final=1.):
        return application_facts(kind="continuous_source", final_time=final, signed_shift=shift, scalars=(-shift, 0.),
                                 initial_norm=zero, source_norm=norm, source_interval=interval)

    def value(mass, exponent=0):
        return ldexp(mass.mantissa, mass.exponent-exponent)

    unit_norm, big = PhysicalScale(mantissa=.5, exponent=1), PhysicalScale(mantissa=.5, exponent=1601)
    # A zero source or an empty interval has zero mass and no active time.
    for case in (continuous(zero, 0., (0., 1.)), continuous(unit_norm, 0., (.5, .5))):
        assert case.source_mass.mantissa == 0 and case.active_time == 0.
    # s*Delta = 2**-1200 underflows to 0; the mass is still Delta = 2**-600.
    tiny = continuous(unit_norm, final, (0., final), final=final)
    assert value(tiny.source_mass) == final and tiny.active_time == final
    # s*Delta = .8 and 1.6 (the source norm 2**1600 keeps its exponent), and
    # s*Delta overflowing for s = -1e300, where the integral is 1/|s|.
    # expm1, exp and the products round to a few units in the last place.
    assert (value(continuous(unit_norm, 2., (.6, 1.)).source_mass)
            == pytest.approx((exp(2.)-exp(1.2))/2, rel=4e-15, abs=0))
    assert (value(continuous(big, -2., (.2, 1.)).source_mass, 1600)
            == pytest.approx(expm1(-2.)/-2-expm1(-.4)/-2, rel=4e-15, abs=0))
    assert (value(continuous(unit_norm, -1e300, (0., 1e10), final=1e10).source_mass)
            == pytest.approx(1e-300, rel=4e-15, abs=0))


def test_strong_known_decay_moves_into_the_recovery_and_keeps_every_coefficient():
    """A common compensation exp(s*T) far below binary64 changes the recovery, not the coefficient state.

    L = 708.1 I + .1 X with H = 0 gives the known decay s close to -708 and the
    bound .2 on ||L + sI||. A positive factor common to every coefficient
    leaves the normalized PREP amplitudes sqrt(|c_j|/sum|c|) unchanged, so they
    equal those of the uncompensated kernel coefficients, including the
    7.7e-11 amplitude of the near_optimal_eq7 grid at lsb_position 8 that
    exp(-708)*c_j would underflow. On the Cauchy grid at lsb_position -100
    every c_j*exp(-708) underflows; the recovery is then
    ||u0||*exp(s*T)*sum|c_j| times the QSP recovery. Each amplitude carries
    a few rounding errors, and the log2 recovery a relative error of order
    u*|s*T|/ln 2, about 1.2e-13. The request names the separate-parts generator, so the kernel route is used
    even though H = 0 also admits the cheaper imaginary-time filter.
    """
    from math import log, log2
    from nwqlib.algorithms.lchs.providers import resolve_lchs_coefficient_plan
    from nwqlib.operators import ingest_pauli
    A = ingest_pauli([("I", 708.1), ("X", .1)], num_qubits=1)
    initial = np.array([.6, .8j])
    for kernel, lsb in (("near_optimal_eq7", 8), ("cauchy_density", -100)):
        method = LCHS(hamiltonian_evolution_backend='qsp_block_encoding',
                      qsp_route={"implementation": "auto", "parameters": {"joint": "separate_parts"}},
                      lchs_kernel=ProviderConfig(implementation=kernel),
                      k_quadrature=ProviderConfig(implementation='signed_binary_uniform',
                                                  parameters={'num_qubits': 2, 'lsb_position': lsb}))
        chosen = plan(LinearDynamics(A=A, initial_state=initial, time=1.), method=method)
        data = chosen._native["native_data"]
        raw = np.abs(np.asarray(data.quadrature.coefficients))
        amplitudes = np.asarray(resolve_lchs_coefficient_plan(chosen).prep_amplitudes()).real
        np.testing.assert_allclose(amplitudes, np.sqrt(raw/np.sum(raw)), rtol=1e-14, atol=0)
        recovery = chosen.reconstruction.recovery
        expected = (log2(np.linalg.norm(initial)) - chosen.reconstruction.scalar_decay/log(2.) + log2(np.sum(raw))
                    + log2(data.qsp_plan.recovery))
        assert recovery.exponent+log2(recovery.mantissa) == pytest.approx(expected, abs=1e-12)


def test_imaginary_time_filter_route_matches_the_source_sum_and_keeps_a_failed_fit(monkeypatch, tmp_path):
    """The requested H = 0 filter evolves exp(-t_a A) on every application, and a failed fit is reported once.

    A = .3 I + .4 X + .2 Z is Hermitian, so H = 0 and L = A, whose Pauli lower bound -.3 gives the PSD shift
    .3 of the kernel route. The filter needs no shift: with L = c_L I + L_0, c_L = .3, and the contraction
    normalization alpha of L_0, exp(-t A) = kappa exp[-beta (1 - X)] with kappa = exp[t (alpha - c_L)]
    (AT9-AT12, M7.5 with H_0 = 0). The independent sum is exp(-T A) u0 + sum_q w_q exp(-(T - r_q) A) b over
    NumPy's two-point Gauss-Legendre nodes r_q, from explicit matrices, with no global-phase alignment. Its
    difference from the solved vector is the filter's own tail and fit error, which the qsp_synthesis fact
    gamma*(algorithmic + child) bounds; the kernel stages are exactly zero on this route. Using the shifted
    c_L + s or the time T instead of t_a in kappa moves the sum by .55 and .23. The saved Plan, with its requested
    route and the filter expansion, reloads equal to the planned one.

    When the selected filter fit fails (here a phase-owner failure is injected at margin 1e-3), planning raises
    CoherentPreparationError with the row and the owner's error as cause, after exactly one selected
    preparation and one filter fit, so no other margin or route is tried. The case has 8 qubits.
    """
    from qiskit.quantum_info import SparsePauliOp
    from nwqlib._choice_archive import ArchiveFiles, load_plan, save_plan
    from nwqlib._limits import ResourceLimitError
    from nwqlib.algorithms.lchs import coherent_selection
    from nwqlib.operators import ingest_pauli
    from nwqlib.subroutines.qsp.phases import _QSPPhaseConvergenceError, _QSPPhaseLimitError
    A = ingest_pauli([("I", .3), ("X", .4), ("Z", .2)], num_qubits=1)
    matrix = SparsePauliOp.from_list(A.pauli_terms().labels()).to_matrix()
    initial, source, elapsed = np.array([.6, .8j]), np.array([.3j, -.2]), 2.5
    problem = LinearDynamics(A=A, initial_state=initial, source=source, time=elapsed)
    filtered = dict(hamiltonian_evolution_backend='qsp_block_encoding', duhamel_nodes=2)
    chosen = plan(problem, method=LCHS(**filtered, qsp_route={"implementation": "imaginary_gqet",
                                                               "parameters": {"level": 1/16}}))
    prepared = chosen._native["native_data"].qsp_plan
    assert prepared.candidate.label == "filter:convex_contraction:imaginary_gqet[0.0625]"
    assert sum(register.width for register in chosen.construction.program.registers) <= 8
    points, weights = np.polynomial.legendre.leggauss(2)
    expected = expm(-elapsed*matrix) @ initial + sum(
        elapsed/2*w*(expm(-(elapsed-elapsed/2*(1+x))*matrix) @ source) for x, w in zip(points, weights, strict=True))
    facts = {item.fact.quantity: item.fact.value.value for item in chosen.facts}
    assert facts["kernel_approximation"] == 0 and facts["k_quadrature"] == 0
    assert np.linalg.norm(solve(chosen).solution-expected) <= facts["qsp_synthesis"]+1e-12
    files = ArchiveFiles(tmp_path, 8_000_000)
    assert load_plan(save_plan(chosen, files), files) == chosen
    calls = {"route": 0, "filter": 0}
    route = coherent_selection.prepare_selected_coherent_route

    def fit(*args, **kwargs):
        raise _QSPPhaseConvergenceError("phase synthesis residual above its tolerance")

    def counted(name, function):
        def call(*args, **kwargs):
            calls[name] += 1
            return function(*args, **kwargs)
        return call

    monkeypatch.setattr(coherent_selection, "prepare_selected_coherent_route", counted("route", route))
    monkeypatch.setattr(coherent_selection, "prepare_imaginary_time_filter", counted("filter", fit))
    with pytest.raises(coherent_selection.CoherentPreparationError) as failure:
        plan(problem, method=LCHS(**filtered, qsp_route={"implementation": "imaginary_gqet",
                                                         "parameters": {"level": 1e-3}}))
    assert failure.value.candidate == "filter:convex_contraction:imaginary_gqet[0.001]"
    assert isinstance(failure.value.__cause__, _QSPPhaseConvergenceError)
    assert not isinstance(failure.value, ResourceLimitError)
    assert calls == {"route": 1, "filter": 1}
    # A stop at a capacity limit keeps the row and the cause and is a limit refusal, so compare gives a limited row.
    stop = _QSPPhaseLimitError("the next FFT does not fit", evidence="the attempts")

    def fit_limited(*args, **kwargs):
        raise stop
    monkeypatch.setattr(coherent_selection, "prepare_imaginary_time_filter", fit_limited)
    with pytest.raises(ResourceLimitError) as limited:
        plan(problem, method=LCHS(**filtered, qsp_route={"implementation": "imaginary_gqet",
                                                         "parameters": {"level": 1e-3}}))
    assert isinstance(limited.value, coherent_selection.CoherentPreparationError)
    assert limited.value.candidate == failure.value.candidate and limited.value.__cause__ is stop
    assert limited.value.evidence == "the attempts"


def test_explicit_mps_coefficient_prep_prepares_each_selected_factor():
    """lcu_state_preparation="mps_circuit" prepares the kernel and application factors as two MPS circuits.

    The problem and its independent finite sum are those of
    test_compact_source_applies_decay_and_scalar_phase_at_each_elapsed_time. The coefficient state is the tensor
    product of the kernel factor sqrt(|c_j|/beta) and the application factor sqrt(m_a/B_app), each on two
    qubits, so each has MPS bond dimension at most 2, which one layer of the layered circuit prepares exactly (Ran,
    arXiv:1908.07958v2, Sec. III); the default two layers then reproduce the direct-PREP sum within the same
    1e-4 QSP tolerance. A factor decomposed from the other factor, from the joint 16-entry table or from labels in
    another order would prepare a different coefficient state and miss the sum. The planning fact of the
    coefficient PREP stays unknown, because the TT-SVD discarded weight does not bound the circuit error.
    """
    from qiskit.quantum_info import SparsePauliOp
    from nwqlib.operators import ingest_pauli
    A = ingest_pauli([("I", 1.+.3j), ("X", .25), ("Z", .4j), ("Y", -.1)], num_qubits=1)
    matrix = SparsePauliOp.from_list(A.pauli_terms().labels()).to_matrix()
    l_part, h_part = (matrix+matrix.conj().T)/2, (matrix-matrix.conj().T)/(2j)
    initial, source, elapsed = np.array([.6, .8j]), np.array([.3j, -.2]), .7
    chosen = plan(LinearDynamics(A=A, initial_state=initial, source=source, time=elapsed),
                  method=LCHS(**GRID, lcu_state_preparation="mps_circuit"))
    data = chosen._native["native_data"]
    kernel, application = data.coefficient_mps
    assert (kernel.num_qubits, application.num_qubits) == (2, 2)
    facts = {item.fact.quantity: item.fact.value for item in chosen.facts}
    assert facts["lcu_coefficient_preparation"] is None
    nodes = np.array([0., .5, -1., -.5])
    coefficients = .5/(np.pi*(1+nodes**2))
    points, weights = np.polynomial.legendre.leggauss(2)
    applications = [(elapsed, 1., initial)] + [(elapsed-elapsed/2*(1+x), elapsed/2*w, source)
                                               for x, w in zip(points, weights, strict=True)]
    expected = sum(weight*c*np.exp(-.65*time)*(expm(-1j*time*(k*(l_part-.65*np.eye(2))+h_part))@vector)
                   for time, weight, vector in applications for k, c in zip(nodes, coefficients, strict=True))
    assert sum(register.width for register in chosen.construction.program.registers) <= 10
    np.testing.assert_allclose(solve(chosen).solution, expected, rtol=0, atol=1e-4)


def test_commuting_product_of_periodic_bands_recovers_the_exponential():
    """Periodic-band parts commute exactly, and the requested filter-times-H product recovers exp(-T A) u0.

    L = .7 I - .3 (S + S^dagger) and H = .2 i (S - S^dagger) are polynomials in the same cyclic shift S on four
    points, so the representation records [L, H] = 0 without any commutator. Then exp(-T A) =
    kappa exp[-beta (1 - X)] exp(-i T H_0), and the product route composes the imaginary-time filter of L_0 with
    a real-time evolution of H_0, independent success workspaces and the recovery sigma_I zeta_H (M7.5-M7.6). It
    has no kernel, so its finite target is exp(-T A) u0 itself, here from explicit 4x4 matrices with no
    global-phase alignment; the deviation must lie within the qsp_synthesis fact, which composes the factor
    errors as e_I + e_H + e_I e_H after scaling by gamma, plus 1e-12 for simulation rounding. Dropping the scalar
    kappa, applying exp(+i T H_0) or omitting either factor moves the vector by more than 0.1 here. The case has 8
    qubits.
    """
    from nwqlib.operators import ingest_band, ingest_cartesian
    shift = np.roll(np.eye(4), 1, 0)
    L = ingest_band((-1, 0, 1), (-.3, .7, -.3), num_qubits=2)
    H = ingest_band((-1, 1), (-.2j, .2j), num_qubits=2)
    matrix = .7*np.eye(4) - .3*(shift+shift.T) + 1j*(.2j*(shift-shift.T))
    initial = np.array([1., .5, 0., .2])
    problem = LinearDynamics(A=ingest_cartesian(dissipative=L, hamiltonian=H), initial_state=initial, time=1.)
    chosen = plan(problem, method=LCHS(hamiltonian_evolution_backend='qsp_block_encoding', qsp_route={
        "implementation": "commuting_product"}))
    data = chosen._native["native_data"]
    assert data.generator.parts_commute == "periodic_bands"
    assert data.qsp_plan.candidate.route == "commuting_product"
    assert sum(register.width for register in chosen.construction.program.registers) <= 8
    synthesis = next(item.fact.value.value for item in chosen.facts if item.fact.quantity == "qsp_synthesis")
    assert np.linalg.norm(solve(chosen).solution-expm(-matrix)@initial) <= synthesis+1e-12


def test_coherent_routes_are_compared_within_the_byte_limit_on_every_computer(tmp_path, monkeypatch):
    """Every computer with enough memory compares the same coherent routes, and less memory refuses the Plan.

    For the commuting band parts L = .7 I - .3 (S + S^T) and H = .2 i (S - S^T) on four points, the default
    planning bounds the shared QSVT targets of the kernel sum on a norming grid of 83160 bytes, although the
    selected commuting product needs less. Physical memories of 83160 and 10**6 bytes give the same Plan,
    which reopens unchanged, and each distinct target is formed once. 83159 bytes, physical or as an explicit
    max_bytes, refuse the Plan with that requirement after the first target, instead of comparing the routes
    that fit. An explicit max_bytes of 10**6 selects the same row with the same coefficients, construction
    evidence and refused rows, while its Method records the cap, so the Plan identity differs. The refused rows
    are the four routes not admitted for mathematical reasons.
    """
    import nwqlib._limits as limits
    from nwqlib._choice_archive import ArchiveFiles, load_plan, save_plan
    from nwqlib._limits import ResourceLimitError
    from nwqlib.algorithms.lchs import coherent_selection
    from nwqlib.operators import ingest_band, ingest_cartesian
    L = ingest_band((-1, 0, 1), (-.3, .7, -.3), num_qubits=2)
    H = ingest_band((-1, 1), (-.2j, .2j), num_qubits=2)
    problem = LinearDynamics(A=ingest_cartesian(dissipative=L, hamiltonian=H), initial_state=[1., .5, 0., .2], time=1.)
    targets, form = [], coherent_selection._qsp_phase_targets
    monkeypatch.setattr(coherent_selection, "_qsp_phase_targets",
                        lambda expansion, cap, max_bytes: targets.append((expansion.cos_coefficients, cap))
                        or form(expansion, cap, max_bytes))

    def planned(memory, **limit):
        monkeypatch.setattr(limits, "physical_memory_bytes", lambda: memory)
        targets.clear()
        return plan(problem, method=LCHS(hamiltonian_evolution_backend="qsp_block_encoding", **limit), seed=7)

    def chosen_row(chosen):
        selection = chosen._native["native_data"].qsp_plan.selection
        refused = [(row.alternative, row.route, row.reasons) for row in selection.candidates if row.status == "refused"]
        return (selection.selected.label, tuple(chosen._native["native_data"].quadrature.coefficients),
                chosen.construction.selections[0].semantics.approximation_evidence, refused)

    adequate = planned(10**6)
    assert len(targets) == len(set(targets)) == 8
    assert planned(83160).content_id == adequate.content_id
    label, _, _, refused = chosen_row(adequate)
    assert label.startswith("product:") and sorted(route for _, route, _ in refused) == sorted(
        coherent_selection.NOT_ADMITTED_ROUTES)
    assert all(reasons == (coherent_selection.NOT_ADMITTED_ROUTES[route],) for _, route, reasons in refused)
    files = ArchiveFiles(tmp_path, 4_000_000)
    assert load_plan(save_plan(adequate, files), files).content_id == adequate.content_id
    need = "QSP norming grid needs about 83160 bytes, exceeding "
    for memory, limit, source in ((83159, {}, r"the physical memory \(83159 bytes\)"),
                                  (10**6, {"max_bytes": 83159}, "max_bytes=83159")):
        with pytest.raises(ResourceLimitError, match=need + source):
            planned(memory, **limit)
        assert len(targets) == 1
    explicit = planned(10**6, max_bytes=10**6)
    assert chosen_row(explicit) == chosen_row(adequate)
    fields = {"max_bytes", "content_id"}
    assert explicit.method.model_dump(exclude=fields) == adequate.method.model_dump(exclude=fields)
    assert explicit.content_id != adequate.content_id


def test_compact_access_above_max_bytes_is_a_memory_refusal_not_an_inapplicable_method():
    """The Pauli Cartesian split of a compact A that needs more than max_bytes refuses planning and limits its row.

    For A = (.4+.2i)I + .2X + (.1+.3i)Z, the split into L and H needs about 3756 bytes. Under max_bytes=3755 the
    refusal states that requirement and is not an ApplicabilityError, so compare marks the row as limited instead
    of listing the Method as not applicable.
    """
    from nwqlib import compare, solve
    from nwqlib._limits import ResourceLimitError
    from nwqlib.algorithms.protocol import ApplicabilityError
    from nwqlib.operators import ingest_pauli
    problem = LinearDynamics(A=ingest_pauli([("I", .4+.2j), ("X", .2), ("Z", .1+.3j)], num_qubits=1),
                             initial_state=[.6, .8j], time=.5)
    method = LCHS(hamiltonian_evolution_backend="qsp_block_encoding", max_bytes=3755)
    (row,) = compare(problem, methods=[method]).rows
    assert row.plan is None and row.resource_limited
    for call in (lambda: plan(problem, method=method), lambda: solve(row)):
        with pytest.raises(ResourceLimitError, match="Pauli Cartesian parts needs about 3756 bytes") as raised:
            call()
        assert not isinstance(raised.value, ApplicabilityError)


def test_single_a_access_evolves_a_supplied_nonhermitian_encoding(tmp_path):
    """A general A given only as one supplied encoding uses the single-A generator and matches the finite sum.

    A = M + i D with Hermitian M and D has L = M and H = D. The supplied circuit is the Halmos unitary dilation of
    the contraction A, an exact encoding with alpha = 1, whose asserted bound ||A|| <= 1 gives the PSD shift
    s = 1 of L. The single-A generator encodes k_j L + H from the one encoding through the coefficients
    k_j - i (G13), so the independent finite sum is sum_j c_j exp(s T) exp(-i T (k_j (L + s I) + H)) u0 on the
    Cauchy coefficients c_j = .5/(pi (1 + k_j**2)) of the signed-binary grid, from explicit matrices with no
    global-phase alignment. The deviation must lie within the qsp_synthesis fact plus 1e-12, here 7.9e-5.
    Treating the encoded block as L alone, or dropping the -i of the coefficients, changes the sum by 8.8e-3, and
    treating it as H alone changes it by 3.3e-2. The saved Plan, with the supplied encoding and the
    complex coefficient table, reloads equal to the planned one.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from scipy.linalg import sqrtm
    from nwqlib import NormSquared
    from nwqlib._choice_archive import ArchiveFiles, load_plan, save_plan
    from nwqlib.algorithms.lchs.primary_records import FIXED_SHOT_RULES
    from nwqlib.core.records import InputRef, Source
    from nwqlib.operators import ingest_block_encoding
    from nwqlib.subroutines.block_encoding.core import BlockEncoding
    hermitian, defect = np.array([[.3, .2-.1j], [.2+.1j, -.4]]), np.array([[.01, .02], [.02, -.01]])
    a = hermitian+1j*defect
    unitary = np.block([[a, sqrtm(np.eye(2)-a@a.conj().T)], [sqrtm(np.eye(2)-a.conj().T@a), -a.conj().T]])
    order = np.arange(4).reshape(2, 2).T.reshape(-1)  # the ancilla is the low qubit
    circuit = QuantumCircuit(2)
    circuit.append(UnitaryGate(unitary[np.ix_(order, order)]), circuit.qubits)
    supplied = ingest_block_encoding(
        BlockEncoding(circuit=circuit, alpha=1., num_ancillas=1, system_qubits=1, error_bound=0.,
                      implementation="supplied", metadata={}),
        reference=InputRef(identity="user:A", representation="block_encoding",
                           source=Source(name="user", version="1", domain="test", reference="declared")),
        hermitian=False)
    initial, elapsed = np.array([.6, .8j]), .5
    chosen = plan(LinearDynamics(A=supplied, initial_state=initial, time=elapsed), method=LCHS(
        hamiltonian_evolution_backend='qsp_block_encoding', lchs_kernel=ProviderConfig(implementation='cauchy_density'),
        k_quadrature=ProviderConfig(implementation='signed_binary_uniform',
                                    parameters={'num_qubits': 2, 'lsb_position': -1})))
    prepared = chosen._native["native_data"].qsp_plan
    assert prepared.candidate.generator.startswith("single_a/")
    shift = chosen.reconstruction.psd_shift
    assert shift == pytest.approx(1., rel=1e-12)
    nodes = np.array([0., .5, -1., -.5])
    coefficients = .5/(np.pi*(1+nodes**2))
    expected = sum(c*np.exp(shift*elapsed)*(expm(-1j*elapsed*(k*(hermitian+shift*np.eye(2))+defect))@initial)
                   for k, c in zip(nodes, coefficients, strict=True))
    assert sum(register.width for register in chosen.construction.program.registers) <= 10
    synthesis = next(item.fact.value.value for item in chosen.facts if item.fact.quantity == "qsp_synthesis")
    assert np.linalg.norm(solve(chosen).solution-expected) <= synthesis+1e-12
    files = ArchiveFiles(tmp_path, 8_000_000)
    assert load_plan(save_plan(chosen, files), files) == chosen
    # The rows of this input have unknown CX counts, so fixed shots rank them by original-U calls, and the record says so.
    fixed = plan(LinearDynamics(A=supplied, initial_state=initial, time=elapsed), method=LCHS(), output=NormSquared(),
                 shots=256, seed=1)
    prepared_route = fixed._native["native_data"].qsp_plan
    assert prepared_route.candidate.cost.get("objective") is None
    assert fixed.reconstruction.fixed_shot_precision.ranking == "original_u_calls"
    assert prepared_route.candidate.cost["original_u_calls"] == min(
        row.cost["original_u_calls"] for row in prepared_route.selection.candidates if row.status == "admitted" and row.requested)
    assert FIXED_SHOT_RULES["original_u_calls"] in str(fixed)


def test_sampling_accuracy_plans_the_counts_whose_radius_meets_the_request(tmp_path):
    """compare(..., accuracy=...) plans each coherent route at the shots whose reported sampling radius meets it.

    The literal default backend takes the coherent route for this compact A. For the QuadraticForm of
    O = Z + .5 X + .3 I the two counts settings {Z, I} and {X} score within Gamma**2 * 1.3 and Gamma**2 * .5, the
    identity counted once (M7.13), so with N shots per setting and confidence c the reported radius is
    R(N) = Gamma**2 * 1.8 * sqrt(2 log(2 * 2/(1 - c))/N) (M7.14, the radius of analysis._counts_radius). Each
    route's N must be the least integer with R(N) <= epsilon at that route's own recovery Gamma, here computed in
    floating point from the Plan's recovery, independently of the planner's exact rational and Decimal count.
    Leaving the identity out of the range gives a smaller N whose radius exceeds epsilon. The run of the first
    row reports its radius from the actual returned shots at the requested confidence, and the Plan reloads
    with its request.
    """
    from nwqlib import Accuracy, QuadraticForm, compare
    from nwqlib._choice_archive import ArchiveFiles, load_plan, save_plan
    from nwqlib.operators import ingest_pauli
    kernel = dict(lchs_kernel=ProviderConfig(implementation='cauchy_density'),
                  k_quadrature=ProviderConfig(implementation='signed_binary_uniform',
                                              parameters={'num_qubits': 2, 'lsb_position': -1}))
    A = ingest_pauli([("I", 1.+.3j), ("X", .25), ("Z", .4j), ("Y", -.1)], num_qubits=1)
    problem = LinearDynamics(A=A, initial_state=np.array([.6, .8j]), time=.7)
    output = QuadraticForm(observable=ingest_pauli([("Z", 1.), ("X", .5), ("I", .3)], num_qubits=1))
    accuracy, epsilon = Accuracy(absolute_tolerance=.05, confidence=.9, component="sampling"), .05
    comparison = compare(problem, methods=(LCHS(**kernel), LCHS(hamiltonian_evolution_backend='qsp_block_encoding',
                                                                 qsp_route="shared_raw", **kernel)),
                         output=output, accuracy=accuracy, seed=3)
    for row in comparison.rows:
        chosen = row.plan
        assert row.reason is None and chosen.selection_accuracy == accuracy
        assert chosen.reconstruction.selected_backend == "qsp_block_encoding"
        gamma = chosen.reconstruction.recovery.as_float()

        def radius(shots, total=1.8):
            return gamma**2*total*math.sqrt(2*math.log(4/.1)/shots)
        assert radius(chosen.shots) <= epsilon*(1+1e-12) and (chosen.shots == 1 or radius(chosen.shots-1) > epsilon)
        # Without the identity's range the count would be smaller, and its radius at the actual range too wide.
        omitted = math.ceil(2*gamma**4*1.5**2*math.log(4/.1)/epsilon**2)
        assert omitted < chosen.shots and radius(omitted) > epsilon
    chosen = comparison.rows[0].plan
    files = ArchiveFiles(tmp_path, 8_000_000)
    restored = load_plan(save_plan(chosen, files), files)
    assert restored == chosen and restored.selection_accuracy == accuracy
    result = solve(restored)
    sampling = next(item for item in result.facts if item.fact.quantity == "sampling")
    assert sampling.failure_probability == pytest.approx(.1, rel=1e-15)
    assert 0 < sampling.fact.value.value <= epsilon*(1+1e-12)


def test_analytic_kernel_prep_is_refused_outside_the_coherent_low_somma_route(monkeypatch):
    """lcu_state_preparation="analytic" prepares only the Low-Somma kernel factor of the coherent LCU.

    The closed form it evaluates, exp(-k**2/(8 gamma**2))/sqrt(1 + k**2), is the kernel factor
    sqrt(|c_j|/beta_k) only for low_somma_f2 on its equal-weight symmetric_uniform_trapezoid grid, and only
    the qsp_block_encoding route has a separate kernel factor. Another pair or an explicit other backend is
    refused when the Method is built. For a dense A, "auto" chooses dense_exact, so planning refuses it with
    ApplicabilityError, which compare records as a rejected row. Sampled QSP pairs prepare no coefficient
    state and draw from one concrete pair, so the analytic PREP and provider enumeration, which their Method
    accepts, are refused at planning, for fixed shots and for an accuracy request alike, before the
    provider-independent stage (`selection.compact_common`) runs, and compare keeps the rejected rows. The same
    holds for an MPS initial-state preparation on the evolving sampled QSP and product-formula routes, which
    prepare each input with its own preparation and would otherwise plan it as "direct".
    """
    from nwqlib import Accuracy, NormSquared, compare
    from nwqlib.algorithms.lchs import selection
    from nwqlib.algorithms.protocol import ApplicabilityError
    from nwqlib.operators import ingest_pauli
    analytic = dict(lchs_kernel="low_somma_f2", k_quadrature="symmetric_uniform_trapezoid",
                    lcu_state_preparation="analytic")
    with pytest.raises(ValueError, match="'auto' or 'qsp_block_encoding'"):
        LCHS(**analytic, hamiltonian_evolution_backend="dense_exact")
    with pytest.raises(ValueError, match="not 'near_optimal_eq7' with 'composite_gauss'"):
        LCHS(lcu_state_preparation="analytic")
    problem = LinearDynamics(A=np.array([[1., .2], [.1, .5]]), initial_state=[1., 0.], time=.3)
    with pytest.raises(ApplicabilityError, match="chose 'dense_exact'"):
        plan(problem, method=LCHS(**analytic))
    sampled = dict(observable_estimator="sampled_pairs", hamiltonian_evolution_backend="qsp_block_encoding")
    compact = LinearDynamics(A=ingest_pauli([("Z", .4j)], num_qubits=1), initial_state=[.6, .8], time=1.)
    monkeypatch.setattr(selection, "compact_common", lambda *a, **k: pytest.fail("common-stage work before refusal"))
    methods = [LCHS(**analytic, **sampled), LCHS(lchs_kernel="auto", k_quadrature="auto", **sampled),
               LCHS(initial_state_preparation="mps_circuit", **sampled),
               LCHS(observable_estimator="sampled_pairs", hamiltonian_evolution_backend="trotter", trotter_steps=2,
                    initial_state_preparation="mps_circuit")]
    reasons = ("prepares the coefficient state", "compares the providers") + ("initial_state_preparation",) * 2
    for method, reason in zip(methods, reasons):
        with pytest.raises(ApplicabilityError, match=reason):
            plan(compact, method=method, output=NormSquared(), shots=4)
        with pytest.raises(ApplicabilityError, match=reason):
            plan(compact, method=method, output=NormSquared(), accuracy=Accuracy(absolute_tolerance=.1,
                                                                                  component="sampling"))
    rows = compare(compact, methods=methods, output=NormSquared(), shots=4).rows
    assert all(row.plan is None and reason in row.reason for row, reason in zip(rows, reasons))


def test_provider_enumeration_prepares_only_the_chosen_grid(monkeypatch, tmp_path):
    """lchs_kernel="auto" with k_quadrature="auto" compares the finite provider frontier and prepares one grid.

    The frontier (selection.PROVIDER_FRONTIER) has at most 12 entries, starting with the default pair, and a
    sampling-accuracy request adds the stationary Apel-Berntson radius. Each built entry is ranked by the
    sufficient work N G C of its own selected row, so the selected record has the smallest key, the earliest on a
    tie, and the accuracy hook and planning choose the same grid: the Plan's shots are the chosen row's count.
    Exactly one coherent route is prepared for the "auto" Method, which keeps its request while the Plan's
    coefficient plan names the chosen pair, and the Plan reloads equal. The independent counterpart is the
    explicit Method with the winner's public kernel and quadrature parameters: its own one-candidate planning must
    select the same row with the same recovery and shots. A concrete backend other than the coherent one is
    refused when the Method is built, and a dense A, for which "auto" chooses dense_exact, at planning.
    """
    from nwqlib import Accuracy, QuadraticForm, compare
    from nwqlib._choice_archive import ArchiveFiles, load_plan, save_plan
    from nwqlib.algorithms.lchs import coherent_selection
    from nwqlib.algorithms.protocol import ApplicabilityError
    from nwqlib.operators import ingest_pauli
    auto = dict(lchs_kernel="auto", k_quadrature="auto")
    with pytest.raises(ValueError, match="'auto' or 'qsp_block_encoding'"):
        LCHS(**auto, hamiltonian_evolution_backend="trotter")
    with pytest.raises(ValueError, match="together"):
        LCHS(lchs_kernel="auto")
    with pytest.raises(ApplicabilityError, match="backend 'dense_exact'"):
        plan(LinearDynamics(A=np.array([[1., .2], [.1, .5]]), initial_state=[1., 0.], time=.3), method=LCHS(**auto))
    A = ingest_pauli([("I", 1.+.3j), ("X", .25), ("Z", .4j), ("Y", -.1)], num_qubits=1)
    problem = LinearDynamics(A=A, initial_state=np.array([.6, .8j]), time=.7)
    output = QuadraticForm(observable=ingest_pauli([("Z", 1.), ("X", .5), ("I", .3)], num_qubits=1))
    prepared = []
    route = coherent_selection.prepare_selected_coherent_route
    monkeypatch.setattr(coherent_selection, "prepare_selected_coherent_route",
                        lambda selection: prepared.append(selection.selected.label) or route(selection))
    from nwqlib.algorithms.lchs import quantum, selection
    calls = dict(route=0, choice=0)
    for owner, name, key in ((selection, "_route", "route"), (quantum, "choose_provider", "choice")):
        monkeypatch.setattr(owner, name, lambda *a, _o=getattr(owner, name), _k=key, **k: (
            calls.__setitem__(_k, calls[_k] + 1), _o(*a, **k))[1])
    accuracy = Accuracy(absolute_tolerance=.05, confidence=.9, component="sampling")
    comparison = compare(problem, methods=(LCHS(**auto),), output=output, seed=3, accuracy=accuracy)
    # The accuracy hook and planning share one route and one provider choice.
    assert calls == dict(route=1, choice=1)
    row = comparison.rows[0]
    assert row.reason is None
    chosen = row.plan
    data = chosen._native["native_data"]
    records = data.provider_selection
    labels = [item["label"] for item in records]
    assert len(set(labels)) == len(labels) <= 12 and labels[0] == "acl" and "apel_berntson_kk6" in labels
    assert {item["status"] for item in records} <= {"selected", "admitted", "refused", "not_generated", "duplicate"}
    ranked = [item for item in records if item["key"] is not None]
    winner = min(ranked, key=lambda item: item["key"])
    assert [item["label"] for item in records if item["status"] == "selected"] == [winner["label"]]
    parameters = winner["parameters"]
    pair = data.quadrature.coefficient_plan
    assert (pair.resolved_lchs_kernel.implementation, pair.resolved_k_quadrature.implementation) == (
        parameters["lchs_kernel"], parameters["k_quadrature"])
    assert prepared == [winner["row"]] == [data.qsp_plan.candidate.label]
    assert chosen.shots == data.qsp_plan.candidate.cost["shots"]
    assert chosen.method.lchs_kernel.implementation == "auto"
    files = ArchiveFiles(tmp_path, 50_000_000)
    assert load_plan(save_plan(chosen, files), files) == chosen
    # The winner of this input needs no private grid seam, so an explicit Method can name it.
    assert all(value is None for value in parameters["seams"].values())
    explicit = LCHS(lchs_kernel=ProviderConfig(implementation=parameters["lchs_kernel"], parameters={
        name: value for name, value in parameters["kernel_parameters"].items() if value is not None}),
        k_quadrature=ProviderConfig(implementation=parameters["k_quadrature"],
                                    parameters=dict(parameters["quadrature_parameters"])))
    counterpart = plan(problem, method=explicit, output=output, accuracy=accuracy, seed=3)
    selected = counterpart._native["native_data"].qsp_plan.candidate
    assert selected.label == winner["row"] and prepared == [winner["row"]] * 2
    assert counterpart.shots == chosen.shots and counterpart.reconstruction.recovery == chosen.reconstruction.recovery
    assert counterpart._native["native_data"].provider_selection == ()


def test_provider_records_copy_the_scalars_of_each_built_grid():
    """A built grid's provider record copies that grid's scalars, and a pair above a work limit refuses the choice.

    The winner's record equals the scalars of the winning grid that the choice carries, another built entry
    differs from the winner, and every pair of this 1-qubit problem builds its grid. With max_quadrature_work =
    4000 the default pair's grid needs 8449 work units, so the choice stops with that requirement instead of
    recording the pair as refused and choosing among the other pairs. This is selection only: no phase is
    prepared.
    """
    from nwqlib import NormSquared
    from nwqlib._limits import ResourceLimitError
    from nwqlib.algorithms.lchs import selection
    from nwqlib.algorithms.lchs.quantum import choose_provider
    from nwqlib.operators import ingest_pauli
    problem = LinearDynamics(A=ingest_pauli([("I", .3), ("X", .2), ("Z", .2j)], num_qubits=1), initial_state=[1, 0],
                             time=.5)
    names = ("range_k", "interval_count_each_side", "node_count", "total_node_count", "h1", "coefficient_l1_norm")
    bounds = ("approximate_lchs_error_bound", "quadrature_error_bound")
    method = LCHS(lchs_kernel="auto", k_quadrature="auto")
    choice = choose_provider(method, problem, selection.compact_common(method, problem), output=NormSquared(),
                             shots=1000)
    records, grid = choice.data.provider_selection, choice.data.quadrature
    winner = next(record for record in records if record["status"] == "selected")
    assert all(winner[name] == getattr(grid, name) for name in names)
    assert all(winner[name] == getattr(grid.coefficient_plan.quadrature, name) for name in bounds)
    assert all(record["status"] in ("admitted", "selected") and record["range_k"] is not None for record in records)
    assert any(record["range_k"] != winner["range_k"] or record["node_count"] != winner["node_count"]
               for record in records if record is not winner)
    limited = LCHS(lchs_kernel="auto", k_quadrature="auto", max_quadrature_work=4000)
    with pytest.raises(ResourceLimitError, match="require 8449 work units, exceeding LCHS.max_quadrature_work=4000"):
        choose_provider(limited, problem, selection.compact_common(limited, problem), output=NormSquared(), shots=1000)


def test_route_selection_fits_no_phase_and_never_replaces_a_refused_request(monkeypatch):
    """Coherent selection fits no phase and builds no circuit, and a refused requested route is never replaced.

    A = (.4+.2i)I + .2X + (.1+.3i)Z on the four-node Cauchy grid has a Hermitian joint generator target with a full
    involution, so the free choice has admitted rows of several families. An explicit qsp_route="berry", a route
    listed as not admitted, raises ApplicabilityError rather than selecting an admitted route of another family.
    """
    from qiskit import QuantumCircuit
    from nwqlib import Solution
    from nwqlib.algorithms.lchs import coherent_selection
    from nwqlib.algorithms.lchs.quantum import choose_provider
    from nwqlib.algorithms.lchs.selection import _route
    from nwqlib.algorithms.protocol import ApplicabilityError
    from nwqlib.operators import ingest_pauli
    from nwqlib.subroutines.qsp import phases

    def forbidden(*args, **kwargs):
        pytest.fail("route selection fitted phases, prepared a route or built a circuit")
    for owner, name in ((phases, "_solve"), (phases, "solve_wx_qsp_phases"), (phases, "solve_laurent_qsp_phases"),
                        (coherent_selection, "prepare_selected_coherent_route"), (QuantumCircuit, "__init__")):
        monkeypatch.setattr(owner, name, forbidden)
    problem = LinearDynamics(A=ingest_pauli([("I", .4+.2j), ("X", .2), ("Z", .1+.3j)], num_qubits=1),
                             initial_state=np.array([1., 0.]), time=.8)
    base = dict(hamiltonian_evolution_backend='qsp_block_encoding', lchs_kernel='cauchy_density',
                k_quadrature=GRID['k_quadrature'], max_bytes=32*2**20)

    def select(method):
        route, common = _route(method, problem, output=Solution(), execution='quantum')
        assert route == 'coherent'
        return choose_provider(method, problem, common, output=Solution(), shots=None).selection
    free = select(LCHS(**base))
    rows = [row for row in free.candidates if row.status == 'admitted' and row.alternative == 'lchs']
    generator = free.selected.handles['generator']
    assert generator.target_operator_is_hermitian and generator.full_unitary_involution
    assert len({row.route for row in rows}) > 1
    with pytest.raises(ApplicabilityError):
        select(LCHS(**base, qsp_route='berry'))


def _ring_laplacian(q):
    """The Laplacian L = (2 I - S - S^T)/4 of a ring of 2**q points, as a Pauli sum and as a matrix.

    Its eigenvalues (1 - cos(2 pi j/2**q))/2 lie in [0, 1], so L is positive semidefinite, which its
    Pauli coefficients do not show: at q = 3 the structural bound c_I - sum |c_P| is -.25.
    """
    from qiskit.quantum_info import SparsePauliOp
    from nwqlib.operators import ingest_pauli
    shift = np.roll(np.eye(2**q), 1, axis=0)
    matrix = .25*(2*np.eye(2**q)-shift-shift.T)
    terms = SparsePauliOp.from_operator(matrix)
    return ingest_pauli(tuple(zip(terms.paulis.to_labels(), terms.coeffs, strict=True)), num_qubits=q), matrix


_KERNEL_ROUTE = dict(hamiltonian_evolution_backend="qsp_block_encoding", approximation_tolerance=.1,
                     qsp_route={"implementation": "shared_oaa", "parameters": {"level": .999}})


def _component_sum(chosen):
    """The sum of the planned physical error components, each a concrete fact of this Plan."""
    assert all(item.fact.availability == "concrete" for item in chosen.facts)
    return math.fsum(item.fact.value.value for item in chosen.facts)


def test_declared_lower_bound_of_a_ring_laplacian_reduces_the_shift_on_a_kernel_route():
    """A true declaration L >= 0 removes the shift of a Pauli ring Laplacian and keeps the solution of expm.

    The oracle is scipy.linalg.expm of the 8-dimensional L at T = 5. The kernel route's solution lies within
    the planned component bounds, plus 1e-12 for simulation rounding, as in the other compact routes of this
    file. Without the declaration the structural bound -.25 gives the shift .25, and the recovery is larger by
    about exp(.25 T) = 3.49 (only the shift differs up to the kernel grid). The shots of an accuracy request are
    the least N with Gamma**2 sqrt(log(2/delta)/(2 N)) <= epsilon, the two-sided Hoeffding count of a Bernoulli
    norm estimate with Gamma from each Plan's own recovery, so their unrounded counts scale as Gamma**4.
    Both the premise of the record and each fact that uses the shift stay conditional on the declaration.
    """
    from nwqlib import Accuracy, NormSquared
    from nwqlib.algorithms.lchs.primary_records import ASSERTED_L_BOUNDS
    from nwqlib.operators import ingest_cartesian
    L, matrix = _ring_laplacian(3)
    initial = np.eye(8)[0]
    problems = dict(structural=LinearDynamics(A=L, initial_state=initial, time=5.),
                    declared=LinearDynamics(A=ingest_cartesian(L, dissipative_lower_bound=0.), initial_state=initial,
                                            time=5.))
    chosen = {name: plan(problem, method=LCHS(**_KERNEL_ROUTE), seed=1) for name, problem in problems.items()}
    # rel=1e-14 allows the outward rounding of one binary64 sum over the seven coefficients of this ring.
    assert chosen["structural"].reconstruction.psd_shift == pytest.approx(.25, rel=1e-14)
    assert chosen["structural"].reconstruction.psd_premise == "certified_bound"
    declared = chosen["declared"]
    rec = declared.reconstruction
    assert (rec.psd_shift, rec.scalar_decay, rec.psd_premise) == (0., 0., "user_assertion")
    assert declared._native["native_data"].qsp_plan.candidate.route == "shared_oaa"
    assert ASSERTED_L_BOUNDS in declared.assumptions
    assert all(ASSERTED_L_BOUNDS in item.fact.assumptions for item in declared.facts)
    assert not any(ASSERTED_L_BOUNDS in item.fact.assumptions for item in chosen["structural"].facts)
    # exp(.25*5) = 3.49 is the ratio of the recoveries when only the shift differs, so 3 leaves room for the kernel grid.
    assert rec.recovery.as_float() < chosen["structural"].reconstruction.recovery.as_float()/3
    result = solve(declared)
    assert np.linalg.norm(result.solution-expm(-5.*matrix)@initial) <= _component_sum(declared)+1e-12
    assert ASSERTED_L_BOUNDS in next(item for item in result.facts
                                     if item.fact.quantity == "algorithmic_approximation").fact.assumptions
    assert ASSERTED_L_BOUNDS in str(declared) and ASSERTED_L_BOUNDS in str(result)
    epsilon, delta = .01, .05
    accuracy = Accuracy(absolute_tolerance=epsilon, component="sampling", confidence=1-delta)
    for problem in problems.values():
        sized = plan(problem, method=LCHS(**_KERNEL_ROUTE), output=NormSquared(), accuracy=accuracy, seed=1)
        gamma = sized.reconstruction.recovery.as_float()

        def radius(shots):
            return gamma**2*math.sqrt(math.log(2/delta)/(2*shots))
        # 1e-12 allows the binary64 evaluation of this radius, as the planner rounds its exact count upward.
        assert radius(sized.shots) <= epsilon*(1+1e-12) and radius(sized.shots-1) > epsilon


@pytest.mark.parametrize("terms, declared, shift, decay", (
    ([("I", .9), ("X", .25), ("Z", .25)], .5, 0., .5),
    ([("I", .3), ("X", .4), ("Z", .2)], -.2, .2, 0.)))
def test_declared_lower_bound_recovers_each_source_duration_and_real_growth(terms, declared, shift, decay):
    """A valid declaration keeps the source solution, with its own factor exp(-ell (T - v)) for each duration.

    A = L + iH with H = .1 Z, and L = .9 I + .25 X + .25 Z (lowest eigenvalue .9 - sqrt(.125) = .546, known decay)
    or L = .3 I + .4 X + .2 Z (lowest eigenvalue .3 - sqrt(.2) = -.147, real growth). The structural bounds are .4
    and -.3, and each declaration lies between the structural bound and the true eigenvalue. Two oracles apply. The
    closed form exp(-A T) u0 + A^{-1} (I - exp(-A T)) b from explicit matrices must lie within the planned components,
    on the default kernel. On the small Cauchy grid of GRID, the finite sum over applications a with durations t_a,
    weights w_a and own inputs of w_a c_j exp(-ell t_a) exp(-i t_a (k_j (L - ell I) + H)) psi_a must lie within the
    qsp_synthesis fact, plus 1e-12 for simulation rounding. One overall factor exp(-ell T) on every application,
    which the source relation forbids, moves that sum by more than 20 times that allowance, which this case checks.
    """
    from qiskit.quantum_info import SparsePauliOp
    from nwqlib.algorithms.lchs import LCHSRefinement
    from nwqlib.algorithms.lchs.primary_records import ASSERTED_L_BOUNDS
    from nwqlib.operators import ingest_cartesian, ingest_pauli
    L, H = ingest_pauli(terms, num_qubits=1), ingest_pauli([("Z", .1)], num_qubits=1)
    l_part, h_part = SparsePauliOp.from_list(terms).to_matrix(), .1*np.diag([1., -1.]).astype(complex)
    matrix = l_part+1j*h_part
    initial, source, elapsed = np.array([.6, .8j]), np.array([.3j, -.2]), 1.
    problem = LinearDynamics(A=ingest_cartesian(L, H, dissipative_lower_bound=declared), initial_state=initial,
                             source=source, time=elapsed)
    coarse = plan(problem, method=LCHS(hamiltonian_evolution_backend="qsp_block_encoding", duhamel_nodes=2,
                                       approximation_tolerance=.05, qsp_route="shared_raw"))
    rec = coarse.reconstruction
    assert (rec.psd_shift, rec.scalar_decay, rec.psd_premise) == (shift, decay, "user_assertion")
    propagator = expm(-elapsed*matrix)
    closed_form = propagator@initial+np.linalg.solve(matrix, (np.eye(2)-propagator)@source)
    solved = solve(coarse)
    assert np.linalg.norm(solved.solution-closed_form) <= _component_sum(coarse)+1e-12
    _, refined = solved.verify(checks=LCHSRefinement(components=("duhamel",)))
    assert ASSERTED_L_BOUNDS in next(item for item in refined if item.fact.quantity == "duhamel_quadrature").fact.assumptions

    chosen = plan(problem, method=LCHS(**GRID))
    nodes = np.array([0., .5, -1., -.5])
    coefficients = .5/(np.pi*(1+nodes**2))
    points, weights = np.polynomial.legendre.leggauss(2)
    applications = [(elapsed, 1., initial)] + [(elapsed-elapsed/2*(1+x), elapsed/2*w, source)
                                               for x, w in zip(points, weights, strict=True)]

    def finite_sum(recovery):
        return sum(weight*c*recovery(time)*(expm(-1j*time*(k*(l_part-declared*np.eye(2))+h_part))@vector)
                   for time, weight, vector in applications for k, c in zip(nodes, coefficients, strict=True))
    expected = finite_sum(lambda time: np.exp(-declared*time))
    allowance = next(item.fact.value.value for item in chosen.facts if item.fact.quantity == "qsp_synthesis")+1e-12
    assert np.linalg.norm(solve(chosen).solution-expected) <= allowance
    # 20 is a margin between the forbidden answer and the allowance, not a mathematical constant of the relation.
    assert np.linalg.norm(finite_sum(lambda time: np.exp(-declared*elapsed))-expected) > 20*allowance


def test_declaration_that_does_not_improve_keeps_the_structural_selection_and_evidence():
    """No declaration, and one at or below the structural bound, plan the same route, grid and evidence as a plain Pauli L.

    The three inputs differ only in the operator identifier, which the declaration enters. A negative zero names
    the same bound as zero, so it has the same identifier. make_l_psd=False excludes a positive shift, so it refuses a
    Pauli L whose bound stays negative and accepts a nonnegative declaration, including the extraction of known decay.

    For Hermitian L the smallest eigenvalue is at most the mean eigenvalue tr(L)/D, with equality only for a scalar L.
    L = .5 I + .3 X has the eigenvalues .2 and .8 and the mean .5, so .5, .79 and .8 - 1e-9 contradict the premise and
    are refused, and the true bound .2 plans. .25 is false but lies below the mean, so it plans: this comparison
    detects only what the mean contradicts. .5 I, and a Pauli sum or band whose non-identity rows cancel on ingestion,
    are scalar with mean .5, so the declaration .5 is true and plans with ell = .5, which is the known decay
    max(0, ell). A supplied encoding gives no trace to compare, so ingest_cartesian refuses a declaration for it: U = I
    with alpha = .5 is an exact encoding of .5 I, whose stored identity coefficient 0 is not its mean.
    """
    from qiskit import QuantumCircuit
    from nwqlib.algorithms.protocol import ApplicabilityError
    from nwqlib.core.records import InputRef, Source
    from nwqlib.operators import ingest_band, ingest_block_encoding, ingest_cartesian, ingest_pauli
    from nwqlib.subroutines.block_encoding.core import BlockEncoding
    L, _ = _ring_laplacian(3)
    initial = np.eye(8)[0]

    def planned(A, **options):
        return plan(LinearDynamics(A=A, initial_state=initial, time=1.), method=LCHS(**_KERNEL_ROUTE, **options), seed=1)
    plain = planned(L)
    for A in (ingest_cartesian(L), ingest_cartesian(L, dissipative_lower_bound=-1.)):
        chosen = planned(A)
        data, reference = chosen._native["native_data"], plain._native["native_data"]
        assert chosen.reconstruction.psd_premise == "certified_bound" and chosen.reconstruction == plain.reconstruction
        assert data.qsp_plan.candidate.label == reference.qsp_plan.candidate.label
        np.testing.assert_array_equal(data.quadrature.coefficients, reference.quadrature.coefficients)
    identity = {value: ingest_cartesian(L, dissipative_lower_bound=value).reference.identity for value in (0., -0., .1)}
    assert identity[0.] == identity[-0.] != identity[.1] != ingest_cartesian(L).reference.identity
    tilted = ingest_pauli([("I", .5), ("X", .3)], num_qubits=1)

    def stated(part, value):
        return plan(LinearDynamics(A=ingest_cartesian(part, dissipative_lower_bound=value),
                                   initial_state=np.eye(part.basis.dimension)[0], time=1.),
                    method=LCHS(hamiltonian_evolution_backend="qsp_block_encoding")).reconstruction
    # compact_common converts the ValueError of cartesian_access. A plain ValueError from later planning is not this refusal.
    for value in (.5, .79, .8-1e-9):
        with pytest.raises(ApplicabilityError):
            stated(tilted, value)
    assert stated(tilted, .2).scalar_decay == .2
    assert stated(tilted, .25).psd_premise == "user_assertion"
    for part in (ingest_pauli([("I", .5)], num_qubits=1), ingest_pauli([("I", .5), ("X", .3), ("X", -.3)], num_qubits=1),
                 ingest_band([0, 1, 5], [.5, .3, -.3], num_qubits=2)):
        assert stated(part, .5).scalar_decay == .5
    assert planned(ingest_cartesian(L, dissipative_lower_bound=0.), make_l_psd=False).reconstruction.psd_shift == 0
    for A in (L, ingest_cartesian(L, dissipative_lower_bound=-.1)):
        with pytest.raises(ApplicabilityError):
            planned(A, make_l_psd=False)
    decay = plan(LinearDynamics(A=ingest_cartesian(ingest_pauli([("I", .9), ("X", .25), ("Z", .25)], num_qubits=1),
                                                   dissipative_lower_bound=.5), initial_state=[.6, .8], time=1.),
                 method=LCHS(hamiltonian_evolution_backend="qsp_block_encoding", make_l_psd=False)).reconstruction
    assert (decay.psd_shift, decay.scalar_decay, decay.psd_premise) == (0., .5, "user_assertion")
    supplied = ingest_block_encoding(
        BlockEncoding(circuit=QuantumCircuit(1), alpha=.5, num_ancillas=0, system_qubits=1, error_bound=0.,
                      implementation="supplied", metadata={}),
        reference=InputRef(identity="user:scalar", representation="block_encoding",
                           source=Source(name="user", version="1", domain="test", reference="declared")),
        hermitian=True, lower_bound=.5)
    for bad in (dict(dissipative=None, hamiltonian=L, dissipative_lower_bound=0.),
                dict(dissipative=L, dissipative_lower_bound=float("nan")),
                dict(dissipative=supplied, dissipative_lower_bound=.5)):
        with pytest.raises(ValueError):
            ingest_cartesian(**bad)


def test_fixed_shots_report_the_precision_of_the_route_chosen_by_cx(tmp_path):
    """Under fixed shots the record, print(plan) and print(result) give the radius factor of the route chosen by CX.

    The ranking stays CX per circuit, so with the declaration L >= 0 the filter is still chosen, whose recovery
    comes from the normalization of its encoding and does not gain from the declaration: it equals the recovery
    without the declaration. The kernel route's recovery does gain, and an independent Plan that requests it
    gives the same Gamma as the record's smallest row. The factor is the square of the ratio of the two recoveries,
    since the Hoeffding radius at fixed shots and statistical model is proportional to Gamma**2. A success-conditional
    output keeps the same record, read as the inverse factor on its accepted probability at equal q. Without the
    declaration the filter already has the smallest recovery, so the record is empty. The record survives saving and
    reloading the Plan and the Result, and the reloaded input keeps the declared bound.

    Under lchs_kernel="auto" the record names the grid too, and planning that route on that pair through the public
    Method gives the recovery the record reports, which is at most the recovery on the two other compared pairs named
    here. With the route named under "auto" the grids are still compared, so a smaller recovery on another grid is
    reported. A preparation circuit leaves the CX counts unknown. Then the fixed order chooses between alternatives,
    so the selected route can have more original-U calls than an admitted route of another alternative, and the record
    says that the fixed order decided, as it does when several grids are compared with one alternative.
    """
    from qiskit import QuantumCircuit
    from nwqlib import NormalizedExpectation, NormSquared, load_result
    from nwqlib._choice_archive import ArchiveFiles, load_plan, save_plan
    from nwqlib.algorithms.lchs.primary_records import FIXED_SHOT_RULES
    from nwqlib.algorithms.lchs.selection import PROVIDER_FRONTIER
    from nwqlib.core.records import InputRef, Source
    from nwqlib.operators import ingest_cartesian, ingest_pauli
    from nwqlib.problems.inputs import bind_preparation_circuit
    L, _ = _ring_laplacian(3)
    initial = np.eye(8)[0]
    method = LCHS(approximation_tolerance=.1)
    settings = dict(method=method, output=NormSquared(), shots=256, seed=1)
    plain = plan(LinearDynamics(A=L, initial_state=initial, time=5.), **settings)
    assert plain.reconstruction.fixed_shot_precision is None
    problem = LinearDynamics(A=ingest_cartesian(L, dissipative_lower_bound=0.), initial_state=initial, time=5.)
    chosen = plan(problem, **settings)
    rec = chosen.reconstruction
    assert chosen._native["native_data"].qsp_plan.candidate.alternative == "filter"
    assert rec.recovery == plain.reconstruction.recovery
    precision = rec.fixed_shot_precision
    assert precision.ranking == "cx_per_circuit" and precision.readout == "unconditional_counts"
    assert precision.smallest_row == "lchs:separate_parts/table:shared_oaa[0.999]"
    route = plan(problem, method=LCHS(**_KERNEL_ROUTE), seed=1).reconstruction.recovery
    assert (precision.smallest_recovery.mantissa, precision.smallest_recovery.exponent) == (route.mantissa, route.exponent)
    # rel=1e-14 allows the few binary64 products and quotients on either side of the same ratio.
    assert precision.factor.as_float() == pytest.approx((rec.recovery.as_float()/route.as_float())**2, rel=1e-14)
    printed = format(precision.factor.as_float(), ".4g")
    assert printed in str(chosen)
    assert FIXED_SHOT_RULES[precision.ranking] in str(chosen)
    assert plan(problem, **settings) == chosen
    automatic = dict(lchs_kernel="auto", k_quadrature="auto")
    fixed_shots = dict(output=NormSquared(), shots=256, seed=1)
    compared = plan(problem, method=LCHS(**automatic, approximation_tolerance=.1), **fixed_shots).reconstruction.fixed_shot_precision
    # The default pair is the first grid that "auto" compares and acl_beta_0.7 a later one, so each must enter the record.
    on_beta = plan(problem, method=LCHS(**_KERNEL_ROUTE, lchs_kernel=ProviderConfig(
        implementation="near_optimal_eq7", parameters={"beta": .7})), seed=1).reconstruction.recovery
    assert compared.smallest_recovery.as_float() <= min(route.as_float(), on_beta.as_float())
    # This node encoding needs a uniform grid, so it leaves the Low-Somma pairs and their free grids, which have no
    # entry in the frontier list because a Method cannot name them. A free grid here can have the smaller recovery.
    uniform = {**_KERNEL_ROUTE, "qsp_route": {"implementation": "shared_oaa", "parameters": {"level": .999, "node": "rotation_table"}}}
    named_node = plan(problem, method=LCHS(**automatic, **uniform), **fixed_shots).reconstruction.fixed_shot_precision
    for record, named_route in ((compared, _KERNEL_ROUTE), (named_node, uniform)):
        entry = next(item for item in PROVIDER_FRONTIER if item.label == record.smallest_row.partition(" on grid ")[2])
        reached = plan(problem, method=LCHS(**named_route, lchs_kernel=entry.lchs_kernel, k_quadrature=entry.k_quadrature),
                       seed=1).reconstruction.recovery
        assert (reached.mantissa, reached.exponent) == (record.smallest_recovery.mantissa, record.smallest_recovery.exponent)
    within_plan = plan(problem, method=LCHS(**automatic, **_KERNEL_ROUTE), **fixed_shots)
    within = within_plan.reconstruction.fixed_shot_precision
    within_grid = next(item["label"] for item in within_plan._native["native_data"].provider_selection if item["status"] == "selected")
    assert within.selected_row == f"{precision.smallest_row} on grid {within_grid}" != within.smallest_row
    assert within.smallest_row == compared.smallest_row

    circuit = QuantumCircuit(1)
    circuit.ry(1.25, 0)
    state = bind_preparation_circuit(circuit, reference=InputRef(identity="user:state", representation="circuit", source=Source(
        name="user", version="1", domain="test", reference="declared")), basis=ingest_pauli([("Z", 1.)], num_qubits=1).basis, norm=1.)
    tilted = ingest_pauli([("I", .5), ("X", .3), ("Z", .3)], num_qubits=1)
    unknown = plan(LinearDynamics(A=tilted, initial_state=state, time=.5), method=LCHS(approximation_tolerance=.2), **fixed_shots)
    selected = unknown._native["native_data"].qsp_plan
    assert selected.candidate.cost.get("objective") is None and selected.candidate.cost["original_u_calls"] > min(
        row.cost["original_u_calls"] for row in selected.selection.candidates if row.status == "admitted" and row.requested)
    assert unknown.reconstruction.fixed_shot_precision.ranking == "original_u_calls_and_order"
    assert FIXED_SHOT_RULES["original_u_calls_and_order"] in str(unknown)
    # With H the kernel sum is the one alternative, so here only the grids of "auto" make the fixed order decide.
    grids = plan(LinearDynamics(A=ingest_cartesian(tilted, ingest_pauli([("Z", .1)], num_qubits=1)), initial_state=state, time=1.),
                 method=LCHS(**automatic, approximation_tolerance=.1), **fixed_shots).reconstruction.fixed_shot_precision
    assert grids.ranking == "original_u_calls_and_order"

    conditional_plan = plan(problem, method=method, output=NormalizedExpectation(observable=ingest_pauli(
        [("ZII", 1.)], num_qubits=3)), shots=256, seed=1)
    conditional = conditional_plan.reconstruction.fixed_shot_precision
    assert conditional.smallest_row == precision.smallest_row and conditional.factor == precision.factor
    assert conditional.readout == "conditional_counts"
    assert format(1/conditional.factor.as_float(), ".4g") in str(conditional_plan)
    files = ArchiveFiles(tmp_path, 8_000_000)
    restored = load_plan(save_plan(chosen, files), files)
    assert restored == chosen and restored.problem.A.cartesian_parts().dissipative_lower_bound == 0.
    result = solve(chosen)
    reloaded = load_result(result.save(tmp_path/"result"))
    assert printed in str(result) and printed in str(reloaded)
    assert reloaded.plan.reconstruction.psd_premise == "user_assertion"


def test_sampled_pairs_keep_the_declared_shift_and_cover_the_exact_norm():
    """The sampled estimator uses the declared bound in its masses, and its covering interval contains the exact norm.

    L = .5 I + .3 X + .3 Z has the lowest eigenvalue .5 - sqrt(.18) = .076 and the structural bound -.1, so the
    declaration 0 is valid and improves on it. The oracle at T = 1 is ||expm(-L T) u0||**2, which lies within the
    reported covering radius of 32 rows, at the planned failure probability, for this fixed seed. On one kernel
    grid, with one duration and the same rows, Gamma carries exp(s T), so the Hoeffding radius at fixed shots, which
    scales as Gamma**2, is larger by exp(2 s T) = exp(.2) on the structural Plan. The tolerance 1e-12 covers binary64
    rounding of the two evaluations of the same formula.
    """
    from qiskit.quantum_info import SparsePauliOp
    from nwqlib import NormSquared
    from nwqlib.algorithms.lchs.primary_records import ASSERTED_L_BOUNDS
    from nwqlib.operators import ingest_cartesian, ingest_pauli
    terms = [("I", .5), ("X", .3), ("Z", .3)]
    L, matrix = ingest_pauli(terms, num_qubits=1), SparsePauliOp.from_list(terms).to_matrix()
    initial = np.array([.6, .8])
    declared = LinearDynamics(A=ingest_cartesian(L, dissipative_lower_bound=0.), initial_state=initial, time=1.)
    structural = LinearDynamics(A=L, initial_state=initial, time=1.)

    def sampled(problem, **options):
        return plan(problem, method=LCHS(observable_estimator="sampled_pairs", **options), output=NormSquared(),
                    shots=32, seed=3)
    chosen = sampled(declared, hamiltonian_evolution_backend="qsp_block_encoding", approximation_tolerance=.3)
    assert (chosen.reconstruction.psd_shift, chosen.reconstruction.psd_premise) == (0., "user_assertion")
    assert ASSERTED_L_BOUNDS in chosen.assumptions
    result = solve(chosen)
    assert ASSERTED_L_BOUNDS in next(item for item in result.facts
                                     if item.fact.quantity == "algorithmic_approximation").fact.assumptions
    estimate = result.pair_estimate
    assert ASSERTED_L_BOUNDS in estimate.qualification
    assert abs(estimate.norm_squared-np.linalg.norm(expm(-matrix)@initial)**2) <= estimate.radius_n
    grid = {key: GRID[key] for key in ("hamiltonian_evolution_backend", "lchs_kernel", "k_quadrature")}
    radii = [solve(sampled(problem, **grid)).pair_estimate.sampling_radius_n for problem in (structural, declared)]
    assert radii[0]/radii[1] == pytest.approx(math.exp(.2), rel=1e-12)
