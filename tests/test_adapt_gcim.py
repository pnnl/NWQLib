"""ADAPT configuration, exact query association and original scientific conventions."""

from math import cos, pi, sin, sqrt
from types import SimpleNamespace
import numpy as np
import pytest
from test_adapt_primary import plan_for
from nwqlib.algorithms.gcim import ADAPT, adapt_acquisition
from nwqlib._numerics import stable_vector_norm
from nwqlib.execution import PauliValue
from nwqlib.operators import ingest_pauli


def test_processed_commutator_sign_at_subnormal_squared_scale():
    tiny = 2.0**-600
    for shift in (0.0, 1e100):
        plan = plan_for(
            A=ingest_pauli((("Z", tiny), ("I", shift)), num_qubits=1),
            pool_rows=((("Y", 1j),),),
            max_iterations=1,
            pauli_coefficient_cutoff=0.0,
        )
        assert plan.reconstruction.pool[0].commutator.rows() == (
            ("X", 2 * tiny),
        )
    gradient = np.array([2 * tiny, 2 * tiny])
    assert gradient @ gradient == 0.0
    assert stable_vector_norm(gradient) == pytest.approx(2 * tiny * sqrt(2), rel=2e-15, abs=0.0)


def test_query_completeness_and_exact_point_identity():
    plan = plan_for()
    point = adapt_acquisition._point(plan, "screen")
    chunk = SimpleNamespace(
        realization_id=point.content_id,
        bindings=point.bindings,
        population="unconditional",
        values=(),
    )
    with pytest.raises(adapt_acquisition._PartialData, match="missing X"):
        adapt_acquisition._read_values(plan, point, chunk)
    chunk.values = (PauliValue(label="X", value=0.5),)
    assert adapt_acquisition._read_values(plan, point, chunk) == {"X": 0.5}
    changed = adapt_acquisition._point(plan, "screen", right=((0, pi / 8),))
    with pytest.raises(ValueError, match="exact query"):
        adapt_acquisition._read_values(plan, changed, chunk)


@pytest.mark.parametrize("representation", ["dense", "csr"])
def test_classical_matrix_identity_offset_moves_the_ritz_value_by_its_coefficient(representation):
    """E(A0 + cI) - c equals E(A0) to rounding for a dense or sparse matrix Hamiltonian."""
    from scipy import sparse
    from nwqlib import Eigenproblem, solve

    pauli = {"I": np.eye(2), "X": np.array([[0., 1.], [1., 0.]]),
             "Y": np.array([[0., -1j], [1j, 0.]]), "Z": np.diag([1., -1.])}
    terms = (("ZI", .5), ("XX", .25), ("IZ", .375), ("YY", .125))
    a0 = sum(c * np.kron(pauli[label[0]], pauli[label[1]]) for label, c in terms)
    pool = tuple(ingest_pauli(((p, 1j),), num_qubits=2) for p in ("YI", "IY", "XY", "YX"))
    # A fixed angle of 1e-3 makes the basis states nearly parallel, cond(S) about 2e6.
    method = ADAPT(initial_state=[1, 0, 0, 0], pool=pool, theta=1e-3, max_iterations=3)
    values, selections = {}, set()
    for c in (.25, -1e4):
        matrix = a0 + c * np.eye(4)
        result = solve(Eigenproblem(A=matrix if representation == "dense" else sparse.csr_matrix(matrix)),
                       method=method, execution="classical", seed=7)
        assert result.plan.reconstruction.terms == ()
        assert result.plan.reconstruction.host_identity_shift == c  # A0 is traceless.
        assert result.pencil.overlap_condition_number > 1e6
        values[c], selections = result.eigenvalue, selections | {result.selected}
    assert len(selections) == 1
    # Both runs solve the same offset-free pencil, and only adding c back
    # rounds. Solving (A0 + cI, S) directly would err by about u|c|cond(S),
    # 1e-6 here.
    assert abs((values[-1e4] + 1e4) - (values[.25] - .25)) <= (
        2 * 2.**-53 * (abs(values[-1e4]) + abs(values[.25])))


def test_shared_full_chain_query_serves_the_diagonal_and_the_screen(tmp_path, monkeypatch):
    """One exact query of each full chain supplies its pencil diagonal and its screening values.

    H = Z, reference |+>, theta = pi/8, one iteration. Each full chain (the
    empty chain, then the selected one) is acquired once, during its matrix
    stage, with a weighted diagonal-reduction point and, when active
    screening labels remain, a screen point at the same boundary. The later
    screen reuses that observation, and the round's ``basis_realization``
    names it. No diagonal pair query is acquired. A saved Result reanalyzes without
    acquisition.
    """
    from nwqlib import Eigenproblem, load_result, solve

    method = ADAPT(initial_state=[1, 1], pool=tuple(ingest_pauli(((p, 1j),), num_qubits=1) for p in ("Y", "X")),
                   theta=pi / 8, max_iterations=1)
    result = solve(Eigenproblem(A=ingest_pauli((("Z", 1.0),), num_qubits=1)), method=method, execution="quantum")
    chunks = result.data.observations.chunks
    chains = ((), tuple(zip(result.selected, result.theta)))
    shared = [adapt_acquisition._point(result.plan, "screen", right=chain) for chain in chains]
    points = {point.content_id: sorted(c.point for c in chunks if c.realization_id == point.content_id)
              for point in shared}
    assert points[shared[0].content_id] == ["diagonal", "screen"]
    assert points[shared[1].content_id] == ["diagonal", "screen"]
    assert len({c.attempt for c in chunks if c.experiment == "screen"}) == 2
    pairs = [c for c in chunks if c.experiment == "pair"]
    assert len({c.attempt for c in pairs}) == 1 and all(c.point == "pair" for c in pairs)
    assert result.history[-1].basis_realization == shared[1].content_id
    loaded = load_result(result.save(tmp_path / "result"))
    from nwqlib.backends import qiskit_aer

    monkeypatch.setattr(qiskit_aer, "_submit_aer_execution", lambda *a, **k: pytest.fail("reanalysis acquired"))
    assert loaded.analyze(overlap_cutoff=1e-10).eigenvalue == pytest.approx(result.eigenvalue, abs=1e-14, rel=0)


def _real_adapt(acquisition=None, *, shots=32, pool=("YI", "IY", "XY", "YX"),
                terms=(("ZI", 1.0), ("IZ", 1.0), ("XX", .5)), max_iterations=3, **settings):
    """Sampled ADAPT from |00> with single-word generators i P, without optimization; ``settings`` go to ADAPT."""
    from nwqlib import Eigenproblem, plan
    from nwqlib.algorithms import PencilAcquisition
    from nwqlib.problems import ingest_occupation

    method = ADAPT(initial_state=ingest_occupation((0, 0), num_qubits=2),
                   pool=tuple(ingest_pauli(((p, 1j),), num_qubits=2) for p in pool), theta=pi / 8,
                   max_iterations=max_iterations, gradient_norm_floor=0.0,
                   acquisition=PencilAcquisition() if acquisition is None else acquisition, **settings)
    return plan(Eigenproblem(A=ingest_pauli(terms, num_qubits=2)), method=method, execution="quantum",
                shots=shots, seed=7)


def test_grouped_adapt_acquires_only_the_new_settings_of_each_stage(tmp_path, monkeypatch):
    """Each matrix stage measures only the settings that no earlier basis measured.

    H = ZI + IZ + 0.5 XX has G = 2 groups and L = 3 terms, and the pool iYI,
    iIY, iXY, iYX has purely imaginary coefficients on words with one Y, so
    from |00> every chain state is real and the default resolves real
    parts. Without optimization each basis contains the previous one, so a
    round with b states after one with b' measures c(b) - c(b') settings:
    c(b) = b(b+1)G/2 real-only, b**2 G with both parts, and
    bL + b(b-1)(L+1)/2 for the real-only termwise comparison. The bases have
    1, 2, 4 and 6 states. From two to three selections the old states (),
    a, b and ab move to positions (0, 1, 2, 5). Every setting of two old
    states was measured in an earlier round, by chain identity, and every
    other one in the third round, 11G = 22 real-only or 20G = 40 with both
    parts. A Y term in H or a generator iXZ, whose word has no Y, keeps both
    parts, and an explicit real request is then refused. A = 0.75 I with the
    complex reference (1, i)/sqrt(2) measures nothing: it has no
    nonidentity energy settings and [A, iY] = 0 gives no screening labels,
    so the run stops at the zero gradient before any selection. The same
    explicit real request then plans with shots, resolves no choice, runs
    no circuit and returns the scalar energy 0.75 exactly, and with
    PencilSampling both of its stages have zero shots. Bounded Ritz growth
    on the scalar 0.75 II measures overlaps only, in both stages of its
    PencilSampling, once its basis can grow. The saved grouped
    Result reloads with its resolved acquisition, group moments, pair
    variances and pair constructions, and a new overlap cutoff reanalyzes
    it without acquisition from the same moments. Each of the 6 * 5 / 2 = 15
    off-diagonal grouped pairs has a joint-state construction. A termwise
    Hadamard test places each Pauli term between the two controlled
    preparations, so it prepares no joint state and records no construction.
    Its seeded estimate rises at the third selection, and that round's
    energy_decrease is the negative change, not clipped to zero.
    """
    from nwqlib import load_result, solve
    from nwqlib.algorithms import ApplicabilityError, PencilAcquisition
    from nwqlib.backends import qiskit_aer

    groups, terms, sizes = 2, 3, (1, 2, 4, 6)
    laws = (
        (PencilAcquisition(), lambda b: b * (b + 1) * groups // 2),
        (PencilAcquisition(quadratures="complex"), lambda b: b * b * groups),
        (PencilAcquisition(matrix_measurement="termwise", pair_preparation="controlled"),
         lambda b: b * terms + b * (b - 1) * (terms + 1) // 2),
    )
    for request, law in laws:
        plan = _real_adapt(request)
        assert plan.reconstruction.acquisition.quadratures == (
            "complex" if request.quadratures == "complex" else "real")
        result = solve(plan, progress=False)
        pencil = {c.content_id: c for c in result.data.observations.chunks if c.experiment in ("pair", "diagonal")}
        rounds = [[pencil[i].realization_id for i in row.contribution_ids if i in pencil] for row in result.history]
        assert [len(ids) for ids in rounds] == [law(b) - law(a) for a, b in zip((0, *sizes), sizes)]
        selected = tuple(zip(result.selected, result.theta))
        chains, old = adapt_acquisition.basis_chains(selected), adapt_acquisition.basis_chains(selected[:2])
        assert len(chains) == 6 and [chains.index(chain) for chain in old] == [0, 1, 2, 5]
        earlier = {i for ids in rounds[:3] for i in ids}
        for (i, j), _, _, point in adapt_acquisition._matrix_entries(plan, chains):
            identity, both_old = point().content_id, chains[i] in old and chains[j] in old
            assert (identity in earlier, identity in rounds[3]) == (both_old, not both_old)

        if request is laws[0][0]:  # the default grouped acquisition
            assert len(result.group_moments) == law(6)
            shared = {row.pair: (row.route, row.shared_blocks) for row in result.pair_preparations}
            assert len(shared) == 15 and shared[0, 1] == ("prefix", 1)
            assert shared[1, 4] == ("prefix", 2) and shared[4, 5] == ("prefix", 3)  # (a, abc), (abc, ab)
            loaded = load_result(result.save(tmp_path / "grouped"))
            assert loaded.plan.reconstruction.acquisition == plan.reconstruction.acquisition
            assert (loaded.group_moments, loaded.sampled_pair_variances, loaded.pair_preparations) == (
                result.group_moments, result.sampled_pair_variances, result.pair_preparations)
            with monkeypatch.context() as patch:
                patch.setattr(qiskit_aer, "_submit_aer_execution", lambda *a, **k: pytest.fail("reanalysis acquired"))
                assert loaded.analyze(overlap_cutoff=1e-10).group_moments == result.group_moments
        elif request is laws[2][0]:  # termwise Hadamard tests prepare no joint state
            assert result.pair_preparations == ()
            energies = [row.energy for row in result.history]
            decreases = [row.energy_decrease for row in result.history]
            assert decreases == [None, *(x - y for x, y in zip(energies, energies[1:]))] and decreases[3] < 0

    with_y = (("ZI", 1.0), ("IZ", 1.0), ("XX", .5), ("YI", .25))
    for change, reason in ((dict(terms=with_y), "Hamiltonian word YI has an odd number of Y"),
                           (dict(pool=("YI", "IY", "XY", "YX", "XZ")), "pool generator 4 term XZ is not real")):
        acquisition = _real_adapt(**change).reconstruction.acquisition
        assert (acquisition.quadratures, acquisition.real_proof, acquisition.reason) == ("complex", "unknown", reason)
        with pytest.raises(ApplicabilityError, match="needs a structural real proof"):
            _real_adapt(PencilAcquisition(quadratures="real"), **change)

    request = PencilAcquisition(quadratures="real")
    scalar = plan_for(A=ingest_pauli((("I", .75),), num_qubits=1), initial_state=(1, 1j),
                      pool_rows=((("Y", 1j),),), shots=32, max_iterations=1, acquisition=request)
    acquisition = scalar.reconstruction.acquisition
    assert (acquisition.requested, acquisition.matrix_measurement, acquisition.pair_preparation,
            acquisition.quadratures, acquisition.real_proof) == (request, None, None, None, None)
    result = solve(scalar, progress=False)
    assert (result.eigenvalue, result.selected, result.stop_reason) == (0.75, (), "gradient_norm_floor")
    assert not (result.data.trace.events or result.data.observations.chunks or result.data.receipts)

    # With PencilSampling the scalar Plan still measures nothing, and both stages have zero shots.
    from nwqlib.algorithms import BoundedRitzGrowth, PencilSampling

    staged = solve(plan_for(A=ingest_pauli((("I", .75),), num_qubits=1), initial_state=(1, 1j),
                            pool_rows=((("Y", 1j),),), shots=32, max_iterations=1, acquisition=request,
                            sampling=PencilSampling(total_shots=12, pilot_fraction=.25)), progress=False)
    allocation = staged.shot_allocation
    assert (staged.eigenvalue, allocation.pilot_shots, allocation.main_shots, allocation.effective_fraction,
            allocation.pilot) == (.75, 0, 0, "not_applicable", None)
    assert not (staged.data.trace.events or staged.data.observations.chunks)
    # Bounded Ritz growth on a scalar operator measures overlaps once its basis can hold more than the
    # reference. Its final basis of b = 2 states has m = 1 real overlap setting, whose estimate v_s is zero
    # because H0 = 0, so the main stage takes the uniform B - B_p = 48 - 12 = 36 shots, and only the
    # main chunk forms the result.
    bounded = solve(_real_adapt(terms=(("II", .75),), shots=4096, max_iterations=2,
                                basis_growth=BoundedRitzGrowth(max_chain_length=1),
                                sampling=PencilSampling(total_shots=48, pilot_fraction=.25)), progress=False)
    chunks = bounded.data.observations.chunks
    stages = [{item.parameter: item.value for item in chunk.bindings}["stage"] for chunk in chunks]
    allocation = bounded.shot_allocation
    assert len(bounded.value_basis) == 2 and stages.count(0) == stages.count(1) == 1
    assert (allocation.pilot_counts, allocation.main_counts, allocation.fallback_reason) == (
        (12,), (36,), "pilot_variances_all_zero")
    assert set(bounded.contribution_ids) == {c.content_id for c, stage in zip(chunks, stages) if stage == 1}


def test_pencil_sampling_measures_the_final_basis_again_after_an_unchanged_selection(tmp_path, monkeypatch):
    """PencilSampling leaves the selection as it is and measures its final basis in a fresh pilot and main stage.

    The Plan of the grouped keeper above ends with 3 selections, a basis of
    b = 6 states with G = 2 groups and real parts, so its final pencil has
    m = b(b+1)G/2 = 42 settings, each of them already measured during the
    selection. The same Plan with PencilSampling(B = 1000, f = 1/4) makes
    the same selection queries with the same counts, so its history equals
    that of the Run without sampling apart from the identities of its
    queries. Then the Run submits 2m fresh measurements, the pilot first:
    B_p = min(B - 2m, max(2m, floor(B/4))) = 250 shots, 6 for each of the
    first 250 - 42*5 = 40 settings in setting order and 5 for the other two,
    and B - B_p = 750 main shots of at least 2 per setting. The pilot pencil
    of the decision comes from the pilot chunks alone. Only the main chunks
    form the eigenvalue, the pencil and the group moments, while the
    selection and pilot chunks stay in the Run. The Result counts the
    analyses of the Run without sampling and two more, the pilot allocation
    and the analysis of the main stage. The checkpoints encode the pilot
    counts and the decision once each, when they are published, although
    the stage cursor is saved again at every setting. Saved, loaded and
    reanalyzed at another cutoff, the Result keeps its decision and
    main-stage contributions without measuring or allocating again.
    Planning refuses a budget below 4m and a Plan without grouped stage
    settings. A selection, pilot and main point of one setting are three
    query points, each estimated with the shots it binds, and the estimate
    of the whole Plan leaves the stage-dependent counts unavailable.
    """
    import json
    from fractions import Fraction
    import nwqlib._run_journal as journal
    from nwqlib import load_result, solve
    from nwqlib.algorithms import PencilSampling
    from nwqlib.algorithms.gcim import pencil_workflow
    from nwqlib.algorithms.gcim.fixed_basis import pencil_pairs, sampled_settings
    from nwqlib.backends import qiskit_aer

    b, groups, budget = 6, 2, 1000
    m = b * (b + 1) * groups // 2
    p, q = (.25).as_integer_ratio()
    pilot = min(budget - 2 * m, max(2 * m, p * budget // q))
    pilot_counts = tuple(pilot // m + (k < pilot % m) for k in range(m))
    assert (m, pilot, pilot_counts.count(6), pilot_counts.count(5)) == (42, 250, 40, 2)
    # Planning knows only the largest basis the Plan can reach, 2*min(max_iterations, pool size) = 6
    # states, whose m settings bound those of any final basis. It refuses B = 4m - 1 before any query
    # and accepts 4m, two pilot and two main shots per setting.
    with pytest.raises(ValueError, match=f"{4 * m} shots"):
        _real_adapt(sampling=PencilSampling(total_shots=4 * m - 1, pilot_fraction=.25))
    _real_adapt(sampling=PencilSampling(total_shots=4 * m, pilot_fraction=.25))
    plain = solve(_real_adapt(), progress=False)
    sampled = _real_adapt(sampling=PencilSampling(total_shots=budget, pilot_fraction=.25))
    # A Plan that measures without grouped settings, by termwise Hadamard tests, exact readout or classical
    # execution, would end its final phase with no stage setting and silently leave the budget unspent,
    # so planning refuses it.
    from nwqlib import plan as plan_of
    from nwqlib.algorithms import PencilAcquisition

    termwise = PencilAcquisition(matrix_measurement="termwise", pair_preparation="controlled")
    with pytest.raises(ValueError, match="grouped"):
        _real_adapt(termwise, sampling=sampled.method.sampling)
    with pytest.raises(ValueError, match="shots="):
        _real_adapt(shots=None, sampling=sampled.method.sampling)
    with pytest.raises(ValueError, match="execution='quantum'"):
        plan_of(sampled.problem, method=sampled.method, execution="classical", seed=7)
    encoded, encode = [], journal._encode_admitted

    def spy(value, max_bytes):
        """Record each row value that the run log encodes, as read back from its JSON text."""
        text, size = encode(value, max_bytes)
        encoded.append(json.loads(text))
        return text, size

    with monkeypatch.context() as patch:
        patch.setattr(journal, "_encode_admitted", spy)
        result = solve(sampled, progress=False)

    def rounds(record):
        return [row.model_dump(exclude={"contribution_ids", "basis_realization"}, exclude_computed_fields=True)
                for row in record.history]

    assert rounds(result) == rounds(plain) and result.stop_reason == plain.stop_reason == "max_iterations"
    assert result.analysis_attempts == plain.analysis_attempts + 2
    assert [len(row.contribution_ids) for row in result.history] == [len(row.contribution_ids) for row in plain.history]
    assert len(result.selected) == 3
    chunks, n = result.data.observations.chunks, len(plain.data.observations.chunks)
    events = result.data.trace.events
    assert len(events) == len(plain.data.trace.events) + 2 * m == n + 2 * m
    assert sum(e.shots for e in events) == sum(e.shots for e in plain.data.trace.events) + budget

    def bound(chunk):
        return {item.parameter: item.value for item in chunk.bindings}

    assert [bound(c)["stage"] for c in chunks] == [-1] * n + [0] * m + [1] * m
    # Each stage measures the settings of the final basis in setting order.
    settings = list(sampled_settings(pencil_pairs(b), groups, "real"))
    for stage in (chunks[n:n + m], chunks[n + m:]):
        assert [(c.experiment, bound(c)["group"]) for c in stage] == [
            ("diagonal" if i == j else "pair", g) for _, i, j, _, g in settings]
    allocation = result.shot_allocation
    # Each setting has one pilot chunk of N = 5 or 6 shots, and the parts are
    # real. The pilot overlap S_ij of an off-diagonal group-0 setting is the
    # mean of the ancilla bit 0 of its chunk. Each of the K stored bin counts
    # is divided by N before a K-term sum of signs, so the binary64 entry is
    # within gamma_K = K u/(1 - K u), u = 2**-53, of the exact mean.
    for chunk, (_, i, j, _, g) in zip(chunks[n:n + m], settings, strict=True):
        if i == j or g:
            continue
        total, k = sum(item.count for item in chunk.values), len(chunk.values)
        mean = Fraction(sum(item.count * (-1) ** (int(item.bits, 2) & 1) for item in chunk.values), total)
        assert total == chunk.returned_shots and total in (5, 6)
        assert abs(Fraction(allocation.pilot.overlap[i][j].real) - mean) <= Fraction(k, 2**53 - k)
    assert (allocation.pilot_shots, allocation.main_shots, allocation.pilot_counts) == (pilot, budget - pilot,
                                                                                        pilot_counts)
    assert [c.returned_shots for c in chunks[n:n + m]] == list(pilot_counts)
    assert [c.returned_shots for c in chunks[n + m:]] == list(allocation.main_counts)
    assert sum(allocation.main_counts) == budget - pilot and min(allocation.main_counts) >= 2
    if allocation.fallback_reason is not None:
        main_q, main_r = divmod(budget - pilot, m)
        assert allocation.main_counts == tuple(main_q + (k < main_r) for k in range(m))
    main = {c.content_id for c in chunks[n + m:]}
    assert set(result.contribution_ids) == main == {moment.contribution_id for moment in result.group_moments}
    assert len(result.group_moments) == m and result.eigenvalue == result.pencil.eigenvalues[0]
    # One main chunk per setting: each real overlap S_ij of the reported
    # pencil is the group-0 overlap mean of its main-stage moment.
    for moment in result.group_moments:
        i, j = moment.pair
        if moment.group == 0 and i != j:
            entry = result.pencil.overlap[i][j]
            assert (entry.real, entry.imag) == (moment.overlap_mean, 0.0)
    # A checkpoint encodes only the fields it changes. The decision is
    # encoded in one row, and the pilot counts in one other row apart from
    # the decision's own copy of them. A row that held either one as a field,
    # such as a stage cursor saved at every setting, would count again.
    decision = allocation.model_dump(mode="json", exclude_computed_fields=True)

    def rows(payload):
        """The encoded rows that are ``payload`` or hold it as one of their fields."""
        def same(value):
            return type(value) is type(payload) and value == payload

        return [row for row in encoded if same(row) or type(row) is dict and any(map(same, row.values()))]

    assert len(rows(decision)) == 1
    assert len([row for row in rows(list(pilot_counts)) if row != decision]) == 1

    # Each final-stage setting is a query point of its own stage, so even with the selection's shots the
    # selection point and the pilot and main points of one setting differ and no observation of one stage
    # serves another. The estimate of a point gives the shots it binds, while the estimate of the whole
    # Plan reports its settings and shots as unavailable, since the stage shots depend on the pilot, and
    # every measurement batch of the Plan reads counts.
    from nwqlib import estimate as estimate_plan
    from nwqlib.ir import ExprRef, MeasurementBatch
    from nwqlib.resources import estimate as estimate_point

    points = [adapt_acquisition._point(sampled, "diagonal", (), (), group=0, stage=stage, shots=shots)
              for stage, shots in ((-1, None), (0, 32), (1, 32), (1, 13))]
    assert len({point.content_id for point in points[:3]}) == 3
    assert [estimate_point(point.selected_construction(sampled)).quantity("shots").fact.value.numerator
            for point in points] == [32, 32, 32, 13]
    workload = estimate_plan(sampled)
    assert {workload.quantity(metric).fact.availability for metric in ("settings", "shots", "measurements")} == {
        "unknown"}
    assert workload.quantity("exact_evaluations").fact.value.numerator == 0
    batches = [d.node for d in sampled.construction.program.definitions if isinstance(d.node, MeasurementBatch)]
    assert {(batch.observation_kind, batch.repetitions) for batch in batches} == {("counts", ExprRef(expression="shots"))}

    path = result.save(tmp_path / "sampled")

    def fail(*args, **kwargs):
        pytest.fail("loading or reanalysis measured or allocated again")

    monkeypatch.setattr(qiskit_aer, "_submit_aer_execution", fail)
    for name in ("pilot_decision", "floored_allocation", "uniform_allocation", "pilot_split"):
        monkeypatch.setattr(pencil_workflow, name, fail)
    loaded = load_result(path)
    assert loaded.shot_allocation == allocation and loaded.eigenvalue == result.eigenvalue
    for source in (result, loaded):
        again = source.analyze(overlap_cutoff=1e-6)
        assert again.shot_allocation == allocation and set(again.contribution_ids) == main
        assert again.pencil.overlap_cutoff == 1e-6


def _contract(left, operator, right):
    """Return the real and imaginary parts of ``left^dagger operator right`` as exact rationals of the binary64 entries."""
    from fractions import Fraction as F

    action = []
    for row in operator:
        real = imag = F(0)
        for m, r in zip(row, right, strict=True):
            real += F(m.real) * F(r.real) - F(m.imag) * F(r.imag)
            imag += F(m.real) * F(r.imag) + F(m.imag) * F(r.real)
        action.append((real, imag))
    real = imag = F(0)
    for value, (ar, ai) in zip(left, action, strict=True):
        real += F(value.real) * ar + F(value.imag) * ai
        imag += F(value.real) * ai - F(value.imag) * ar
    return real, imag


@pytest.mark.parametrize("family", ["real", "complex"])
def test_one_basis_through_fixed_gcim_and_adapt_matches_its_independent_pencil(family):
    """One ordered basis through grouped ADAPT and grouped FixedGCIM, each compared with an independent pencil.

    ``real``: ADAPT on H = ZI + IZ + 0.5 XX from |00> with the real pool
    selects one generator iP and measures the basis |00>, exp(theta iP)|00>.
    Both Methods resolve real parts and measure b(b+1)G/2 = 6 settings, with
    G = 2 groups that can only be {ZI, IZ} and {XX}, and all eight real
    entries of the two pencils are asserted. ``complex``: A = 0.5 Z + 0.3 Y,
    the pool iY, iX, the reference |0> and the selection ((1, 0.7),
    (0, -0.6)) give the basis |0>, a, b, ab with a = exp(0.7 iX)|0> and
    b = exp(-0.6 iY)|0>. In canonical chain order b comes before a, so
    ADAPT measures the pair (1, 2) reversed, and the pairs (1, 2) and (1, 3)
    have nonzero imaginary H and S. ADAPT's current-basis matrices come from
    its grouped queries of these chains (``_matrix_from_observations``) and
    FixedGCIM's from the oracle vectors, b**2 G = 32 settings each, and the
    real and imaginary parts of H and S of both pairs, sixteen values, are
    asserted.

    The reference vectors v are independent products of the factors
    cos(t) I + sin(t) iP, formed from binary64 cos and sin. With one-ulp cos
    and sin each factor's entries err by at most 2 sqrt(2) u, which bounds
    its norm error by 4Du, and one dense product errs by at most
    2D gamma_2D (u = 2**-53, gamma_n = nu/(1 - nu), D = 2**q), so after p
    factors 1 + delta_p = ((1 + 4Du)(1 + 2D gamma_2D))**p bounds
    ||v - psi||. FixedGCIM normalizes v, and | ||v|| - 1 | <= delta_p, so the
    ideal states of both Methods lie within delta_p of the contracted v. The
    entries of v and of the Pauli matrices are contracted exactly as
    rationals, and expanding both sides with Cauchy-Schwarz adds
    C(delta_i + delta_j + delta_i delta_j) to an H bound, C the absolute
    coefficient sum, and the same without C to an S bound. Each asserted
    value lies within that error plus ``_hoeffding_radius`` of its
    settings: an empirical check of one seeded Aer run (seed 7, 4096 shots
    per setting) against the ideal-model reference radius. Mutual agreement
    of the two pencils alone would show only shared wiring.
    """
    from fractions import Fraction as F
    from qiskit.quantum_info import SparsePauliOp
    import nwqlib
    from nwqlib.algorithms import FixedGCIM, PencilAcquisition
    from nwqlib.problems import ingest_occupation, ingest_vector
    from nwqlib._prepared_execution import Run, prepare_experiment, submit_experiment
    from test_gcim_production import _entry_ranges, _hoeffding_radius, matrix

    if family == "real":
        q, terms, labels = 2, (("ZI", 1.0), ("IZ", 1.0), ("XX", .5)), ("YI", "IY", "XY", "YX")
        adapt = _real_adapt(shots=4096, max_iterations=1, terms=terms)
        result = nwqlib.solve(adapt, progress=False)
        selected = tuple(zip(result.selected, result.theta))
        h_adapt, s_adapt = matrix(result.pencil.hamiltonian), matrix(result.pencil.overlap)
        acquired = sum(c.experiment in ("pair", "diagonal") for c in result.data.observations.chunks)
        asserted, parts = ((0, 0), (0, 1), (1, 1)), ("real",)
    else:
        q, terms, labels = 1, (("Z", .5), ("Y", .3)), ("Y", "X")
        adapt = plan_for(A=ingest_pauli(terms, num_qubits=1), initial_state=ingest_occupation((0,), num_qubits=1),
                         pool_rows=tuple(((p, 1j),) for p in labels), shots=4096, max_iterations=2,
                         acquisition=PencilAcquisition(quadratures="complex"))
        selected, queries = ((1, .7), (0, -.6)), []
        with Run(adapt) as run:

            def acquire(point):
                queries.append(point.content_id)
                return submit_experiment(prepare_experiment(point, run=run), run=run)

            h_adapt, s_adapt, _, _ = adapt_acquisition._matrix_from_observations(
                adapt, adapt_acquisition.basis_chains(selected), adapt_acquisition.AdaptContext(), acquire=acquire)
        acquired = len(set(queries))
        assert acquired == len(queries)
        asserted, parts = ((1, 2), (1, 3)), ("real", "imag")
    chains = adapt_acquisition.basis_chains(selected)
    assert family == "real" or chains[2] < chains[1]
    dimension = 2**q
    generators = [1j * SparsePauliOp(label).to_matrix() for label in labels]
    vectors = []
    for chain in chains:
        v = np.eye(dimension, dtype=complex)[0]
        for index, t in chain:
            v = (np.cos(t) * np.eye(dimension) + np.sin(t) * generators[index]) @ v
        vectors.append(v)
    fixed = nwqlib.plan(nwqlib.Eigenproblem(A=ingest_pauli(terms, num_qubits=q)),
                        method=FixedGCIM(basis=(ingest_occupation((0,) * q, num_qubits=q),
                                                *map(ingest_vector, vectors[1:]))),
                        execution="quantum", shots=4096, seed=7)
    b, groups = len(chains), len(fixed.reconstruction.groups)
    for plan in (adapt, fixed):
        assert plan.reconstruction.acquisition.quadratures == family
        assert plan.reconstruction.identity_shift == 0.0
    assert acquired == len(fixed.experiments) == (b * (b + 1) // 2 if family == "real" else b * b) * groups
    # Both partitions have the same coefficient sums, so _entry_ranges applies to both pencils.
    assert sorted(sum(abs(c) for _, c in (adapt_acquisition._group_rows(adapt)[k] for k in group))
                  for group in adapt.reconstruction.energy_groups) == sorted(
        sum(abs(fixed.reconstruction.nonidentity_terms[k][1]) for k in group.indices)
        for group in fixed.reconstruction.groups)

    u = F(1, 2**53)
    gamma = 2 * dimension * u / (1 - 2 * dimension * u)
    growth = (1 + 4 * dimension * u) * (1 + 2 * dimension * gamma)
    delta = [growth**len(chain) - 1 for chain in chains]
    mass = sum(F(abs(c)) for _, c in terms)
    hamiltonian = SparsePauliOp.from_list(terms).to_matrix()
    count = 2 * sum(len(parts) * (1 + (i != j)) for i, j in asserted)
    solved = nwqlib.solve(fixed, progress=False)
    for h, s in ((h_adapt, s_adapt), (matrix(solved.pencil.hamiltonian), matrix(solved.pencil.overlap))):
        if family == "real":
            assert not h.imag.any() and not s.imag.any()
        for i, j in asserted:
            reference_error = delta[i] + delta[j] + delta[i] * delta[j]
            exact_h = _contract(vectors[i], hamiltonian, vectors[j])
            exact_s = _contract(vectors[i], np.eye(dimension), vectors[j])
            radius_h = F(_hoeffding_radius(_entry_ranges(fixed.reconstruction, i, j), count, 4096))
            radius_s = F(_hoeffding_radius([1.0], count, 4096))
            for k, part in enumerate(parts):
                assert abs(F(getattr(h[i, j], part)) - exact_h[k]) <= radius_h + mass * reference_error
                if i != j:
                    assert abs(F(getattr(s[i, j], part)) - exact_s[k]) <= radius_s + reference_error


def test_prefix_joint_state_keeps_each_preparation_phase():
    """Exact readout of factored joint states equals independently multiplied primitive products, phases included.

    One system qubit, A0 = 0.5 Z + 0.25 X, reference |1> (one X gate) and the pool
    i(Y + I), iY, iX. exp(t i(Y + I)) = exp(i t)(cos t I + i sin t Y)
    carries the phase exp(i t). The chains (a, b) and (a, c) with
    a = (0, 0.3), b = (1, 0.5) and c = (2, -0.7) share the reference and a,
    which run once without control (two shared blocks), and (a) and (a, c)
    leave the left suffix empty. Moving a's angle by one ulp in one chain
    leaves only the reference shared, so a's phase is then controlled on
    both branches. The reduction's S and H0 entries of each pair, with no
    phase alignment, lie within ``pair_reducer.entry_bounds`` of the pair's
    own receipt plus the error of the reference vectors: each factor is a
    phase gate exp(i t) I or a rotation cos t I + i sin t P with one-ulp cos
    and sin, so after p factors ||v - psi|| <= delta_p with
    1 + delta_(k+1) = (1 + delta_k)(1 + 4Du)(1 + 2D gamma_2D), D = 2, which
    adds C(delta_i + delta_j + delta_i delta_j) with C = 0.75 to an H0 bound
    and the same without C to an S bound. The reference components are
    contracted exactly as rationals. Under ``pair_preparation="controlled"``
    the same pairs match their products with more native operations.

    The grouped readout tails of the same builder: with the reference
    |-i> (H, then S^dagger), the pool iY and A = Y, the pair of () and
    ((0, -1/2),) has phi_1 = exp(-iY/2)|-i> = exp(i/2)|-i>, so
    S = exp(i/2) and H = -exp(i/2). Its real and imaginary grouped settings,
    with their measurements removed, have the ancilla Z parity cos(1/2) and
    sin(1/2) and, after the system Y rotation, the ancilla-system parity
    -cos(1/2) and -sin(1/2), within the primitive-gate bound of
    ``test_gcim_production._grouped_tail`` plus one ulp of cos or sin.
    """
    import json
    import math
    from fractions import Fraction as F
    from qiskit import QuantumCircuit
    from nwqlib.algorithms import PencilAcquisition
    from nwqlib.algorithms.gcim.pair_reducer import entry_bounds
    from nwqlib.problems import ingest_occupation
    from nwqlib._prepared_execution import Run, prepare_experiment, submit_experiment
    from test_gcim_production import _grouped_tail

    a, b, c = (0, .3), (1, .5), (2, -.7)
    moved = (0, math.nextafter(.3, 1.0))
    pairs = {((a, b), (a, c)): 2, ((a,), (a, c)): 2, ((a, c), (moved, b)): 1}
    identity, y, x = np.eye(2), np.array([[0, -1j], [1j, 0]]), np.array([[0, 1], [1, 0]])

    def factors(item):
        index, t = item
        rotation = math.cos(t) * identity + math.sin(t) * 1j * (x if index == 2 else y)
        return [complex(math.cos(t), math.sin(t)) * identity, rotation] if index == 0 else [rotation]

    def reference(chain):
        vector, count = np.array([0j, 1.0 + 0j]), 0
        for item in chain:
            for factor in factors(item):
                vector, count = factor @ vector, count + 1
        return vector, count

    u = F(1, 2**53)
    growth = (1 + 8 * u) * (1 + 4 * (4 * u / (1 - 4 * u)))
    a0 = ((F(1, 2), F(1, 4)), (F(1, 4), F(-1, 2)))

    def exact(left, right):
        (vl, pl), (vr, pr) = reference(left), reference(right)
        dl, dr = growth**pl - 1, growth**pr - 1
        bra = [(F(z.real), -F(z.imag)) for z in vl]
        ket = [(F(z.real), F(z.imag)) for z in vr]
        action = [(sum(a0[r][k] * ket[k][0] for k in range(2)), sum(a0[r][k] * ket[k][1] for k in range(2)))
                  for r in range(2)]

        def dot(p, q):
            return (sum(s[0] * t[0] - s[1] * t[1] for s, t in zip(p, q, strict=True)),
                    sum(s[0] * t[1] + s[1] * t[0] for s, t in zip(p, q, strict=True)))

        return dot(bra, ket), dot(bra, action), dl + dr + dl * dr

    native = {}
    for rule in ("prefix", "controlled"):
        plan = plan_for(A=ingest_pauli((("Z", .5), ("X", .25)), num_qubits=1),
                        initial_state=ingest_occupation((1,), num_qubits=1),
                        pool_rows=((("Y", 1j), ("I", 1j)), (("Y", 1j),), (("X", 1j),)), max_iterations=2,
                        acquisition=PencilAcquisition(pair_preparation=rule))
        with Run(plan) as run:
            for (left, right), shared in pairs.items():
                assert left < right
                assert adapt_acquisition.chain_pair_preparation(left, right, rule, (0, 1)).shared_blocks == (
                    shared if rule == "prefix" else 0)
                handle = prepare_experiment(adapt_acquisition._point(plan, "pair", left, right), run=run)
                chunk = adapt_acquisition._single_chunk(submit_experiment(handle, run=run))
                values = adapt_acquisition._pair_reduction_values(chunk)
                (parameters,) = (json.loads(p.parameters) for p in chunk.observation.positions
                                 if p.kind == "reduction")
                delta = handle.record.saved_state_error(("amplitude-derived masses",))[0]
                assert delta is not None
                overlap_bound, h0_bound = entry_bounds(delta, float.fromhex(parameters["c1"]),
                                                       1 << parameters["qubits"], parameters["terms"])
                s, h0, reference_error = exact(left, right)
                for actual, target, bound in ((values["overlap"], s, F(overlap_bound) + reference_error),
                                              (values["h0"], h0, F(h0_bound) + F(3, 4) * reference_error)):
                    assert abs(F(actual.real) - target[0]) <= bound and abs(F(actual.imag) - target[1]) <= bound
                native[rule, left, right] = handle.record.native_operations
    for left, right in pairs:
        assert native["prefix", left, right] < native["controlled", left, right]

    minus_i = QuantumCircuit(1)
    minus_i.h(0)
    minus_i.sdg(0)
    grouped = plan_for(A=ingest_pauli((("Y", 1.0),), num_qubits=1), initial_state=minus_i,
                       pool_rows=((("Y", 1j),),), max_iterations=1, shots=16)
    assert grouped.reconstruction.acquisition.matrix_measurement == "grouped"
    with Run(grouped) as run:
        for quadrature, value in ((0, math.cos(.5)), (1, math.sin(.5))):
            point = adapt_acquisition._point(grouped, "pair", (), ((0, -.5),), quadrature=quadrature, group=0)
            (ancilla, joint), bound = _grouped_tail(prepare_experiment(point, run=run))
            allowance = bound + F(math.ulp(value))
            assert abs(F(ancilla) - F(value)) <= allowance and abs(F(joint) + F(value)) <= allowance


# The vacuum fixture on two qubits: the reference |01> with qubit 0 occupied,
# a = i(0.5 II + 0.5 (XX + YY) + 0.25 ZZ) and b = 0.5 i(XY - YX).
_VACUUM_POOL = ((("II", .5j), ("XX", .5j), ("YY", .5j), ("ZZ", .25j)), (("XY", .5j), ("YX", -.5j)))
_VACUUM_TERMS = (("ZI", .5), ("XX", .25), ("IY", .375))


def _vacuum_plan(rule, *, pool_rows=_VACUUM_POOL, terms=_VACUUM_TERMS, max_iterations=2, **settings):
    """Plan ADAPT from |01> on two qubits with ``pair_preparation=rule``, by default with at most two selections."""
    from nwqlib.algorithms import PencilAcquisition
    from nwqlib.problems import ingest_occupation

    return plan_for(A=ingest_pauli(terms, num_qubits=2), initial_state=ingest_occupation((1, 0), num_qubits=2),
                    pool_rows=pool_rows, max_iterations=max_iterations, acquisition=PencilAcquisition(pair_preparation=rule),
                    **settings)


def _pauli(label):
    """Return the matrix of a two-qubit Pauli word whose left letter acts on qubit 1."""
    single = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]),
              "Z": np.diag([1, -1])}
    return np.kron(single[label[0]], single[label[1]]).astype(complex)


def _vacuum_vector(chain):
    """Return the oracle state of a chain of the vacuum fixture and its number of primitive factors.

    The words of a generator commute, so exp(t a) is the product of the
    phase exp(i t/2) and the rotations exp(i s P) = cos(s) I + i sin(s) P,
    s = t/2 for XX and YY and t/4 for ZZ, and exp(t b) that of the rotations
    by t/2 for XY and -t/2 for YX. Halving and quartering t are exact.
    """
    def rotation(label, s):
        return cos(s) * np.eye(4) + 1j * sin(s) * _pauli(label)

    vector, count = np.eye(4, dtype=complex)[1], 0
    for index, t in chain:
        factors = ([complex(cos(t / 2), sin(t / 2)) * np.eye(4), rotation("XX", t / 2),
                    rotation("YY", t / 2), rotation("ZZ", t / 4)] if index == 0
                   else [rotation("XY", t / 2), rotation("YX", -t / 2)])
        for factor in factors:
            vector, count = factor @ vector, count + 1
    return vector, count


def _native_tail(handle):
    """Return the exact ancilla and ancilla-system Z parities of a prepared grouped setting, with their error bound.

    ``test_gcim_production._grouped_tail`` for circuits that also hold
    rotations and phase gates by stored angles. Every gate must be H, X, S,
    S^dagger, CX, a phase gate P, RZ, RXX, RYY or RZZ. Each entry of such a
    gate is exact, one rounding of 2**-0.5, or one cos, sin or complex exp
    of a stored angle, so under the one-ulp assumption for cos, sin and exp
    it lies within 2 sqrt(2) u of the gate's exact matrix at that angle and
    its norm error is at most 4Du, D = 2**qubits. The bound is then that of
    ``_grouped_tail``. It concerns the circuit at its stored angles, so the
    fixture's angles must be the exact products of its inputs.
    """
    from fractions import Fraction as F
    from qiskit.quantum_info import SparsePauliOp, Statevector

    circuit = handle._restore_native().circuit
    measured = {circuit.find_bit(item.clbits[0]).index: circuit.find_bit(item.qubits[0]).index
                for item in circuit.data if item.operation.name == "measure"}
    body = circuit.remove_final_measurements(inplace=False)
    dimension = 1 << body.num_qubits
    u = F(1, 2**53)
    gamma = 2 * dimension * u / (1 - 2 * dimension * u)
    growth = F(1)
    for item in body.data:
        assert item.operation.name in {"h", "x", "s", "sdg", "cx", "p", "rz", "rxx", "ryy", "rzz"}, item
        growth *= (1 + 4 * dimension * u) * (1 + 2 * dimension * gamma)
    delta = growth - 1
    state = Statevector(body)

    def parity(bits):
        label = ["I"] * body.num_qubits
        for bit in bits:
            label[body.num_qubits - 1 - measured[bit]] = "Z"
        return state.expectation_value(SparsePauliOp("".join(label))).real

    return (parity((0,)), parity((0, 1))), 2 * delta + delta**2 + F(3, 2) * gamma * (1 + delta)**2


def test_vacuum_joint_states_correct_the_known_vacuum_phase():
    """Exact readout of the vacuum constructions equals independently multiplied primitive products, phases included.

    Two system qubits, A0 = 0.5 ZI + 0.25 XX + 0.375 IY, the reference |01>
    and the pool a = i(0.5 II + 0.5 (XX + YY) + 0.25 ZZ),
    b = 0.5 i(XY - YX) (``_vacuum_vector``). On |00> the flips of XX and YY
    cancel (1 - 1 = 0), and so do those of XY and YX (i - i), while II and
    ZZ give a the vacuum phase gamma(t) = t/2 + t/4, and b has none. The
    pairs of a1 = (0, 0.3), a2 = (0, 0.9) and b1 = (1, 0.6):
    ``"vacuum_reference"`` prepares the reference rows ((), (a1)) and
    ((), (a1, b1)) by its reference-row construction and (a1), (a2, b1) by
    the prefix construction. ``"vacuum_restore"`` prepares (a1), (a2, b1),
    whose left chain is shorter, (a1, b1), (a2), whose left chain of two
    noncommuting generators is longer, and the reference row ((), (a2, b1)).
    gamma_j - gamma_i = 0.45 and gamma_i = 0.225 for both general pairs, so a
    missing, sign-reversed or one-signed correction, or an adjoint of the
    left chain in the wrong order, moves an entry far outside its bound. The
    reduction's S and H0 entries of each pair, with no phase alignment, lie
    within ``pair_reducer.entry_bounds`` of the pair's own receipt plus the
    error of the reference vectors: each factor is a phase gate or a
    rotation with one-ulp cos and sin, so after p factors ||v - psi|| <=
    delta_p with 1 + delta_(k+1) = (1 + delta_k)(1 + 4Du)(1 + 2D gamma_2D),
    D = 4, which adds C(delta_i + delta_j + delta_i delta_j), C = 9/8, to an
    H0 bound and the same without C to an S bound. The reference components
    are contracted exactly as rationals. The circuit that ``"vacuum_reference"``
    builds for (a1), (a2, b1), whose recorded route is ``"prefix"``, equals the
    circuit of the same pair under a ``"prefix"`` Plan, so the route that the
    slot law of the Plan bounds is the construction that runs.

    One grouped setting: with A = IY and shots, the real and imaginary
    settings of (a1), (a2, b1) under ``"vacuum_restore"``, with their
    measurements removed, have the ancilla parity Re S and Im S and the
    ancilla-system parity Re and Im of <phi_i|IY|phi_j> within the bound of
    ``_native_tail`` plus the reference error. A correction placed after the
    ancilla's readout rotation would commute with that measurement and be
    lost there. Every emitted angle and phase component of the fixture is
    an exact product of its dyadic coefficients and the angles.
    """
    import json
    from fractions import Fraction as F
    from nwqlib.algorithms.gcim.pair_reducer import entry_bounds
    from nwqlib._prepared_execution import Run, prepare_experiment, submit_experiment

    a1, a2, b1 = (0, .3), (0, .9), (1, .6)
    cases = {
        "vacuum_reference": {((), (a1,)): "vacuum_reference", ((), (a1, b1)): "vacuum_reference",
                             ((a1,), (a2, b1)): "prefix"},
        "vacuum_restore": {((a1,), (a2, b1)): "vacuum_restore", ((a1, b1), (a2,)): "vacuum_restore",
                           ((), (a2, b1)): "vacuum_restore"},
    }
    u = F(1, 2**53)
    growth = (1 + 16 * u) * (1 + 8 * (8 * u / (1 - 8 * u)))

    def oracle(left, right, operator):
        (vl, pl), (vr, pr) = _vacuum_vector(left), _vacuum_vector(right)
        dl, dr = growth**pl - 1, growth**pr - 1
        return _contract(vl, operator, vr), dl + dr + dl * dr

    a0 = sum(c * _pauli(label) for label, c in _VACUUM_TERMS)
    for rule, pairs in cases.items():
        plan = _vacuum_plan(rule)
        assert plan.reconstruction.acquisition.pair_preparation == rule
        with Run(plan) as run:
            for (left, right), route in pairs.items():
                assert left < right
                assert adapt_acquisition.chain_pair_preparation(left, right, rule, (0, 1)).route == route
                handle = prepare_experiment(adapt_acquisition._point(plan, "pair", left, right), run=run)
                if route == "prefix":
                    hybrid = (left, right), handle._restore_native().circuit
                chunk = adapt_acquisition._single_chunk(submit_experiment(handle, run=run))
                values = adapt_acquisition._pair_reduction_values(chunk)
                (parameters,) = (json.loads(p.parameters) for p in chunk.observation.positions
                                 if p.kind == "reduction")
                delta = handle.record.saved_state_error(("amplitude-derived masses",))[0]
                assert delta is not None
                overlap_bound, h0_bound = entry_bounds(delta, float.fromhex(parameters["c1"]),
                                                       1 << parameters["qubits"], parameters["terms"])
                s, reference_error = oracle(left, right, np.eye(4))
                h0, _ = oracle(left, right, a0)
                for actual, target, bound in ((values["overlap"], s, F(overlap_bound) + reference_error),
                                              (values["h0"], h0, F(h0_bound) + F(9, 8) * reference_error)):
                    assert abs(F(actual.real) - target[0]) <= bound and abs(F(actual.imag) - target[1]) <= bound

    (left, right), built = hybrid
    prefix = _vacuum_plan("prefix")
    with Run(prefix) as run:
        point = adapt_acquisition._point(prefix, "pair", left, right)
        assert built == prepare_experiment(point, run=run)._restore_native().circuit

    grouped = _vacuum_plan("vacuum_restore", terms=(("IY", 1.0),), shots=16)
    assert grouped.reconstruction.acquisition.quadratures == "complex"
    s, reference_error = oracle((a1,), (a2, b1), np.eye(4))
    y, _ = oracle((a1,), (a2, b1), _pauli("IY"))
    with Run(grouped) as run:
        for quadrature in (0, 1):
            point = adapt_acquisition._point(grouped, "pair", (a1,), (a2, b1), quadrature=quadrature, group=0)
            (ancilla, joint), bound = _native_tail(prepare_experiment(point, run=run))
            allowance = bound + reference_error
            assert abs(F(ancilla) - s[quadrature]) <= allowance and abs(F(joint) - y[quadrature]) <= allowance


@pytest.mark.parametrize("rule", ["vacuum_reference", "vacuum_restore"])
def test_vacuum_constructions_plan_only_with_a_known_vacuum_phase(rule):
    """ADAPT plans a vacuum construction only when every pool generator has a structural vacuum phase.

    In a = i(0.5 II + 0.5 (XX + YY) + 0.25 ZZ) the words XX and YY flip both
    qubits of |00>, with amplitudes 1 and i**2 = -1 at the one coefficient
    magnitude 0.5, which cancel, and b = 0.5 i(XY - YX) cancels the same way
    (i - i), so the pool of the vacuum fixture qualifies. A lone i XI flips
    |00> with nothing to cancel it. In i(XX + 0.5 YY) the two terms flip
    |00> at the magnitudes 1 and 0.5, so the bucket of magnitude 1 sums to +1
    and that of magnitude 0.5 to i**2 = -1 at the same flip mask, and
    neither is zero, while a sum over the flip mask alone would cancel. A
    pool that holds either generator is refused at planning, with exact
    readout and with shots.
    """
    from nwqlib.algorithms import ApplicabilityError

    lone = (_VACUUM_POOL[0], (("XI", 1j),))
    unequal = (_VACUUM_POOL[0], (("XX", 1j), ("YY", .5j)))
    for shots in (None, 8):
        assert _vacuum_plan(rule, shots=shots).reconstruction.acquisition.pair_preparation == rule
        for pool_rows in (lone, unequal):
            with pytest.raises(ApplicabilityError):
                _vacuum_plan(rule, pool_rows=pool_rows, shots=shots)


@pytest.mark.parametrize("rule", ["vacuum_reference", "vacuum_restore"])
def test_grouped_vacuum_adapt_records_its_routes_and_reanalyzes_without_acquisition(rule, tmp_path, monkeypatch):
    """A sampled vacuum ADAPT run records each pair's construction, saves and reanalyzes, and declares its slot law.

    On the vacuum fixture with 16 shots per setting the run selects both
    generators g1, g2, so its value basis is (), (g1), (g2), (g1, g2).
    ``"vacuum_reference"`` prepares the three pairs of the reference by its
    reference-row construction and the others by the prefix construction,
    with 1 + 0, 1 + 1 and 1 + 0 shared blocks, and ``"vacuum_restore"``
    prepares all six by restoration. The saved Result reloads with the
    resolved acquisition and these records, and a new overlap cutoff
    reanalyzes it without acquisition. Each selected definition declares
    the logical construction slots of the rule for q = 2 qubits, chains of
    at most l = 2 generators, the commuting pool's per-generator bound
    B = 6 q T_max with T_max = 4 terms, and W_ref = q for the occupation
    reference: W = 2 W_ref + 2l(1 + B) + 8q + 7 = 223 is larger than
    W_reference_pair = 2 W_ref + l(1 + B) + l(T_max + 1) + 8q + 7 = 135, so
    ``"vacuum_reference"`` declares max(W_reference_pair, W) = 223, and
    ``"vacuum_restore"`` declares W_restore = 2 W_ref + 3l(1 + B)
    + 12288 l + 2l(T_max + 1) + 8q + 7 = 24917.
    """
    import nwqlib
    from nwqlib.backends import qiskit_aer

    plan = _vacuum_plan(rule, shots=16)
    q, ell, t_max, w_ref = 2, 2, 4, 2
    slots = 6 * q * t_max
    old = 2 * w_ref + 2 * ell * (1 + slots) + 8 * q + 7
    reference_pair = 2 * w_ref + ell * (1 + slots) + ell * (t_max + 1) + 8 * q + 7
    restore = 2 * w_ref + 3 * ell * (1 + slots) + 12288 * ell + 2 * ell * (t_max + 1) + 8 * q + 7
    assert (old, reference_pair, restore) == (223, 135, 24917)
    assert {s.construction_work for s in plan.construction.selections} == {
        max(reference_pair, old) if rule == "vacuum_reference" else restore}
    result = nwqlib.solve(plan, progress=False)
    g1, g2 = zip(result.value_selected, result.value_theta, strict=True)
    if rule == "vacuum_reference":
        expected = {(0, 1): (rule, 0), (0, 2): (rule, 0), (0, 3): (rule, 0),
                    (1, 2): ("prefix", 1), (1, 3): ("prefix", 2), (2, 3): ("prefix", 1)}
    else:
        expected = {(i, j): (rule, 0) for i in range(4) for j in range(i + 1, 4)}
    assert g1 != g2 and result.pencil is not None
    assert {p.pair: (p.route, p.shared_blocks) for p in result.pair_preparations} == expected
    loaded = nwqlib.load_result(result.save(tmp_path / rule))
    assert loaded.plan.reconstruction.acquisition == plan.reconstruction.acquisition
    assert (loaded.pair_preparations, loaded.group_moments) == (result.pair_preparations, result.group_moments)
    monkeypatch.setattr(qiskit_aer, "_submit_aer_execution", lambda *a, **k: pytest.fail("reanalysis acquired"))
    again = loaded.analyze(overlap_cutoff=1e-10)
    assert (again.pair_preparations, again.group_moments) == (result.pair_preparations, result.group_moments)


def _four_shot_growth():
    """Return a complex grouped Plan, the selections (a, b, c) and (a, b, c, d), the pair key of (ab, abc) and a context.

    A = 0.5 II + ZI on two qubits has one group with coefficient 1. The
    context holds a four-shot count chunk for every grouped setting of both
    bases. The imaginary group-0 setting of the canonical pair (ab, abc)
    has x = +1 three times and -1 once with ZI parity +1, so y = x. Every
    other setting has x = +1 and parity +1 on all four shots.
    """
    from nwqlib.algorithms import PencilAcquisition
    from nwqlib.core import Source
    from nwqlib.core.planning import ObservationSpec
    from nwqlib.execution import ObservationChunk, RegisterMap

    plan = plan_for(A=ingest_pauli((("II", .5), ("ZI", 1.0)), num_qubits=2), shots=4, max_iterations=4,
                    pool_rows=tuple(((p, 1j),) for p in ("YI", "IY", "XY", "YX")), initial_state=(1, 0, 0, 0),
                    acquisition=PencilAcquisition(quadratures="complex"))
    a, b, c, d = (0, .1), (1, .2), (2, .3), (3, .4)
    target = ((a, b), (a, b, c), 1, 0, -1)  # imaginary part, group 0, selection stage
    identity = "sha256:" + "1" * 64
    observations = {}
    for selected in ((a, b, c), (a, b, c, d)):
        for _, _, key, point in adapt_acquisition._matrix_entries(plan, adapt_acquisition.basis_chains(selected)):
            point = point()
            # Ancilla at bit 0, system qubit k at bit k + 1: '001' is x = -1 with ZI parity +1.
            table = {"000": 3, "001": 1} if key == target else {"000": 4}
            observations[point.content_id] = ObservationChunk.from_histogram(
                table, returned_shots=4, run_id="r", plan_id=identity, realization_id=point.content_id,
                prepared_id=identity, experiment=point.experiment, setting="s", bindings=point.bindings,
                quantum_layout=(), classical_layout=(RegisterMap(name="c", bits=(0, 1, 2)),), attempt="a",
                job="j", chunk="c", observation=ObservationSpec(kind="counts", shots=4),
                population="unconditional", trajectories=None,
                source=Source(name="supplied", version="1", domain="test", reference="counts"))
    return plan, (a, b, c), (a, b, c, d), target, adapt_acquisition.AdaptContext(observations=observations)


def test_reversed_pair_conjugates_its_cached_moments_once():
    """Growth from three to four selections reverses one cached pair, whose imaginary moments are conjugated once.

    The bases of (a, b, c) and (a, b, c, d) are (), a, b, c, abc, ab and
    (), a, b, c, d, abcd, ab, abc, so the old positions 0 to 5 move to
    (0, 1, 2, 3, 7, 6). The canonical pair (ab, abc) is the upper pair
    (4, 5) of the first basis in reversed order and the upper pair (6, 7)
    of the second in canonical order. Its imaginary group-0 table of four
    shots, x = +1 three times and -1 once with y = x, has the exact moments
    mean = overlap mean = label mean = 1/2, second = cross = 1, and mean
    variance = overlap variance = covariance = (1 - 1/4)/3 = 1/4. Reversal
    conjugates the entry, so the (4, 5) view has first moments -1/2 and the
    same second and cross moments, variances and covariance, while the
    (6, 7) view, oriented again from the cached canonical moment, has +1/2.
    The pooled entries and the assembled H0 and S carry these signs above
    the diagonal and the opposite below. A doubled conjugation, positions
    moved without conjugation, a covariance sign flip or unflipped label
    means fails.
    """
    plan, k3, k4, target, context = _four_shot_growth()
    old, new = adapt_acquisition.basis_chains(k3), adapt_acquisition.basis_chains(k4)
    a, b, c, d = k4
    assert old == ((), (a,), (b,), (c,), (a, b, c), (a, b))
    assert new == ((), (a,), (b,), (c,), (d,), (a, b, c, d), (a, b), (a, b, c))
    assert [new.index(chain) for chain in old] == [0, 1, 2, 3, 7, 6]
    moments3, pairs3, _ = adapt_acquisition._grouped_statistics(plan, old, context)
    raw = context.pair_values[target][0]
    moments4, pairs4, _ = adapt_acquisition._grouped_statistics(plan, new, context)

    def view(moments, pair):
        return next(m for m in moments if m.pair == pair and m.quadrature == "imag" and m.group == 0)

    assert raw.pair == (0, 1)
    for moment, sign in ((raw, 1), (view(moments3, (4, 5)), -1), (view(moments4, (6, 7)), 1)):
        assert (moment.label_means, moment.mean, moment.overlap_mean) == ((sign * .5,), sign * .5, sign * .5)
        assert (moment.second, moment.overlap_cross, moment.shots) == (1.0, 1.0, 4)
        assert (moment.mean_variance, moment.overlap_variance, moment.mean_covariance) == (.25, .25, .25)
    assert (pairs3[4, 5][0].imag, pairs3[4, 5][1].imag, pairs4[6, 7][0].imag, pairs4[6, 7][1].imag) == (
        -.5, -.5, .5, .5)
    h3, s3, sampled, _ = adapt_acquisition._matrix_from_observations(plan, old, context)
    assert sampled and (h3[4, 5].imag, s3[4, 5].imag, h3[5, 4].imag, s3[5, 4].imag) == (-.5, -.5, .5, .5)
    h4, s4, _, _ = adapt_acquisition._matrix_from_observations(plan, new, context)
    assert (h4[6, 7].imag, s4[6, 7].imag, h4[7, 6].imag, s4[7, 6].imag) == (.5, .5, -.5, -.5)


def test_grouped_stage_is_refused_before_its_first_acquisition():
    """A grouped matrix stage whose complete pass exceeds max_products is refused before any of its queries run.

    H = ZI + IZ + 0.5 XX + 0.25 YI has a word with one Y, so ADAPT measures
    both parts, b**2 G settings with G = 3 groups. With 32 shots the pass
    of the third stage (b = 6), which charges min(32, 2**3) = 8 bins to
    every table not yet acquired, exceeds max_products = 30000. The Run then
    holds exactly the 4**2 * 3 = 48 settings of the earlier bases and none
    of the third stage's, and its checkpoint stays in that stage's matrix
    phase with three selections. The refused work is the grouped pass work
    of the whole stage, the law of
    ``nwqlib.algorithms.gcim.fixed_basis.grouped_pass_requirements`` and of
    the docs/ENGINEERING_CONSTANTS.md row "Sampled FixedGCIM analysis
    admission", evaluated term by term here: n = 2, c = 3,
    w = 1, L = 4, b = 6, C = M = 108 chunks and settings (each group in
    b**2 = 36 of them) on P = 21 pairs, and o = 1 for ADAPT's oriented
    view. Each of the 48 cached tables has the bins of its saved chunk and
    the size of the group in its bindings, and each of the 60 others has 8.
    """
    from nwqlib._prepared_execution import Run

    plan = _real_adapt(terms=(("ZI", 1.0), ("IZ", 1.0), ("XX", .5), ("YI", .25)), max_products=30000)
    rec = plan.reconstruction
    assert rec.acquisition.quadratures == "complex" and len(rec.energy_groups) == 3
    with Run(plan) as run:
        with pytest.raises(ValueError, match="exceeding max_products=30000") as refused:
            run.wait()
        cached = [c for c in run.observations.chunks if c.experiment in ("pair", "diagonal")]
        assert len(cached) == 4**2 * 3
        assert run.checkpoint_state["phase"] == "matrix" and len(run.checkpoint_state["selected"]) == 3
    n, b, size = rec.num_qubits, 6, [len(group) for group in rec.energy_groups]
    c, groups = n + 1, [next(x.value for x in chunk.bindings if x.parameter == "group") for chunk in cached]
    w = -(-c // 64)
    tables = [(size[g], len(chunk.values)) for g, chunk in zip(groups, cached, strict=True)]
    tables += [(size[g], min(plan.shots, 2**c)) for g in range(len(size)) for _ in range(b * b - groups.count(g))]
    count, pairs, terms = len(tables), b * (b + 1) // 2, sum(s for s, _ in tables)
    work = (sum(m * (c + 44 + (4 * w + 4) * s) + s * ((c - 1) * w + 1) for s, m in tables)
            + len(rec.nonidentity_terms) * (n + 1) + 32 * b**2 + 16 * b**3
            + (terms + 2 * count) + 23 * max(count, b * b * len(size)) + 22 * pairs)
    assert str(refused.value) == f"action needs {work} scalar products, exceeding max_products=30000"


def test_native_depth_limit_stops_adapt_on_its_last_solved_pencil(tmp_path, monkeypatch, ibm_runtime_spy):
    """A native two-qubit depth limit stops grouped ADAPT before the first deeper query and keeps the earlier rounds.

    Grouped ADAPT on H = ZI + IZ + 0.5 XX from |00>, with the pool iXY,
    iYX, one selection and an optimization after it, runs on the offline
    FakeManilaV2 device. Without a limit its first queries are the
    reference's group settings and screening settings, whose ISA circuits
    have no two-qubit gate, and then the energy settings of the selected
    product state exp(theta iP)|00>, whose Pauli rotation needs CX gates.
    A limit one below the recorded depth of the first energy setting admits
    the earlier queries and refuses that energy setting after its
    preparation and before its job. The run ends with stop_reason
    native_depth_refused on the reference-only pencil of its first round,
    with the selected generator and no second projected basis, and pending
    names the refused preparation, which has no attempt. The earlier counts
    and shots stay in the Run, and the reopened Run returns the same Result
    without compiling or submitting. A Run that stops after saving its
    complete checkpoint and before its Result, here because its final
    analysis raises once, is completed in the same way when reopened, with
    the same stop, energy and refused preparation. A limit equal to the deepest recorded
    depth admits every query. Such a Run, prepared and then reopened where
    another Qiskit version is reported, raises NativeDepthCapRefusal with
    its provenance before any job and keeps its phase and pending query.
    Reopened where the recorded versions are installed, it completes with
    the queries, depths and stop reason of the run without a limit. Only
    qiskit-ibm-runtime 0.49.0 qualifies an IBM depth, so under another
    installed version no depth is qualified and a limit refuses the first
    query before its job. The synthetic counts choose the generator but
    decide no relation tested here.
    """
    import importlib.metadata

    import nwqlib
    from nwqlib.backends import IBMRuntimeBackend
    from nwqlib.backends.inspection import NativeDepthCapRefusal
    from nwqlib.problems import ingest_occupation

    spy, shots = ibm_runtime_spy, 8
    problem = nwqlib.Eigenproblem(A=ingest_pauli((("ZI", 1.0), ("IZ", 1.0), ("XX", .5)), num_qubits=2))

    def selected_with(limit):
        method = ADAPT(initial_state=ingest_occupation((0, 0), num_qubits=2),
                       pool=tuple(ingest_pauli(((p, 1j),), num_qubits=2) for p in ("XY", "YX")),
                       theta=pi / 8, max_iterations=1, gradient_norm_floor=0.0, optimize_every_m=1,
                       optimize_rounds_n=1, optimize_max_evaluations=1, max_native_two_qubit_depth=limit)
        return nwqlib.plan(problem, method=method, execution="quantum", shots=shots, seed=7)

    def finish(run):
        while run.result is None:
            run.resume()
        return run

    def run_with(limit, directory):
        return finish(nwqlib.submit(nwqlib.prepare(selected_with(limit), backend=spy.connection,
                                                   directory=directory, progress=False)))

    def queries(run):
        return [(a.realization.experiment, a.native_depth.native_two_qubit_depth) for a in run.prepared_artifacts]

    with run_with(None, tmp_path / "unlimited") as run:
        assert run.result.stop_reason != "native_depth_refused"
        recorded, stop = [a.native_depth for a in run.prepared_artifacts], run.result.stop_reason
        unlimited = queries(run)
    installed = importlib.metadata.version("qiskit-ibm-runtime")
    if installed != "0.49.0":
        assert all(not depth.qualified and f"qiskit-ibm-runtime {installed} is installed" in depth.reason
                   for depth in recorded)
        requests = len(spy.api.requests)
        with run_with(0, tmp_path / "unchecked") as run:
            assert run.result.stop_reason == "native_depth_refused" and run.result.history == ()
            (label,) = run.result.pending
            assert run.prepared_artifacts[0].content_id in label
            assert run.trace.events == () and len(spy.api.requests) == requests
        return
    assert all(depth.qualified for depth in recorded)
    kinds = [kind for kind, _ in unlimited]
    first = kinds.index("energy")
    assert set(kinds[:first]) == {"diagonal", "screen"} and {depth for _, depth in unlimited[:first]} == {0}
    limit, requests = unlimited[first][1] - 1, len(spy.api.requests)
    assert limit >= 0

    path = tmp_path / "refused"
    with run_with(limit, path) as run:
        result, artifacts = run.result, run.prepared_artifacts
        assert queries(run) == unlimited[:first + 1]
        assert all(event.status == "completed" for event in run.trace.events)
        assert {event.prepared_id for event in run.trace.events} == {a.content_id for a in artifacts[:first]}
        assert run.exposure["completed"]["shots"] == first * shots and len(spy.api.requests) == requests + first
        assert result.stop_reason == "native_depth_refused"
        (label,) = result.pending
        assert artifacts[first].content_id in label
        assert len(result.history) == 1 and len(result.selected) == 1 and result.value_selected == ()
        assert result.pencil is not None and result.eigenvalue == result.history[0].energy

    # A process that stops after the refusal saved its complete checkpoint and before the Result exists,
    # here a final analysis that raises once, leaves a Run that the reopen leg below completes.
    interrupted = tmp_path / "interrupted"
    prepared = nwqlib.prepare(selected_with(limit), backend=spy.connection, directory=interrupted, progress=False)
    with prepared.run as run, monkeypatch.context() as patch:
        def halt(*args, **kwargs):
            raise RuntimeError("stopped before the Result")

        patch.setattr(adapt_acquisition, "analyze_adapt", halt)
        with pytest.raises(RuntimeError, match="stopped before the Result"):
            finish(nwqlib.submit(prepared))
        assert run.result is None and run.checkpoint_state["phase"] == "complete"
        stopped = (len(run.prepared_artifacts), len(run.trace.events), run.prepared_artifacts[first].content_id)

    moved = tmp_path / "moved"
    with nwqlib.prepare(selected_with(max(depth for _, depth in unlimited)), backend=spy.connection,
                        directory=moved, progress=False).run as run:
        saved, (prepared,) = run.checkpoint_state, run.prepared_artifacts
    sent, real = len(spy.api.requests), importlib.metadata.version
    with monkeypatch.context() as patch:
        patch.setattr(importlib.metadata, "version", lambda name: "9.9.9" if name == "qiskit" else real(name))
        with nwqlib.load_run(moved, backend=spy.connection, progress=False) as run:
            with pytest.raises(NativeDepthCapRefusal) as refused:
                run.resume()
            assert refused.value.prepared_id == prepared.content_id
            assert "qiskit 9.9.9 is installed" in refused.value.provenance
            assert run.result is None and run.trace.events == () and len(spy.api.requests) == sent
            assert [run.checkpoint_state[key] for key in ("phase", "pending", "active_energy")] == [
                saved[key] for key in ("phase", "pending", "active_energy")]
    with finish(nwqlib.load_run(moved, backend=spy.connection, progress=False)) as run:
        assert queries(run) == unlimited and run.result.stop_reason == stop
    monkeypatch.setattr(IBMRuntimeBackend, "prepare", lambda *a, **k: pytest.fail("reopening compiled again"))
    sent = len(spy.api.requests)
    with nwqlib.load_run(path, backend=spy.connection, progress=False) as run:
        run.resume()
        assert (run.result.stop_reason, run.result.pending) == ("native_depth_refused", result.pending)
        assert len(run.trace.events) == first and len(spy.api.requests) == sent
    with nwqlib.load_run(interrupted, backend=spy.connection, progress=False) as run:
        run.resume()
        (label,) = run.result.pending
        assert (run.result.stop_reason, run.result.eigenvalue) == ("native_depth_refused", result.eigenvalue)
        assert stopped[2] in label and (len(run.prepared_artifacts), len(run.trace.events)) == stopped[:2]
        assert len(spy.api.requests) == sent


def _bounded_plan(**growth):
    """Exact quantum bounded growth on H = 0.5 ZZ + ZI + 0.7 IZ + 0.3 XX + 0.2 XI from |00>, pool iYI, iIY, iXY, theta = pi/8."""
    from nwqlib import Eigenproblem, plan
    from nwqlib.algorithms import BoundedRitzGrowth
    from nwqlib.problems import ingest_occupation

    terms = (("ZZ", .5), ("ZI", 1.0), ("IZ", .7), ("XX", .3), ("XI", .2))
    method = ADAPT(initial_state=ingest_occupation((0, 0), num_qubits=2),
                   pool=tuple(ingest_pauli(((p, 1j),), num_qubits=2) for p in ("YI", "IY", "XY")),
                   theta=pi / 8, max_iterations=3, basis_growth=BoundedRitzGrowth(**growth))
    return plan(Eigenproblem(A=ingest_pauli(terms, num_qubits=2)), method=method, seed=7), terms


def test_bounded_growth_builds_short_chains_and_pays_for_every_scored_extension():
    """Exact bounded growth adds one examined extension per round, at most three generators deep, under a CX limit.

    With BoundedRitzGrowth(max_chain_length=3, basis_depth_limit=2) the
    one-qubit rotations exp(theta iYI) and exp(theta iIY) lower to no CX,
    and exp(theta iXY) to one basis change, CX, RZ, CX and the inverse basis
    change, two CX. A chain's bound is twice its number of iXY factors, so
    every chain with two of them, such as the adjacent repeat (iXY, iXY),
    is excluded with bound 4 and obtains no matrix element. Each round's
    winner is the extension with the largest score of an independent dense
    computation: the chain states are ordered products of
    cos(theta) I + sin(theta) iP, Q is an orthonormal basis of the kept
    span of the current states, q the normalized remainder of a candidate
    v outside it, and the score (sqrt(delta**2 + 4 rho**2) - delta)/2 with
    rho = |<q|H - E|psi>| and delta = <q|H|q> - E for the current lowest
    Ritz pair (E, psi). Every winner leads the next score by more than
    0.02, far above rounding. The basis grows as (), (iXY), (iXY, iIY),
    (iXY, iIY, iYI): the reference has three extensions, and each later
    round extends the last chain only, because earlier parents have no pool
    index left. iIY applied twice to (iXY) stays in the span of (iXY) and
    (iXY, iIY), so that candidate is rejected by the independence cutoff.
    Each scored extension of a b-state basis costs b + 1 exact pair
    reductions (its b off-diagonal pairs and its diagonal), whether or not
    it is added, and adding the winner obtains nothing new, so the rounds
    acquire 1 + 3 * 2, 2 * 3, 2 * 4 and 0 queries. Each round records the
    decrease of its energy from the previous round's, the exact binary64
    difference of the two recorded energies. On
    the final basis the window holds positions 0, 1 and 2, and the parents'
    rotation continues across rounds: a round stopped after one examined
    extension and a round of four more examine, in order, the five
    extensions of one round of five, (c), (a, c), (a, b, c) already in the
    basis, (b) and (a, b) already in the basis. With independence_cutoff
    0.2 the reference round scores none of its three singles. With iXY
    twice in the pool, the earlier of two equal scores wins.
    """
    import nwqlib
    from qiskit.quantum_info import SparsePauliOp

    plan, terms = _bounded_plan(max_chain_length=3, basis_depth_limit=2)
    result = nwqlib.solve(plan, progress=False)
    t = pi / 8
    hamiltonian = SparsePauliOp.from_list(terms).to_matrix()
    generators = [1j * SparsePauliOp(p).to_matrix() for p in ("YI", "IY", "XY")]

    def state(chain):
        vector = np.eye(4, dtype=complex)[0]
        for index, angle in chain:
            vector = (np.cos(angle) * np.eye(4) + np.sin(angle) * generators[index]) @ vector
        return vector

    def ritz(basis):
        states = np.array([state(chain) for chain in basis]).T
        values, vectors = np.linalg.eigh(states.conj().T @ states)
        q = states @ (vectors / np.sqrt(values))
        energies, ritz = np.linalg.eigh(q.conj().T @ hamiltonian @ q)
        return energies[0], q @ ritz[:, 0], q

    def scores(basis, chains):
        energy, psi, q = ritz(basis)
        out = []
        for chain in chains:
            rest = state(chain) - q @ (q.conj().T @ state(chain))
            rest /= np.linalg.norm(rest)
            rho = abs(np.vdot(rest, (hamiltonian - energy * np.eye(4)) @ psi))
            delta = np.vdot(rest, hamiltonian @ rest).real - energy
            out.append((np.sqrt(delta**2 + 4 * rho**2) - delta) / 2)
        return out

    a, b, c = (2, t), (1, t), (0, t)
    assert result.basis == ((), (a,), (a, b), (a, b, c)) == result.value_basis
    assert result.selected == result.value_selected == () and result.stop_reason == "max_iterations"
    rounds = [[(row.chain, row.reason, row.depth_bound) for row in r.candidates] for r in result.history]
    assert rounds == [
        [((c,), "ranked", 0), ((b,), "ranked", 0), ((a,), "accepted", 2)],
        [((a, c), "ranked", 2), ((a, b), "accepted", 2), ((a, a), "above_depth_limit", 4)],
        [((a, b, c), "accepted", 2), ((a, b, b), "below_independence_cutoff", 2), ((a, b, a), "above_depth_limit", 4)],
        [],
    ]
    for size, row in zip((1, 2, 3), result.history):
        ranked = [x for x in row.candidates if x.reason in ("accepted", "ranked")]
        independent = scores(result.basis[:size], [x.chain for x in ranked])
        best = int(np.argmax(independent))
        assert ranked[best].reason == "accepted" and sorted(independent)[-1] - sorted(independent + [0])[-2] > .02
    pairs = {chunk.content_id for chunk in result.data.observations.chunks if chunk.experiment == "pair"}
    assert [len(set(row.contribution_ids) & pairs) for row in result.history] == [7, 6, 8, 0]
    assert len(pairs) == len(result.data.observations.chunks) == 21
    assert [row.kept_rank for row in result.history] == [1, 2, 3, 4]
    energies = [row.energy for row in result.history]
    assert [row.energy_decrease for row in result.history] == [None, *(x - y for x, y in zip(energies, energies[1:]))]

    from nwqlib.algorithms import BoundedRitzGrowth
    from nwqlib.algorithms.gcim.bounded_growth import frontier

    def examine(cursor, limit):
        growth = BoundedRitzGrowth(max_chain_length=3, candidates_per_round=5, max_candidate_scans=limit)
        return frontier(result.basis, cursor, pool_size=3, theta=t, growth=growth)

    whole, first = examine((), 5), examine((), 1)
    second = examine(first[1], 4)
    assert [(x["chain"], x["reason"]) for x in whole[0]] == [
        ((c,), "candidate"), ((a, c), "candidate"), ((a, b, c), "duplicate"), ((b,), "candidate"),
        ((a, b), "duplicate")]
    assert first[0] + second[0] == whole[0] and second[1:] == whole[1:]

    # A Method-level independence_cutoff of 0.2 exceeds eta = sin(pi/8)**2 = 0.146 of every single
    # cos(pi/8)|00> - sin(pi/8)|x>, with |x> orthogonal to |00>, so nothing is scored.
    from nwqlib import Eigenproblem
    from nwqlib.problems import ingest_occupation

    def run(growth, pool=("YI", "IY", "XY"), execution="classical", max_iterations=3):
        method = ADAPT(initial_state=ingest_occupation((0, 0), num_qubits=2), theta=t, max_iterations=max_iterations,
                       pool=tuple(ingest_pauli(((p, 1j),), num_qubits=2) for p in pool), basis_growth=growth)
        return nwqlib.solve(Eigenproblem(A=ingest_pauli(terms, num_qubits=2)), method=method,
                            execution=execution, seed=7, progress=False)

    stopped = run(BoundedRitzGrowth(max_chain_length=3, independence_cutoff=.2))
    assert stopped.stop_reason == "no_admissible_candidate" and stopped.basis == ((),)
    assert [x.reason for x in stopped.history[0].candidates] == ["below_independence_cutoff"] * 3
    # Repeating iXY in the pool makes its first two singles one state with bitwise-equal scores,
    # and the earlier examined one is added.
    for execution in ("classical", "quantum"):
        tie = run(BoundedRitzGrowth(max_chain_length=1), pool=("XY", "XY", "YI"), execution=execution,
                  max_iterations=1)
        first, second, _ = tie.history[0].candidates
        assert first.score == second.score and (first.reason, second.reason, tie.basis[1]) == (
            "accepted", "ranked", ((0, t),)), execution


def test_frontier_cursor_ends_with_the_parent_examined_last_and_wraps_to_the_reference():
    """The saved `frontier_cursor` ends with the parent whose extension a round examined last.

    `result.frontier_cursor` holds a (basis position, next pool index) pair
    for each parent of the last round, and the next round, also after
    saving and resuming, starts at the first parent after the last pair's
    parent, wrapping to the reference. The expected labels and cursors
    follow from that rule by hand, with a, b and c the generators of pool
    indices 0, 1 and 2 at one angle:

    - Basis (), (a,) with the cursor ((1, 0), (0, 1)), pool size 2 and two
      scans: the round starts after the reference at (a,), examines (a, a),
      then extends the reference with its last pool index to (b,). The
      reference, now without an index left, stays last, so the next round
      starts at (a,) again with (a, b).
    - Basis (), (a,), (a, b) with the cursor ((0, 2), (1, 2)), pool size 2,
      parent_window 2 and max_chain_length 2: (a, b) is too long to be a
      parent and the reference has no index left, so the round examines
      nothing and returns its cursor unchanged.
    - A consistent history with max_chain_length 2, pool size 3 and one
      extension per round adds (a,), then (a, a). In the third round (a, a)
      is no parent, so (a,), the parent of the last pair, is the largest
      position of the window, and the round wraps to the reference, which
      takes (b,).
    """
    from nwqlib.algorithms import BoundedRitzGrowth
    from nwqlib.algorithms.gcim.bounded_growth import frontier

    t = pi / 8
    a, b = (0, t), (1, t)

    def examine(basis, cursor, pool_size, **growth):
        labels, following, exhausted = frontier(basis, cursor, pool_size=pool_size, theta=t,
                                                growth=BoundedRitzGrowth(**{"max_chain_length": 3, **growth}))
        return [label["chain"] for label in labels], following, exhausted

    chains, cursor, _ = examine(((), (a,)), ((1, 0), (0, 1)), 2, max_candidate_scans=2)
    assert (chains, cursor) == ([(a, a), (b,)], ((1, 1), (0, 2)))
    assert examine(((), (a,)), cursor, 2, candidates_per_round=1)[0] == [(a, b)]
    assert examine(((), (a,), (a, b)), ((0, 2), (1, 2)), 2, parent_window=2, max_chain_length=2) == (
        [], ((0, 2), (1, 2)), True)
    short, cursor = dict(candidates_per_round=1, max_chain_length=2), ()
    for basis, chain, written in ((((),), (a,), ((0, 1),)),
                                  (((), (a,)), (a, a), ((0, 1), (1, 1))),
                                  (((), (a,), (a, a)), (b,), ((1, 1), (0, 2)))):
        chains, cursor, exhausted = examine(basis, cursor, 3, **short)
        assert (chains, cursor, exhausted) == ([chain], written, False)


def test_bounded_rows_reuse_the_screened_generator_circuits_of_their_round(monkeypatch):
    """A bounded round's queries build each generator circuit once, reusing the uncontrolled one built to read its CX count.

    The guide states that, while a round obtains the matrix elements of its
    evaluated extensions without interruption, their queries reuse the
    uncontrolled generator circuits that depth screening built and build
    the controlled and inverse forms separately. A generator circuit is
    built for one (generator, angle, controlled, inverse), so in a run in
    which every generator item is examined in one round only and never
    leaves the basis, each built form is built once. With parent_window=1
    every round examines singles of the reference in pool order:

    - One extension per round on the pool iYI, iIY, iXY adds (0,), (1,) and
      (2,) in three rounds, and every form of every generator circuit is
      built once.
    - Two extensions per round on the pool iYI, iIY, iXY, iYX evaluates
      (0,) and (1,) in the first round and (2,) and (3,) in the second, so
      the second extension's screened circuit must also outlive the queries
      of the first extension of its round.

    Each run, without a CX limit and with basis_depth_limit=10, which every
    chain meets, builds one uncontrolled forward circuit for each generator
    item. The Result does not keep the set of pending generator items of a
    round, as the bounded resume keeper of tests/test_run_archive.py checks.
    """
    from collections import Counter

    import nwqlib
    from nwqlib import Eigenproblem
    from nwqlib.algorithms import BoundedRitzGrowth
    from nwqlib.problems import ingest_occupation
    from nwqlib.subroutines import fermionic_circuits

    builds, build = Counter(), fermionic_circuits.build_generator_circuit

    def counting(generator, theta, **options):
        builds[id(generator), theta, options.get("controlled"), options.get("inverse")] += 1
        return build(generator, theta, **options)

    monkeypatch.setattr(fermionic_circuits, "build_generator_circuit", counting)
    t = pi / 8
    terms = (("ZZ", .5), ("ZI", 1.0), ("IZ", .7), ("XX", .3), ("XI", .2))
    for pool, per_round, rounds in ((("YI", "IY", "XY"), 1, [[(0,)], [(1,)], [(2,)]]),
                                    (("YI", "IY", "XY", "YX"), 2, [[(0,), (1,)], [(2,), (3,)]])):
        for limit in (None, 10):
            builds.clear()
            growth = BoundedRitzGrowth(max_chain_length=3, candidates_per_round=per_round, parent_window=1,
                                       basis_depth_limit=limit)
            method = ADAPT(initial_state=ingest_occupation((0, 0), num_qubits=2), theta=t,
                           pool=tuple(ingest_pauli(((p, 1j),), num_qubits=2) for p in pool),
                           max_iterations=len(rounds), basis_growth=growth)
            result = nwqlib.solve(Eigenproblem(A=ingest_pauli(terms, num_qubits=2)), method=method, seed=7,
                                  progress=False)
            examined = [[tuple(index for index, _ in row.chain) for row in r.candidates] for r in result.history]
            assert examined[:len(rounds)] == rounds, (pool, limit)
            forward = [n for (_, _, controlled, inverse), n in builds.items() if not controlled and not inverse]
            assert forward == [1] * len(pool), (pool, limit)
            if per_round == 1:
                assert result.basis == ((), ((0, t),), ((1, t),), ((2, t),)) and set(builds.values()) == {1}


def test_bounded_score_of_a_nonorthogonal_candidate_matches_its_exact_projection():
    """The exact score of a complex candidate against a nonorthogonal current basis, from the actual solve, by hand.

    In C^3 take A0 = [[2, i, 1/2], [-i, 2, i/2], [1/2, -i/2, 3]] and the
    current states phi0 = e0 and phi1 = (3 e0 + 4i e1)/5, with overlap 3/5.
    They span e0 and e1, where A0 is [[2, i], [-i, 2]] with lowest
    eigenvalue E = 1 and Ritz state psi = (e0 + i e1)/sqrt(2). The candidate
    v = (2, 2, i)/3 overlaps both, its part outside the span is (0, 0, i/3),
    so eta = 1/9 and q = i e2. Then <q|A0 - E|psi> = -i (1/2 + (-i/2) i)/sqrt(2)
    has rho = 1/sqrt(2), delta = <q|A0|q> - E = 2, and the score is
    (sqrt(4 + 2) - 2)/2 = (sqrt(6) - 2)/2. The kernel reads the candidate's
    columns h0 = Phi^dagger A0 v and s = Phi^dagger v in that orientation,
    the solve's coefficient vectors and its offset-free lowest value. The
    allowance 2**-40 is an empirical comparison allowance, not a qualified
    rounding bound. A conjugated column, a missing sqrt(eta) or a
    wrong sign of t^dagger h_r moves a value by more than 0.05.
    """
    from math import sqrt
    from nwqlib.algorithms.gcim.bounded_growth import candidate_score, current_projection
    from nwqlib.algorithms.gcim.fixed_basis import solve_pencil

    a0 = np.array([[2, 1j, .5], [-1j, 2, .5j], [.5, -.5j, 3]])
    phi = np.array([[1, 0, 0], [.6, .8j, 0]]).T
    v = np.array([2 / 3, 2 / 3, 1j / 3])
    h0 = phi.conj().T @ a0 @ phi
    pencil = solve_pencil(h0, phi.conj().T @ phi, cutoff=1e-12, sampled=False)
    current = current_projection(h0, [[complex(z.real, z.imag) for z in row] for row in pencil.coordinate_vectors],
                                 pencil.offset_free_eigenvalues[0])
    score = candidate_score(current, phi.conj().T @ a0 @ v, phi.conj().T @ v, np.vdot(v, a0 @ v).real, 1e-12)
    assert score["reason"] == "ranked"
    for key, expected in (("eta", 1 / 9), ("coupling", sqrt(.5)), ("gap", 2.0), ("score", (sqrt(6) - 2) / 2)):
        assert score[key] == pytest.approx(expected, rel=0, abs=2**-40)
    # An independence cutoff of 1/8 exceeds eta = 1/9 by far more than rounding.
    rejected = candidate_score(current, phi.conj().T @ a0 @ v, phi.conj().T @ v, 3.0, 1 / 8)
    assert (rejected["reason"], rejected["score"]) == ("below_independence_cutoff", None)


def test_zero_bounded_score_does_not_rule_out_an_energy_decrease():
    """An arrowhead candidate scores zero although adding it lowers the lowest eigenvalue by one.

    The current basis e0, e1 is orthonormal with H0 = diag(0, 2), so the
    lowest Ritz pair is E = 0 with psi = e0. The candidate e2 couples only
    to e1, with h0 = (0, 3) and diagonal 2, so rho = 0, delta = 2 and the
    score is exactly zero, while the enlarged matrix
    [[0, 0, 0], [0, 2, 3], [0, 3, 2]] has the eigenvalues -1, 0 and 5. The
    score is a lower bound on the gain of the nested expansion, so the
    candidate stays ranked and a zero score is no reason to stop.
    """
    from nwqlib.algorithms.gcim.bounded_growth import candidate_score, current_projection
    from nwqlib.algorithms.gcim.fixed_basis import solve_pencil

    h0 = np.diag([0.0, 2.0]).astype(complex)
    pencil = solve_pencil(h0, np.eye(2, dtype=complex), cutoff=1e-12, sampled=False)
    current = current_projection(h0, [[complex(z.real, z.imag) for z in row] for row in pencil.coordinate_vectors],
                                 pencil.offset_free_eigenvalues[0])
    score = candidate_score(current, np.array([0, 3]), np.zeros(2), 2.0, 1e-12)
    assert (score["reason"], score["score"], score["coupling"], score["gap"], score["eta"]) == (
        "ranked", 0.0, 0.0, 2.0, 1.0)
    enlarged = np.array([[0, 0, 0], [0, 2, 3], [0, 3, 2]], dtype=float)
    assert np.linalg.eigvalsh(enlarged)[0] == pytest.approx(-1.0, rel=0, abs=1e-14)


def _sampled_bounded():
    """Bounded growth on H = ZI + 0.5 IZ + 0.3 XX from |00>, pool a = iIY and b = (i/16) YI, theta = pi/2, chains of at most 2."""
    from nwqlib import Eigenproblem
    from nwqlib.algorithms import BoundedRitzGrowth
    from nwqlib.problems import ingest_occupation

    pool = (ingest_pauli((("IY", 1j),), num_qubits=2), ingest_pauli((("YI", 1j / 16),), num_qubits=2))
    method = ADAPT(initial_state=ingest_occupation((0, 0), num_qubits=2), pool=pool, theta=pi / 2, max_iterations=3,
                   basis_growth=BoundedRitzGrowth(max_chain_length=2))
    return Eigenproblem(A=ingest_pauli((("ZI", 1.0), ("IZ", .5), ("XX", .3)), num_qubits=2)), method


def _sampled_eligibility_oracle(pencil, z, extensions):
    """Return the sampled eligibility fields of each extension of a one- or two-state real basis from exact inputs.

    ``pencil`` is the current solve and ``z`` the exact overlap S_01 of a
    two-state basis, or None for one state with S = [1]. Each extension is
    ``(s, vz, proxies)``: its exact overlaps with the basis states, the
    variance proxy of S_01 (0 without that measurement) and the variance
    proxies of its overlaps, each the exact value of the binary64 number
    that the eligibility code reads. The test accepts the solve's map C only
    when the largest absolute row sum g of C^dagger S C - I, contracted in
    exact rationals of C's entries, and the distance l of the smallest kept
    overlap eigenvalue from 1 - |z| are both at most
    B0 = gram_formation_allowance(2, 2) < 2**-40. These gates are the test's
    requirement on the computed map, not an error bound of the eigensolver.
    With them, |z| <= 1/2, |s_i| <= 1 and every proxy in [0, 1/16], the
    binary64 rounding of the eligibility fields and of this oracle together
    stays below 2**-13. The oracle forms a = S^(-1) s, q = s^T S^(-1) s,
    sigma**2 = (2 a0 a1)**2 vz + 4 a0**2 V0 + 4 a1**2 V1,
    b_s = (V0 + V1)/(1 - z**2) and lambda = 1 - |z| in fractions, and
    converts to binary64 only for each square root and what follows it. A
    one-state basis is the case z = 0 with s1 = V1 = 0.
    """
    from fractions import Fraction as F
    from math import isfinite

    from nwqlib._projected_eigensolver import gram_formation_allowance

    overlap = [[F(1)]] if z is None else [[F(1), z], [z, F(1)]]
    columns = list(zip(*pencil.coordinate_vectors, strict=True))
    spectrum = pencil.overlap_eigenvalues
    assert pencil.kept_rank == len(columns) == len(pencil.coordinate_vectors) == len(overlap)
    assert all(isfinite(x.real) and isfinite(x.imag) for column in columns for x in column)
    assert all(isfinite(value) for value in spectrum)
    gram = [[_contract(left, overlap, right) for right in columns] for left in columns]
    g = max(sum(abs(real - (i == j)) + abs(imag) for j, (real, imag) in enumerate(row)) for i, row in enumerate(gram))
    z = F(0) if z is None else z
    bound = gram_formation_allowance(2, 2)
    assert abs(z) <= F(1, 2) and bound < 2**-40
    assert g <= bound and abs(F(min(spectrum[-pencil.kept_rank:])) - (1 - abs(z))) <= bound, (g, spectrum)
    expected = []
    for s, vz, proxies in extensions:
        assert all(abs(x) <= 1 for x in s) and all(0 <= v <= F(1, 16) for v in (vz, *proxies))
        (s0, s1), (v0, v1) = (*s, 0)[:2], (*proxies, 0)[:2]
        det = 1 - z * z
        a0, a1 = (s0 - z * s1) / det, (s1 - z * s0) / det
        q = (s0 * s0 - 2 * z * s0 * s1 + s1 * s1) / det
        sigma = sqrt(float((2 * a0 * a1) ** 2 * vz + 4 * a0**2 * v0 + 4 * a1**2 * v1))
        beta = 3 * sqrt(float(vz))
        epsilon = beta / float(1 - abs(z))
        d = 3 * sqrt(float((v0 + v1) / det))
        resolution = 3 * sigma + (d + epsilon * sqrt(float(q))) ** 2 / (1 - epsilon)
        eta = float(1 - q)
        expected.append(dict(eta=eta, metric_beta=beta, metric_epsilon=epsilon, eta_sigma=sigma,
                             extension_noise=float((v0 + v1) / det), eta_resolution=resolution,
                             eta_margin=eta - resolution))
    return expected


def test_sampled_bounded_growth_scores_only_extensions_with_resolved_independence():
    """With shots a nearly dependent extension is unresolved, a distinguishable one is scored, and an unresolved round ends growth.

    At theta = pi/2, exp(theta a)|00> = -|01> is orthogonal to |00>, while
    exp(theta b)|00> = cos(pi/32)|00> - sin(pi/32)|10> has eta =
    sin(pi/32)**2 = 0.0096. Exact readout scores both, and b wins with the
    score 2 = -(<10|H|10> - <00|H|00>). With 256 shots per setting the
    default grouped measurement uses G = 2 real-only groups. In the first
    round (b = 1, S = 1, no current overlap noise) an extension whose
    sampled overlap is s, from k of N = 256 ancilla outcomes -1, has
    etahat = 1 - s**2, raw V = (1 - s**2)/255, the proxy Vt = max(V, 1/257**2),
    sigma = 2 |s| sqrt(Vt), d = 3 sqrt(Vt) and R = 3 sigma + d**2. For b the margin
    etahat - R stays negative unless k >= 9, while k has the binomial law of
    mean 256 sin(pi/64)**2 = 0.62 (probability about 2e-8). For a, s has mean
    0 and |s| < 1/2 except with probability about 1e-13, which leaves a
    margin above 0.5. So a is added and b is unresolved, although its exact
    eta passes the exact floor. In the second round the extensions of a are
    (a, a), which is -|00>, and (a, b), whose exact part outside the span of
    |00> and -|01> again has eta = sin(pi/32)**2. Every outcome of the
    overlap of (a, a) with |00> is -1, and with the sampled S_01 = z and
    overlap s_1 of (a, a) with -|01>, etahat = -(z + s_1)**2/(1 - z**2) <= 0
    for every count table. Neither extension resolves its independence, so
    the run stops with noise_inconclusive on the two-state basis, whose
    solve gives the reported energy, after exactly the 2 + 2 * 4 + 2 * 6
    settings of the reference diagonal and the two rounds (G(b + 1) per
    extension of a b-state basis), with no further shot. The value basis
    keeps its raw moments. Each overlap setting that the eligibility code
    read is the counted chunk, with N = 256 and the raw V of its counts. The
    quantities of both rounds are recomputed from the counts by
    `_sampled_eligibility_oracle`, those of the first round on the solve of
    the reference alone, rebuilt from its measured moments, and those of the
    second on the final two-state solve with S = [[1, z], [z, 1]].

    The same Plan plans only the grouped measurement with shots. At 16
    shots per setting its first rows stage is checked against the Run's
    shot limit before its first query and continues after extend_limits.
    With PencilSampling it stops on the same basis, which the two stages
    then measure again.
    """
    from fractions import Fraction as F

    import nwqlib

    problem, method = _sampled_bounded()
    exact = nwqlib.solve(problem, method=method, seed=7, progress=False)
    assert [(c.chain[0][0], c.reason) for c in exact.history[0].candidates] == [(0, "ranked"), (1, "accepted")]
    # 1e-15 is an empirical allowance with no qualified status.
    assert exact.history[0].candidates[1].eta == pytest.approx(np.sin(pi / 32) ** 2, rel=0, abs=1e-15)

    plan = nwqlib.plan(problem, method=method, shots=256, seed=7)
    assert (plan.reconstruction.acquisition.matrix_measurement, plan.reconstruction.acquisition.quadratures,
            len(plan.reconstruction.energy_groups)) == ("grouped", "real", 2)
    result = nwqlib.solve(plan, progress=False)
    a, b = (0, pi / 2), (1, pi / 2)
    first, second = (row.candidates for row in result.history)
    assert [(c.chain, c.reason) for c in first] == [((a,), "accepted"), ((b,), "screening_independence_unresolved")]
    assert [(c.chain, c.reason) for c in second] == [((a, a), "screening_independence_unresolved"),
                                                     ((a, b), "screening_independence_unresolved")]
    assert result.stop_reason == "noise_inconclusive" and result.basis == result.value_basis == ((), (a,))
    assert len(result.history) == 2 and result.eigenvalue == result.history[-1].energy
    chunks = result.data.observations.chunks
    assert len(chunks) == 2 + 2 * 4 + 2 * 6 and {c.returned_shots for c in chunks} == {256}
    context = adapt_acquisition.AdaptContext(observations=adapt_acquisition._observation_index(chunks))
    reference = adapt_acquisition._pencil_from_observations(plan, ((),), context)
    assert reference.eigenvalues[0] == result.history[0].energy
    floor = F(float(F(1, 257**2)))

    def overlap(basis, i, j):
        """Return the sampled real overlap of basis states i < j as a fraction, its raw V and its variance proxy.

        The setting that the eligibility code reads for this overlap must be
        the counted chunk, with N = 256 and the nearest binary64 number to V.
        """
        point = adapt_acquisition._point(plan, "pair", basis[i], basis[j], quadrature=0, group=0)
        (chunk,) = [c for c in chunks if c.realization_id == point.content_id]
        histogram = chunk.histogram()
        odd = int(histogram.weights[(histogram.packed_indices()[:, 0] & np.uint64(1)).astype(bool)].sum())
        s = F(256 - 2 * odd, 256)
        v = (1 - s * s) / 255
        settings = adapt_acquisition._overlap_settings(adapt_acquisition._grouped_statistics(plan, basis, context)[0])
        assert settings[i, j] == ((chunk.content_id, 256, float(v)),)
        return s, v, max(F(float(v)), floor)

    extensions = [((s,), 0, (proxy,)) for s, _, proxy in (overlap(((), c.chain), 0, 1) for c in first)]
    for candidate, expected in zip(first, _sampled_eligibility_oracle(reference, None, extensions), strict=True):
        for key, value in expected.items():
            assert getattr(candidate, key) == pytest.approx(value, rel=0, abs=2**-13), (candidate.chain, key)
        assert abs(expected["eta_margin"] - 1e-12) > .02
        assert (candidate.multiplier, candidate.independence_cutoff, candidate.overlap_cutoff, candidate.kept_rank) == (
            3, 1e-12, 1e-12, 1)
    assert first[0].score is not None and first[1].score is None
    z, _, vz = overlap(((), (a,)), 0, 1)
    extensions = []
    for candidate, dependent in zip(second, (True, False), strict=True):
        extended = ((), (a,), candidate.chain)
        (s0, v0, p0), (s1, _, p1) = overlap(extended, 0, 2), overlap(extended, 1, 2)
        if dependent:
            assert (s0, v0) == (-1, 0)
        extensions.append(((s0, s1), vz, (p0, p1)))
    for candidate, expected in zip(second, _sampled_eligibility_oracle(result.pencil, z, extensions), strict=True):
        for key, value in expected.items():
            assert getattr(candidate, key) == pytest.approx(value, rel=0, abs=2**-13), (candidate.chain, key)
        assert candidate.kept_rank == 2 and candidate.eta_margin < 0 and candidate.metric_epsilon < .5
    pair = next(m for m in result.group_moments if m.pair == (0, 1) and m.group == 0)
    assert pair.overlap_variance == float(overlap(((), (a,)), 0, 1)[1])

    from nwqlib.algorithms import PencilAcquisition, PencilSampling
    from nwqlib.execution import ExecutionLimits

    # The rule reads overlap settings of the grouped measurement, so planning refuses termwise
    # measurement with shots, while exact readout keeps that request with no measurement to resolve.
    termwise = method.revise(acquisition=PencilAcquisition(matrix_measurement="termwise", pair_preparation="controlled"))
    with pytest.raises(ValueError, match="grouped measurement"):
        nwqlib.plan(problem, method=termwise, shots=256, seed=7)
    assert nwqlib.plan(problem, method=termwise, seed=7).reconstruction.acquisition.matrix_measurement is None

    # With 16 shots per setting the reference diagonal's 2 settings count 2 * 16 shots, and the first
    # round's two extensions of the one-state basis request G(b + 1) = 4 settings each, 8 * 16 shots.
    # One shot less than the sum refuses the round before its first query, and after extend_limits the
    # resumed Run grows the basis of the unlimited one.
    sixteen = nwqlib.plan(problem, method=method, shots=16, seed=7)
    unlimited = nwqlib.solve(sixteen, progress=False)
    prepared = nwqlib.prepare(sixteen, limits=ExecutionLimits(max_total_shots=2 * 16 + 8 * 16 - 1), progress=False)
    with prepared.run as run:
        with pytest.raises(ValueError, match=r"max_total_shots.*\(32 counted, 128 requested, at least 160 needed\)"):
            nwqlib.submit(prepared)
        assert len(run.trace.events) == 2 and run.checkpoint_state["phase"] == "rows"
        run.extend_limits(max_total_shots=10**6)
        run.resume()
        resumed = run.wait(timeout=60, poll_interval=0)
    assert (resumed.basis, resumed.stop_reason, [(r.energy, r.candidates) for r in resumed.history]) == (
        unlimited.basis, unlimited.stop_reason, [(r.energy, r.candidates) for r in unlimited.history])

    # With PencilSampling(B = 200, f = 0.3) the same selection stops on the same two-state basis, and the
    # pilot and main stages then measure its m = b(b+1)G/2 = 6 settings afresh: the pilot takes
    # min(B - 2m, max(2m, floor(0.3 B))) = 59 shots, 10 for each of the first five settings and 9 for the
    # last, and only the main chunks form the result.
    staged = nwqlib.solve(nwqlib.plan(problem, method=method.revise(
        sampling=PencilSampling(total_shots=200, pilot_fraction=.3)), shots=256, seed=7), progress=False)
    assert (staged.stop_reason, staged.value_basis) == ("noise_inconclusive", result.value_basis)
    assert [row.candidates for row in staged.history] == [row.candidates for row in result.history]
    n, staged_chunks = len(chunks), staged.data.observations.chunks
    stages = [{item.parameter: item.value for item in c.bindings}["stage"] for c in staged_chunks]
    assert stages == [-1] * n + [0] * 6 + [1] * 6
    assert [c.returned_shots for c in staged_chunks[n:n + 6]] == [10] * 5 + [9]
    assert [c.returned_shots for c in staged_chunks[n + 6:]] == list(staged.shot_allocation.main_counts)
    assert sum(e.shots for e in staged.data.trace.events) == 256 * n + 200
    assert set(staged.contribution_ids) == {c.content_id for c in staged_chunks[n + 6:]}


def test_current_overlap_noise_alone_changes_sampled_eligibility():
    """The same extension data with three noise levels of the current overlap S_01 are eligible, unresolved or unresolved by the metric.

    The current basis has S = I and the extension the overlaps s = (3/5, 3/5),
    each from N = 6400 shots with V = (1 - 9/25)/6399, so a = s, etahat = 7/25,
    x = sqrt(18)/5 and b_s = 2 V, d = 3 sqrt(b_s) = 0.0424. Only the
    real-only S_01, of sample mean 0, changes: V_01 = 1/9999 (N = 10000),
    1/49 (N = 50) and 1/25 (N = 26), so epsilon = 3 sqrt(V_01) is 0.030, 3/7
    and 3/5. With w = a_0 a_1 = 9/25, sigma**2 = (18/25)**2 V_01 + 2 (6/5)**2 V,
    so R = 3 sigma + (d + epsilon x)**2/(1 - epsilon) is about 0.060 and 0.60
    for the first two levels: the margin etahat - R is +0.22 and -0.32, and
    the third level has epsilon >= 1/2. etahat is the same in all three.
    """
    from fractions import Fraction as F

    from nwqlib.algorithms.gcim.bounded_growth import current_projection, sampled_eligibility, sampled_metric
    from nwqlib.algorithms.gcim.fixed_basis import solve_pencil

    h0 = np.diag([0.0, 1.0]).astype(complex)
    pencil = solve_pencil(h0, np.eye(2, dtype=complex), cutoff=1e-12, sampled=True)
    current = current_projection(h0, [[complex(z.real, z.imag) for z in row] for row in pencil.coordinate_vectors],
                                 pencil.offset_free_eigenvalues[0])
    v = (1 - 9 / 25) / 6399
    column = tuple(((f"s{i}", 6400, v),) for i in range(2))
    levels = (10000, 50, 26)
    records = []
    for shots in levels:
        metric = sampled_metric(current, {(0, 1): (("S01", shots, 1 / (shots - 1)),)}, pencil.overlap_eigenvalues[0])
        records.append(sampled_eligibility(current, metric, [.6, .6], column, 1e-12))
    assert [r["reason"] for r in records] == [None, "screening_independence_unresolved", "screening_metric_unresolved"]
    assert len({r["eta"] for r in records}) == 1
    # The oracle reads the binary64 inputs .6, v and 1/(shots - 1) as exact
    # fractions, with each one-count floor formed from its own N.
    proxy = max(F(v), F(float(F(1, 6401**2))))
    extensions = [((F(.6), F(.6)), max(F(1 / (shots - 1)), F(float(F(1, (shots + 1) ** 2)))), (proxy, proxy))
                  for shots in levels]
    for record, expected in zip(records, _sampled_eligibility_oracle(pencil, F(0), extensions), strict=True):
        # At epsilon >= 1/2 the rule stops before it computes sigma, b_s, R and the margin.
        resolved = record["metric_epsilon"] < .5
        for key, value in expected.items():
            if resolved or key in ("eta", "metric_beta", "metric_epsilon"):
                assert record[key] == pytest.approx(value, rel=0, abs=2**-13), key
            else:
                assert record[key] is None, key
    assert records[0]["eta_margin"] > .2 and records[1]["eta_margin"] < -.3


def test_eta_gradients_match_the_derivative_of_a_complex_perturbed_projection():
    """The first-order change of eta from eta_gradients equals the exact derivative for a complex current overlap.

    The sampled eligibility rule contracts these gradients with the errors of
    the measured overlap parts, so a wrong sign of an imaginary part would
    misjudge every complex extension. With S = I (T = I), t = (1/2, i/4),
    an error of S with the complex off-diagonal E_01 = 1/3 + i/5 and zero
    diagonal (S_ii is exact) and an error f = (1/7 - i/3, 2/5 + i/9) of
    the extension's overlaps s, SymPy differentiates
    eta(h) = 1 - (t + h f)^dagger (I + h E)^(-1) (t + h f) at h = 0. The
    gradients of `eta_gradients` at a = t, contracted with the real and
    imaginary parts of E_01 and f, must give that derivative. The allowances
    1e-25 for the 30-digit evaluation and 2**-45 for the comparison are
    empirical, not qualified rounding bounds. A flipped sign of an imaginary
    gradient changes the contraction by far more than either.
    """
    import sympy as sp

    from nwqlib.algorithms.gcim.bounded_growth import eta_gradients

    h = sp.Symbol("h", real=True)
    t = sp.Matrix([sp.Rational(1, 2), sp.I / 4])
    e01 = sp.Rational(1, 3) + sp.I / 5
    error = sp.Matrix([[0, e01], [sp.conjugate(e01), 0]])
    f = sp.Matrix([sp.Rational(1, 7) - sp.I / 3, sp.Rational(2, 5) + sp.I / 9])
    eta = 1 - ((t + h * f).H * (sp.eye(2) + h * error).inv() * (t + h * f))[0]
    exact = complex(sp.N(sp.diff(eta, h).subs(h, 0), 30))
    current, column = eta_gradients(np.array([complex(x) for x in t]))
    linear = (current[0, 1][0] * float(sp.re(e01)) + current[0, 1][1] * float(sp.im(e01))
              + sum(u[0] * float(sp.re(x)) + u[1] * float(sp.im(x)) for u, x in zip(column, f)))
    assert abs(exact.imag) < 1e-25
    assert linear == pytest.approx(exact.real, rel=0, abs=2**-45)
