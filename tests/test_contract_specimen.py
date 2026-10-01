"""Independent algebra, logical accounting and evidence-boundary witnesses."""

from collections import Counter
from copy import deepcopy
from fractions import Fraction as F
import importlib.util
import json
from pathlib import Path

import pytest
from jsonschema import ValidationError

SCRIPT = Path(__file__).resolve().parents[1] / "docs/scripts/contract_specimen.py"
spec = importlib.util.spec_from_file_location("contract_specimen", SCRIPT)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def test_exact_sector_moments_and_pencil():
    # q1,q2,q3 have Z=-1 and never flip. On q0, X+Z-3I acts
    # as (-2*a+b, a-4*b). This independent pair recurrence does not
    # call the generator's Chebyshev/pencil assembly or any NWQLib kernel.
    state = (1, 0)
    raw = []
    for _ in range(4):
        raw.append(state[0])
        state = (-2 * state[0] + state[1], state[0] - 4 * state[1])
    assert raw == [1, -2, 5, -16]
    mu = [F(1), F(raw[1], 5), 2 * F(raw[2], 25) - 1,
          4 * F(raw[3], 125) - 3 * F(raw[1], 5)]
    assert mu == [F(1), F(-2, 5), F(-3, 5), F(86, 125)]
    bundle = fixture.generate()
    p = fixture.payloads(bundle)
    assert p["request"]["hamiltonian"] == {
        "labels": ["IIIX", "IIIZ", "IIZI", "IZII", "ZIII"],
        "coefficients": [1, 1, 1, 1, 1], "units": "input_energy",
    }
    ref = p["request"]["reference"]
    assert ref["x_system_qubits"] == [1, 2, 3]
    assert ref["system_basis_index"] == sum(1 << q for q in ref["x_system_qubits"]) == 14
    assert ref["display_bits"] == format(ref["system_basis_index"], "04b") == "1110"
    assert p["request"]["qubit_order"] == "little_endian"
    assert p["request"]["label_order"] == "Qiskit_display"
    assert p["plan"]["registers"] == {"index": [0, 1, 2], "system": [3, 4, 5, 6]}
    actual = [F(p["observations"]["known_moments"]["0"])] + [
        F(e["expectation"]) for e in p["observations"]["contributions"]]
    assert actual == mu
    # Direct basis vectors v0=(1,0), v1=(-2/5,1/5) give Gram and H
    # matrix elements independently of moment product identities.
    basis = [(F(1), F(0)), (F(-2, 5), F(1, 5))]
    gram = [[sum(a * b for a, b in zip(v, w)) for w in basis] for v in basis]
    hproj = [[v[0] * (-2 * w[0] + w[1]) + v[1] * (w[0] - 4 * w[1])
              for w in basis] for v in basis]
    assert p["result"]["pencil"] == {
        "overlap": [[str(x) for x in row] for row in gram],
        "hamiltonian": [[str(x) for x in row] for row in hproj],
    }
    # det(Hp-E*S) = c0+c1*E+c2*E² = (E²+6E+7)/25;
    # hence the lower root is exactly -3-sqrt(2), not a fitted float.
    h, s = hproj, gram
    coefficients = [h[0][0] * h[1][1] - h[0][1] ** 2,
                    -h[0][0] * s[1][1] - s[0][0] * h[1][1] + 2 * h[0][1] * s[0][1],
                    s[0][0] * s[1][1] - s[0][1] ** 2]
    assert coefficients == [F(7, 25), F(6, 25), F(1, 25)]
    assert p["result"]["energy"] == {"rational": -3, "sqrt_coefficient": -1, "radicand": 2}
    # Independent full-space identification uses commuting factors, no dense
    # matrix: X+Z squares to 2I and each spectator contributes +/-1.
    spectator_sums = sorted({a + b + c for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)})
    assert p["provenance"]["ground_evidence"]["spectator_sums"] == spectator_sums
    assert min(spectator_sums) == -3
    assert fixture.check(bundle)["real_executions"] == 0


def _dynamic_lowered(recipe):
    # Materialize only bounded test events, independently of the fold's
    # multiplication rule. A Repeat with n=4 really traverses its body 4 times.
    def events(node):
        match node["type"]:
            case "BlockCall":
                if node["block"] in recipe["definitions"]:
                    yield from events(recipe["definitions"][node["block"]])
                else:
                    yield node["block"]
            case "Sequence":
                for child in node["children"]:
                    yield from events(child)
            case "Repeat":
                for _ in range(node["count"]):
                    yield from events(node["body"])
            case "MeasurementBatch":
                for setting in node["settings"]:
                    yield from events(setting["body"])
    return Counter(events(recipe["root"]))


def test_selected_realization_changes_dynamic_work():
    direct = fixture.payloads(fixture.generate())
    alternative_bundle = fixture.generate(choice="hzh")
    alternative = fixture.payloads(alternative_bundle)
    fixture.check(alternative_bundle)
    # Independent unitary identity: H Z H maps (a,b) to (b,a), including
    # phase. Integer Hadamard numerators accumulate denominator 2.
    for a, b in ((1, 0), (0, 1)):
        h1 = (a + b, a - b)
        z = (h1[0], -h1[1])
        assert (F(z[0] + z[1], 2), F(z[0] - z[1], 2)) == (b, a)
    assert direct["result"]["pencil"] == alternative["result"]["pencil"]
    assert direct["result"]["energy"] == alternative["result"]["energy"]
    baseline = {"X": 9, "H": 0, "Z": 0, "PREP": 5, "PREP_inverse": 3,
                "SELECT": 2, "zero_reflection": 2, "SELECT_basis": 2, "measure": 17}
    for p, expected in ((direct, baseline), (alternative, {**baseline, "X": 0, "H": 18, "Z": 9})):
        gates = ["X"] if p["realization"]["choice"] == "direct_x" else ["H", "Z", "H"]
        reference_action = [(gate, [q]) for q in (4, 5, 6) for gate in gates]
        for setting in p["prepared"]["program"]["root"]["settings"]:
            emitted_reference = setting["body"]["children"][0]["children"]
            assert [(node["block"], node["qubits"]) for node in emitted_reference] == reference_action
        assert p["workload"]["inventory"] == expected
        emitted = _dynamic_lowered(p["prepared"]["program"])
        assert {key: emitted[key] for key in expected} == expected
        # Increase only last prefix from 1 to 4: +3 of each walk operation;
        # the recipe still keeps just one walk body in definitions.
        plan = deepcopy(p["plan"])
        plan["program"]["root"]["settings"][-1]["body"]["children"][2]["count"] = 4
        lower = fixture.lower(plan, p["realization"])
        changed = fixture.fold(plan, p["realization"])
        expected_repeat = {**expected, **{key: expected[key] + 3 for key in (
            "SELECT", "PREP", "PREP_inverse", "zero_reflection")}}
        assert changed == expected_repeat
        emitted = _dynamic_lowered(lower)
        assert {key: emitted[key] for key in expected} == expected_repeat
        assert len(lower["definitions"]) == 1
    # A rehashed artifact from another choice cannot be reused under the
    # selected realization even if widths and moment claims agree.
    bad = fixture.generate(choice="hzh")
    fixture.payloads(bad)["prepared"]["program"] = direct["prepared"]["program"]
    fixture.seal(bad)
    with pytest.raises(ValueError, match="lowering"):
        fixture.check(bad)


def test_partial_and_ground_evidence_boundaries():
    for missing in ((1,), (2,), (3,), (1, 2, 3)):
        bundle = fixture.generate(missing=missing)
        p = fixture.payloads(bundle)
        assert fixture.check(bundle)["coverage"] == "partial"
        assert p["result"]["energy"] is None
        p["result"]["status"] = "final"
        fixture.seal(bundle)
        with pytest.raises(ValueError, match="incomplete observations"):
            fixture.check(bundle)
    bundle = fixture.generate(ground_evidence=False)
    p = fixture.payloads(bundle)
    fixture.check(bundle)
    assert p["result"]["status"] == "final"
    assert not p["result"]["ground_accuracy_certificate"]
    assert p["claims"]["target_status"] == "established_for_analytical_fixture_only"
    assert [a["status"] for a in p["claims"]["assessments"]] == ["PASS", "PASS", "INCONCLUSIVE", "NOT_RUN"]
    p["result"]["ground_accuracy_certificate"] = True
    fixture.seal(bundle)
    with pytest.raises(ValueError, match="ground accuracy requires"):
        fixture.check(bundle)
    # Ground evidence is an identity dependency of the Result itself.
    with_ground = {r["id"]: r["sha256"] for r in fixture.generate()["records"]}
    without_ground = {r["id"]: r["sha256"] for r in fixture.generate(ground_evidence=False)["records"]}
    assert with_ground["result"] != without_ground["result"]
    evidence_change = fixture.generate()
    old_result_payload = deepcopy(fixture.payloads(evidence_change)["result"])
    fixture.payloads(evidence_change)["provenance"]["ground_evidence"]["scope"] += "; changed proof annotation"
    fixture.seal(evidence_change)
    assert fixture.payloads(evidence_change)["result"] == old_result_payload
    changed_result = next(r for r in evidence_change["records"] if r["id"] == "result")
    assert changed_result["sha256"] != with_ground["result"]


def test_digest_and_lifecycle_falsifiers():
    bundle = fixture.generate()
    fixture.payloads(bundle)["prepared"]["program"]["root"]["settings"].pop()
    with pytest.raises(ValueError, match="changed payload"):
        fixture.check(bundle)
    bundle = fixture.generate()
    prepared = next(r for r in bundle["records"] if r["id"] == "prepared")
    prepared["refs"][0]["sha256"] = "0" * 64
    prepared["sha256"] = fixture.digest({key: prepared[key] for key in ("id", "kind", "payload", "refs")})
    with pytest.raises(ValueError, match="stale reference"):
        fixture.check(bundle)
    changes = [
        ("trace", "executed_settings", 1), ("trace", "logical_prepared_settings", 2),
        ("prepared", "submission_ready", True), ("workload", "compiler_workspace_bytes", 0),
        ("result", "statistical_interval", [0, 0]),
    ]
    for record, field, value in changes:
        bundle = fixture.generate()
        fixture.payloads(bundle)[record][field] = value
        fixture.seal(bundle)
        with pytest.raises(ValueError):
            fixture.check(bundle)
    bundle = fixture.generate()
    fixture.payloads(bundle)["observations"]["contributions"][0]["trace_event"] = "fixture_mu_2"
    fixture.seal(bundle)
    with pytest.raises(ValueError, match="linkage"):
        fixture.check(bundle)
    bundle = fixture.generate()
    fixture.payloads(bundle)["observations"]["contributions"][0]["expectation"] = 0.0
    fixture.seal(bundle)
    with pytest.raises(ValidationError):
        fixture.check(bundle)


def test_committed_fixture_checks_without_regeneration(tmp_path):
    bundle = json.loads((fixture.ROOT / "specimen.json").read_text())
    assert fixture.check(bundle) == {"records": 14, "coverage": "complete",
        "real_executions": 0, "scope": "analytical_contract_fixture_only"}
    registry = json.loads((fixture.ROOT / "claims.json").read_text())
    assert fixture.check_claims() == len(registry["claims"])
    # A deleted or renamed witness must fail locally as it does in the CI CLI.
    registry["claims"][0]["witnesses"] = [
        "tests/test_contract_specimen.py::test_missing_witness"]
    broken = tmp_path / "claims.json"
    broken.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="missing witness"):
        fixture.check_claims(broken)
