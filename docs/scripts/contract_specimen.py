"""Generate and check the bounded analytical Lanczos development contract.

This is a documentation fixture owner, not an execution engine. No backend,
compiler, full-state operator, or reference eigensolver is invoked here.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1] / "contract-specimen"
VERSION = "nwqlib.contract-specimen/1"
LABELS = ["IIIX", "IIIZ", "IIZI", "IZII", "ZIII"]
REGISTERS = {"index": [0, 1, 2], "system": [3, 4, 5, 6]}
BASIS = ["X", "H", "Z", "PREP", "PREP_inverse", "SELECT", "zero_reflection", "SELECT_basis", "measure"]
SCOPE = "one invocation per required moment setting; no shots multiplier"


def digest(value):
    """Hash canonical JSON, rejecting non-finite values and preserving list order."""
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def seal(bundle):
    """Explicitly regenerate identities after editing a topologically ordered fixture."""
    known = {}
    for record in bundle["records"]:
        record["refs"] = [{"id": ref["id"], "sha256": known[ref["id"]]}
                          for ref in record["refs"]]
        record["sha256"] = digest({key: record[key] for key in ("id", "kind", "payload", "refs")})
        known[record["id"]] = record["sha256"]
    return bundle


def payloads(bundle):
    return {record["id"]: record["payload"] for record in bundle["records"]}


def call(block, qubits=None):
    node = {"type": "BlockCall", "block": block}
    if qubits is not None:
        node["qubits"] = qubits
    return node


def sequence(children):
    return {"type": "Sequence", "children": children}


def program():
    """Keep one walk definition and a Repeat node for each setting."""
    return {
        "definitions": {"walk": sequence([call("SELECT"), call("PREP_inverse"),
                                            call("zero_reflection"), call("PREP")])},
        "root": {"type": "MeasurementBatch", "settings": [
            {"id": f"mu_{k}", "degree": k, "body": sequence([
                call("reference"), call("PREP"),
                {"type": "Repeat", "count": k // 2, "body": call("walk")},
                call("SELECT_basis" if k % 2 else "PREP_inverse"),
                *[call("measure", [q]) for q in range(7 if k % 2 else 3)],
            ])} for k in (1, 2, 3)]},
    }


def reference(choice):
    if choice not in ("direct_x", "hzh"):
        raise ValueError("unsupported reference realization")
    gates = ["X"] if choice == "direct_x" else ["H", "Z", "H"]
    return sequence([call(gate, [q]) for q in REGISTERS["system"][1:] for gate in gates])


def fold(plan, realization):
    """Count dynamic logical work from the selected decompositions, without expansion."""
    definitions = {**plan["program"]["definitions"], "reference": realization["reference"]}

    def visit(node):
        kind = node["type"]
        if kind == "BlockCall":
            block = node["block"]
            return visit(definitions[block]) if block in definitions else Counter({block: 1})
        if kind == "Repeat":
            return Counter({key: value * node["count"] for key, value in visit(node["body"]).items()})
        children = ([s["body"] for s in node["settings"]] if kind == "MeasurementBatch"
                    else node["children"])
        total = Counter()
        for child in children:
            total.update(visit(child))
        return total

    counts = visit(plan["program"]["root"])
    return {gate: counts[gate] for gate in BASIS}


def lower(plan, realization):
    """Resolve reference calls, keeping reusable walk definitions and Repeat bodies."""
    def visit(node):
        if node["type"] == "BlockCall" and node["block"] == "reference":
            return deepcopy(realization["reference"])
        result = deepcopy(node)
        if node["type"] == "Sequence":
            result["children"] = [visit(child) for child in node["children"]]
        elif node["type"] == "Repeat":
            result["body"] = visit(node["body"])
        elif node["type"] == "MeasurementBatch":
            result["settings"] = [{**s, "body": visit(s["body"])} for s in node["settings"]]
        return result

    return {"definitions": deepcopy(plan["program"]["definitions"]),
            "root": visit(plan["program"]["root"])}


def projected_pencil(moments):
    """Reconstruct the m=2 Chebyshev pencil from exact sufficient statistics."""
    mu = {int(k): Fraction(v) for k, v in moments.items()}
    if not {0, 1, 2, 3} <= mu.keys():
        raise ValueError("incomplete moments for projected pencil")
    overlap = [[mu[0], mu[1]], [mu[1], (mu[0] + mu[2]) / 2]]
    hamiltonian = [[5 * mu[1], 5 * (mu[0] + mu[2]) / 2],
                   [5 * (mu[0] + mu[2]) / 2, 5 * (mu[3] + 3 * mu[1]) / 4]]
    return {"overlap": [[str(x) for x in row] for row in overlap],
            "hamiltonian": [[str(x) for x in row] for row in hamiltonian]}


def generate(*, choice="direct_x", missing=(), ground_evidence=True):
    """Construct only this named fixture; optional omissions provide legal negative cases."""
    records = []

    def add(name, kind, payload, refs=()):
        records.append({"id": name, "kind": kind, "payload": payload,
                        "refs": [{"id": ref} for ref in refs]})

    add("request", "Request", {
        "hamiltonian": {"labels": LABELS, "coefficients": [1] * 5, "units": "input_energy"},
        "reference": {"system_basis_index": 14, "display_bits": "1110", "x_system_qubits": [1, 2, 3]},
        "system_qubits": 4, "qubit_order": "little_endian", "label_order": "Qiskit_display",
        "output": "lowest_projected_energy", "accuracy_target": {"absolute": "1/1000", "unit": "input_energy"},
    })
    add("plan", "Plan", {
        "algorithm": "chebyshev_lanczos", "krylov_dimension": 2, "alpha": 5,
        "required_moments": [1, 2, 3], "known_moments": {"0": "1"},
        "registers": REGISTERS, "program": program(),
        "block_semantics": {
            "PREP": "|0^3> -> (|0>+|1>+|2>+|3>+|4>)/sqrt(5)",
            "SELECT": "sum_j |j><j| tensor P_j; j=0..4 in request label order; identity for j=5..7",
            "zero_reflection": "2|0^3><0^3|-I", "walk": "R U; U=SELECT; R=PREP zero_reflection PREP_inverse",
            "SELECT_basis": "index-label-controlled H for X; identity for I/Z",
            "reference": "|0000> -> |1110>; exact norm one; inverse/control legal; no dirty ancillas",
        },
        "readout": {
            "state": "(R U)^floor(k/2) |G,psi>",
            "odd": "mu_k=<SELECT>; after SELECT_basis decode signed system parity keyed by index",
            "even": "mu_k=<R>; after PREP_inverse decode +1 for index=0 else -1",
            "term_table": [{"index": j, "label": label, "sign": 1, "parity_mask": mask}
                           for j, (label, mask) in enumerate(zip(LABELS, [1, 1, 2, 4, 8]))],
            "padding_outcome": 0, "postselection": False,
        },
        "phase_budgets": {"backend_calls": 0, "shots": 0, "target_compilations": 0,
                          "analysis": "fixed 2x2 rational pencil; no reference solve"},
    }, ["request"])
    plan = records[-1]["payload"]
    add("realization", "Realization", {"choice": choice, "reference": reference(choice),
        "bindings": "all logical register/degree bindings resolved", "evidence_kind": "analytical_fixture"}, ["plan"])
    realization = records[-1]["payload"]
    logical = lower(plan, realization)
    add("prepared", "PreparedArtifact", {
        "format": "nwqlib.logical-recipe-fixture/1", "program": logical,
        "program_sha256": digest(logical), "lifecycle": "logical_fixture_prepared",
        "target_compilation": "not_run", "submission_ready": False,
        "compiler": None, "target": "analytical-fixture", "layout": REGISTERS,
        "remaining_requirements": ["target selection", "capability and capacity assessment", "compilation", "execution approval"],
    }, ["realization"])
    add("workload", "WorkloadEstimate", {
        "stage": "logical", "lifecycle": "planned", "population": SCOPE,
        "basis": BASIS, "inventory": fold(plan, realization), "unit": "dynamic_logical_invocations",
        "evidence_kind": "exact_structural_relation", "system_width": 4, "index_width": 3,
        "logical_width": 7, "required_settings": 3, "stored_walk_definitions": 1,
        "native_gates": None, "T_count": None, "compiled_depth": None,
        "compiler_workspace_bytes": None, "runtime_seconds": None, "currency_cost": None,
        "unknown_reason": "no native lowering, synthesis precision, compiler, device or calibrated cost model",
    }, ["realization", "prepared"])
    add("profile", "DeviceProfile", {"name": "analytical-fixture", "version": "1",
        "capabilities": ["exact_expectation_fixture"], "device_qubit_capacity": None,
        "workspace_model": None, "runtime_model": None, "calibration": None,
        "freshness": "not_applicable_no_device_snapshot"})
    add("allocation", "Allocation", {"device_qubits": None, "memory_bytes": None,
        "concurrency": None, "backend_jobs_allowed": 0}, ["profile"])
    add("assessment", "Assessment", {"logical_width": 7, "capacity": "UNKNOWN",
        "runtime": "UNKNOWN", "cost": "UNKNOWN", "real_submission": "NOT_ASSESSED",
        "reason": "analytical fixture has no device allocation or calibrated model"},
        ["workload", "profile", "allocation"])
    available = [k for k in (1, 2, 3) if k not in missing]
    add("trace", "ExecutionTrace", {"status": "fixture_complete" if len(available) == 3 else "fixture_partial",
        "evidence_kind": "analytical_fixture", "prepared_id": "prepared", "target": "analytical-fixture",
        "planned_settings": 3, "logical_prepared_settings": 3, "target_prepared_settings": 0,
        "submitted_settings": 0, "executed_settings": 0, "returned_shots": 0,
        "backend_jobs": [], "measured_runtime_seconds": None,
        "fixture_events": [{"id": f"fixture_mu_{k}", "setting": f"mu_{k}", "attempt": "analytical-1",
                            "chunk": f"exact-{k}", "source": "analytical-moment-identity/1"} for k in available],
    }, ["prepared", "assessment"])
    exact = {1: "-2/5", 2: "-3/5", 3: "86/125"}
    add("observations", "ObservationView", {
        "coverage": "complete" if len(available) == 3 else "partial", "kind": "exact_expectation_fixture",
        "known_moments": {"0": "1"},
        "contributions": [{"setting": f"mu_{k}", "degree": k, "expectation": exact[k],
            "trace_event": f"fixture_mu_{k}", "attempt": "analytical-1", "chunk": f"exact-{k}",
            "prepared_id": "prepared", "parameter_point": "fixed-H-and-reference",
            "measured_qubits": list(range(7 if k % 2 else 3)), "returned_shots": None,
            "conditioning": "none", "uncertainty": "not_applicable_exact_fixture"} for k in available],
    }, ["plan", "prepared", "trace"])
    add("provenance", "Provenance", {
        "origin": "synthetic_analytical_contract_fixture", "generator": "docs/scripts/contract_specimen.py@1",
        "source_revision": "6d5172da8943a39b0a997e53ba9da089e61d5d88", "source_revision_role": "implementation_starting_point",
        "software_environment": "not_a_backend_run; generator environment not claimed as execution provenance",
        "moment_evidence": {"method": "invariant_sector_exact_algebra", "basis_indices": [14, 15],
            "restricted_H": [[-2, 1], [1, -4]], "raw_power_moments": [1, -2, 5, -16]},
        "ground_evidence": {"method": "commuting_tensor_factors", "spectator_sums": [-3, -1, 1, 3],
            "local_eigenvalues": "plus_or_minus_sqrt(2)", "minimum": "-3-sqrt(2)",
            "scope": "full_16_dimensional_H_analytical_fixture"} if ground_evidence else None,
        "data_storage": "exact sufficient moments and compact recipes; no raw shots or full states exist",
    }, ["request"])
    complete = len(available) == 3
    add("result", "Result", {"type": "ProjectedEnergyResult", "status": "final" if complete else "unavailable",
        "coverage": "complete" if complete else "partial", "unit": "input_energy",
        "pencil": projected_pencil({"0": "1", **{str(k): exact[k] for k in available}}) if complete else None,
        "energy": {"rational": -3, "sqrt_coefficient": -1, "radicand": 2} if complete else None,
        "interpretation": "lowest eigenvalue in span{psi,(H/5)psi}",
        "statistical_interval": None, "ground_accuracy_certificate": complete and ground_evidence,
        "evidence_kind": "analytical_fixture", "missing_moments": sorted(set((1, 2, 3)) - set(available)),
    }, ["request", "plan", "observations", "trace", "provenance"])
    add("errors", "ErrorModel", {"target": "lowest_projected_energy", "unit": "input_energy",
        "terms": [
            {"name": "sampling", "evidence_kind": "not_applicable", "composition_role": "listed_only",
             "coverage": "exact_fixture_only", "bound": None},
            {"name": "implementation", "evidence_kind": "unknown", "composition_role": "listed_only",
             "coverage": "no_compiled_or_executed_artifact", "bound": None},
            {"name": "subspace", "evidence_kind": "proved_relation" if ground_evidence else "unknown",
             "composition_role": "conditional_term", "coverage": "analytical_full_spectrum" if ground_evidence else "unresolved",
             "target": "additional_ground_identification_claim",
             "bound": "0" if ground_evidence else None}],
        "requested_accuracy": "1/1000", "backend_total_error_bound": None,
    }, ["request", "result"])
    add("claims", "ClaimAssessments", {"assessments": [
        {"claim": "projected_energy", "status": "PASS" if complete else "INCONCLUSIVE",
         "method": "exact_rational_pencil", "coverage": "analytical_fixture"},
        {"claim": "near_spectrum", "status": "PASS" if complete else "INCONCLUSIVE",
         "method": "invariant_sector_eigenpair", "coverage": "analytical_fixture"},
        {"claim": "ground_accuracy", "status": "PASS" if complete and ground_evidence else "INCONCLUSIVE",
         "method": "independent_full_spectrum_identification", "coverage": "analytical_fixture_only"},
        {"claim": "backend_accuracy", "status": "NOT_RUN", "method": "none", "coverage": "none"}],
        "target_status": "established_for_analytical_fixture_only" if complete else "target_only",
    }, ["result", "errors", "provenance", "prepared"])
    return seal({"schema_version": VERSION, "records": records})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(bundle, schema_path=ROOT / "records.schema.json"):
    """Validate schema, content references and the named fixture's semantic boundaries."""
    schema = json.loads(Path(schema_path).read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(bundle)
    records = bundle["records"]
    by_id = {record["id"]: record for record in records}
    require(len(by_id) == len(records), "duplicate record identity")
    for record in records:
        require(record["sha256"] == digest({key: record[key] for key in ("id", "kind", "payload", "refs")}),
                f"changed payload or references: {record['id']}")
        require(len({r["id"] for r in record["refs"]}) == len(record["refs"]), "duplicate reference")
        for ref in record["refs"]:
            require(ref["id"] in by_id and by_id[ref["id"]]["sha256"] == ref["sha256"], "stale reference")
    p = payloads(bundle)
    expected = generate(choice=p["realization"]["choice"],
                        missing=p["result"]["missing_moments"],
                        ground_evidence=p["provenance"]["ground_evidence"] is not None)
    expected_records = {r["id"]: r for r in expected["records"]}
    require(set(by_id) == set(expected_records), "missing or unsupported record")
    for name, record in by_id.items():
        require(record["kind"] == expected_records[name]["kind"], "record kind mismatch")
        require([r["id"] for r in record["refs"]] == [r["id"] for r in expected_records[name]["refs"]],
                f"required references differ: {name}")
    # This validator deliberately admits only the declared specimen domain and
    # its two preparation choices / missing evidence variants, not arbitrary IR.
    # Pinning semantic premises prevents rehashed payloads changing the problem
    # while keeping this example's analytical claims.
    for name in ("request", "plan", "profile", "allocation", "assessment", "provenance"):
        require(p[name] == payloads(expected)[name], f"unsupported fixture premises: {name}")
    require(p["realization"]["reference"] == reference(p["realization"]["choice"]), "reference realization mismatch")
    require(p["realization"] == payloads(expected)["realization"], "realization evidence mismatch")
    lowered = lower(p["plan"], p["realization"])
    require(p["prepared"]["program"] == lowered, "lowering does not match selected realization")
    require(p["prepared"]["program_sha256"] == digest(lowered), "prepared content digest mismatch")
    require(p["prepared"] == payloads(expected)["prepared"], "fixture cannot be target-prepared or submission-ready")
    require(p["workload"]["inventory"] == fold(p["plan"], p["realization"]), "fold does not match realization")
    require(p["workload"] == payloads(expected)["workload"], "logical population/basis or unknown resource mismatch")
    observations = p["observations"]
    entries = observations["contributions"]
    degrees = [entry["degree"] for entry in entries]
    require(len(set(degrees)) == len(degrees), "duplicate observation contribution")
    needed = set(p["plan"]["required_moments"])
    require(set(degrees) <= needed, "unknown observation setting")
    missing = sorted(needed - set(degrees))
    require(missing == p["result"]["missing_moments"], "missing observations mismatch")
    require(observations["coverage"] == ("partial" if missing else "complete"), "observation coverage mismatch")
    trace = p["trace"]
    require(trace == payloads(expected)["trace"], "execution population or fixture trace mismatch")
    events = {e["id"]: e for e in trace["fixture_events"]}
    for entry in entries:
        event = events.get(entry["trace_event"])
        require(event is not None and entry["setting"] == event["setting"]
                and entry["attempt"] == event["attempt"] and entry["chunk"] == event["chunk"],
                "observation execution linkage mismatch")
        require(entry["prepared_id"] == trace["prepared_id"] == "prepared", "prepared observation linkage mismatch")
    require(observations == payloads(expected)["observations"], "observation statistics/register/evidence mismatch")
    if missing:
        require(p["result"]["status"] != "final" and p["result"]["energy"] is None,
                "incomplete observations cannot yield a final answer")
    else:
        moments = {**observations["known_moments"], **{str(e["degree"]): e["expectation"] for e in entries}}
        require(p["result"]["pencil"] == projected_pencil(moments), "projected reconstruction mismatch")
    require(not p["result"]["ground_accuracy_certificate"] or
            (not missing and p["provenance"]["ground_evidence"] is not None),
            "ground accuracy requires independent ground identification")
    for name in ("result", "errors", "claims"):
        require(p[name] == payloads(expected)[name], f"unsupported scientific claim or error coverage: {name}")
    return {"records": len(records), "coverage": observations["coverage"],
            "real_executions": 0, "scope": "analytical_contract_fixture_only"}


def check_claims(path=ROOT / "claims.json"):
    """Check development-claim traceability; this does not execute its witnesses."""
    registry = json.loads(Path(path).read_text())
    require(registry["format"] == "nwqlib.development-claims/1", "unknown claims format")
    claims = registry["claims"]
    require(bool(claims), "empty claims register")
    require(len({claim["id"] for claim in claims}) == len(claims), "duplicate claim identity")
    test_functions = {}
    for claim in claims:
        for field in ("id", "owner", "claim", "independent_relation", "falsifier",
                      "evidence_kind", "cost", "unverified"):
            require(isinstance(claim[field], str) and bool(claim[field].strip()),
                    f"missing claim field: {field}")
        require(claim["status"] in ("specified", "implemented", "witnessed"), "unknown claim status")
        require(bool(claim["witnesses"]), "claim has no witness reference")
        for witness in claim["witnesses"]:
            source, function = witness.split("::")
            if source not in test_functions:
                tree = ast.parse((ROOT.parents[1] / source).read_text())
                test_functions[source] = {node.name for node in tree.body
                                          if isinstance(node, ast.FunctionDef)}
            require(function in test_functions[source], f"missing witness: {witness}")
    return len(claims)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--generate", action="store_true")
    parser.add_argument("--path", type=Path, default=ROOT / "specimen.json")
    args = parser.parse_args()
    if args.generate:
        bundle = generate()
        check(bundle)
        args.path.write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n")
        print(f"Generated analytical fixture: {args.path}")
    else:
        summary = check(json.loads(args.path.read_text()))
        summary["claim_references"] = check_claims()
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
