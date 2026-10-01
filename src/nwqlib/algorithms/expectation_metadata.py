"""Inert scientific capability and source description."""

from nwqlib.algorithms.protocol import AlgorithmDescriptor
from nwqlib.core.records import Source

_METHOD = Source(
    name="finite_pauli_expectation",
    version="2",
    domain="normalized expectation or physical quadratic form of admitted Hermitian input",
    reference="nwqlib.algorithms.expectation.ExpectationMethod",
)

_DESCRIPTOR = AlgorithmDescriptor(
    method=_METHOD.name,
    version=_METHOD.version,
    problem_families=("expectation",),
    output_families=("normalized_expectation", "quadratic_form"),
    access_families=("pauli", "dense", "csr", "csc"),
    resource_coverage=(
        "bounded selected preparation/parity size and actual science/calibration shots",
    ),
    evidence_coverage=(
        "exact grouped Pauli expectation, finite-shot parity statistics or explicit provider estimate",
        "conditional fixed-time/time-uniform/Beta inference and selected binary readout calibration",
    ),
    limitations=(
        # 16 repeats algorithms/_eigen_inputs.py::AUTO_DENSE_PAULI_DIMENSION,
        # which this descriptor module does not import. Change both together.
        "quantum: power-of-two Pauli or dense dimension <=16; classical: original Hermitian matvec access",
        "common ErrorModel preserves unknown preparation/simulation/physical accuracy",
        "sampling/channel assumptions are explicit; provider uncertainty does not establish physical accuracy",
        "fixed-time intervals require the original complete acquisition; no automatic anytime conversion",
        "no certified bound or state/output fallback",
    ),
    references=(_METHOD,),
    maintenance="NWQLib",
)
