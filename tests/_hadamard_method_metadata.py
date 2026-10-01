"""Inert registration of the test suite's reference Hadamard Method."""

from nwqlib.algorithms.protocol import AlgorithmDescriptor
from nwqlib.algorithms.registry import Registration
from nwqlib.core import Source

# Keep discovery metadata inert so listing this extension does not import
# its circuit implementation or perform scientific input preparation.
SOURCE = Source(
    name="example.hadamard_pauli_expectation",
    version="1",
    domain="normalized expectation of one real nonzero Pauli term",
    reference="Hadamard test: H-controlled-P-H gives E[Z_ancilla]=Re<psi|P|psi>",
)
DESCRIPTOR = AlgorithmDescriptor(
    method=SOURCE.name,
    version=SOURCE.version,
    problem_families=("expectation",),
    output_families=("normalized_expectation",),
    access_families=("pauli_terms", "state_preparation"),
    resource_coverage=("actual selected PREP, controlled signed Pauli and ancilla readout",),
    limitations=("one nonzero real Pauli term; quantum execution only; no total-error guarantee",),
    references=(SOURCE,),
    maintenance="reference external Method in the NWQLib test suite",
)
# The factory path resolves the implementation only when explicitly used.
# The descriptor states the same narrow domain that the Method admits.
REGISTRATION = Registration(
    source=SOURCE,
    factory="_hadamard_method:HadamardPauliExpectation",
    descriptor=DESCRIPTOR,
)
