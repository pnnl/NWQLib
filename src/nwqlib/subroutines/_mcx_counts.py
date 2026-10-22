"""CX counts of Qiskit's multi-controlled X synthesis without ancillas.

``qiskit_compat.controlled`` adds controls through Qiskit, and Qiskit lowers
a multi-controlled X with its own synthesis. Above 32 controls no published
closed form gives the CX count of that synthesis, so this module stores the
counts measured with Qiskit 2.6.0. It imports nothing, so the circuit-free
resource laws of QLS and LCHS read the counts without importing Qiskit.
"""

# CX count of ``controlled(XGate(), k)`` after Qiskit 2.6.0 lowers it without
# ancillas to the basis {cx, u} at optimization level 0, stored at index
# k - 1. That is the logical circuit NWQLib's lowering emits, before hardware
# routing. Open controls add only one-qubit X gates, so every control state
# has the same count. Qiskit's default multi-controlled X synthesis gives
# these counts. For k <= 32 it is ``synth_mcx_noaux_sp22``, which uses
# Qiskit's dedicated circuits up to four controls and the construction of
# da Silva and Park (Phys. Rev. A 106, 042602 (2022),
# doi:10.1103/PhysRevA.106.042602) from five, with 4k**2 - 4k + 2 CX. For
# larger k it is ``synth_mcx_noaux_hp24`` (Huang and Palsberg, PLDI 2024,
# doi:10.1145/3656436). QLS prices its
# projector flips and its Q_b' and A_t predicates with the table, and the
# LCHS SELECT laws price the QSP reflections and the multi-controlled X gates
# that Qiskit adds when it controls a branch. The table covers 64 controls, well beyond the
# predicates of the 20 to 25 qubit systems in the library's scope. A gate
# with more controls has an unknown CX count, not an extrapolated one.
# tests/test_mcx_counts.py recomputes every entry with the installed Qiskit.
# Revisit when that test fails after a Qiskit upgrade, or when the gates move
# to an ancilla-assisted construction with a closed-form count.
MCX_CX_BY_CONTROLS = (
    1, 6, 14, 36, 82, 122, 170, 226,
    290, 362, 442, 530, 626, 730, 842, 962,
    1090, 1226, 1370, 1522, 1682, 1850, 2026, 2210,
    2402, 2602, 2810, 3026, 3250, 3482, 3722, 3970,
    4002, 4262, 4250, 4526, 4498, 4790, 4746, 5054,
    4994, 5318, 5242, 5582, 5490, 5846, 5738, 6110,
    5986, 6374, 6234, 6638, 6482, 6902, 6730, 7166,
    6978, 7430, 7226, 7694, 7474, 7958, 7722, 8222,
)
