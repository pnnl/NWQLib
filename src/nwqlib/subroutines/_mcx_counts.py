"""CX counts of Qiskit's multi-controlled X synthesis without ancillas.

``qiskit_compat.controlled`` adds controls through Qiskit, and Qiskit lowers
a multi-controlled X with its own synthesis. No published closed form gives
the CX count of that synthesis, so this module stores the counts measured
with Qiskit 2.5.2. It imports nothing, so the circuit-free resource laws of
QLS and LCHS read the counts without importing Qiskit.
"""

# CX count of ``controlled(XGate(), k)`` after Qiskit 2.5.2 lowers it without
# ancillas to the basis {cx, u} at optimization level 0, stored at index
# k - 1. That is the logical circuit NWQLib's lowering emits, before hardware
# routing. Open controls add only one-qubit X gates, so every control state
# has the same count. Qiskit's default multi-controlled X synthesis gives
# these counts. They coincide with ``synth_mcx_noaux_v24`` for k <= 5 and
# with ``synth_mcx_noaux_hp24`` (Huang and Palsberg, PLDI 2024) for the larger
# k that were compared. Neither synthesis has a published closed-form CX
# count, so the table stores the values. QLS prices its projector flips and
# its Q_b' and A_t predicates with the table, and the LCHS SELECT laws price
# the QSP reflections and the multi-controlled X gates that Qiskit adds when
# it controls a branch. The table covers 64 controls, well beyond the
# predicates of the 20 to 25 qubit systems in the library's scope. A gate
# with more controls has an unknown CX count, not an extrapolated one.
# tests/test_mcx_counts.py recomputes every entry with the installed Qiskit.
# Revisit when that test fails after a Qiskit upgrade, or when the gates move
# to an ancilla-assisted construction with a closed-form count.
MCX_CX_BY_CONTROLS = (
    1, 6, 14, 36, 84, 136, 192, 264,
    344, 464, 576, 728, 864, 1048, 1200, 1416,
    1624, 1872, 2048, 2328, 2474, 2678, 2762, 2942,
    3010, 3206, 3258, 3470, 3506, 3734, 3754, 3998,
    4002, 4262, 4250, 4526, 4498, 4790, 4746, 5054,
    4994, 5318, 5242, 5582, 5490, 5846, 5738, 6110,
    5986, 6374, 6234, 6638, 6482, 6902, 6730, 7166,
    6978, 7430, 7226, 7694, 7474, 7958, 7722, 8222,
)
