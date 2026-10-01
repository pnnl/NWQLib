"""LCHS PREP, selected evolution and inverse PREP in the actual shared Program."""

from math import fsum

import numpy as np

from nwqlib._preparation_laws import direct_preparation_cx_bound, direct_preparation_controlled_cx_bound
from nwqlib._quantum_readout import qwc_groups
from nwqlib.algorithms._eigen_inputs import AUTO_DENSE_PAULI_DIMENSION
from nwqlib.amplitudes import AmplitudeReadout
from nwqlib.artifacts import ArrayOutput
from nwqlib.blocks.records import BlockSemantics, SelectedConstruction, SelectedDefinition
from nwqlib.blocks.selection import SelectedBlock, _signature, select_pauli_parity, transform_block
from nwqlib.core.planning import Plan, Experiment, ObservationSpec, ReadoutDetails
from nwqlib.core.records import Basis, InputRef
from nwqlib.evidence.records import Evidence
from nwqlib.ir import Allocate, Binding, BlockCall, ClassicalValue, Definition, Measure, MeasurementBatch, MetadataRef, PortMap, Register, Release, Sequence, Setting
from nwqlib.ir.validation import admitted_program
from nwqlib.operators.access import _check_bytes
from nwqlib.operators.inputs import _digest
from nwqlib.problems.inputs import StatePreparationSpec, compose_recovery
from nwqlib.problems.records import Solution, StateVector, Samples, NormalizedExpectation, NormSquared, QuadraticForm
from nwqlib.resources.records import ResourceLaw, Workspace
from .primary_records import METHOD, LCHSReconstruction
from .solution_error_budget import psd_recovery_exponent, psd_recovery_scale


def target_identity(target):
    """Return the SHA-256 identity of the normalized PREP tensor, its dtype, shape and C-order bytes.

    It is the one pass over the vector at each lifecycle stage: planning
    binds it as the PREP input and derives preparation_identity from it, and
    archive loading recomputes it once for the same comparison.
    """
    return _digest("mps.normalized.C-order", (len(target),), (target,))


def preparation_identity(identity,decomposition,layers):
    """Bind the tensor's identity (target_identity), chosen cores and layered synthesis settings.

    Archive loading recomputes this digest and rejects a PREP payload whose
    tensor, cores or layer count differ from the selected record.
    """
    settings = None if decomposition is None else (
        decomposition.num_qubits,decomposition.original_dimension,decomposition.bond_dimensions,
        decomposition.max_bond_dim,decomposition.threshold,decomposition.discarded_weight,
        decomposition.input_id,decomposition.input_norm)
    return _digest('lchs.selected_preparation',(identity,layers,settings),
        () if decomposition is None else decomposition.cores)


def select_vector_preparation(name, target, *, method, decomposition=None, layers=2):
    """Bind the actual normalized tensor and optional already-selected cores."""
    from .native import construct_preparation
    q = len(target).bit_length()-1
    identity = target_identity(target)
    if decomposition is not None and decomposition.input_id != identity:
        raise ValueError("selected MPS cores differ from the actual PREP tensor")
    signature = _signature(name, "lchs.preparation", tuple((f"system_{j}", 1) for j in range(q)))
    input_ref = InputRef(identity=identity,representation="normalized_vector",source=METHOD)
    basis = Basis(identity="lchs-prep",dimension=len(target),ordering="little-endian qubits")
    work, known_bytes = max(1,q)*len(target), 16*len(target)
    if decomposition is not None:
        # The same law and limits that the builder checks when the block is
        # lowered (native.construct_preparation), so lowering cannot refuse
        # a planned construction.
        from nwqlib.subroutines.state_preparation.mps import admit_layered_construction
        work, known_bytes = admit_layered_construction(decomposition.bond_dimensions, layers,
            max_svd_work=method.max_svd_work, max_bytes=method.max_bytes)
    preparation = StatePreparationSpec(input=input_ref,basis=basis,physical_scale=compose_recovery(1.),
        required_ancillas=0,inverse_available=True,control_available=True,
        implementation="qiskit.direct" if decomposition is None else "qiskit.mps_circuit",
        approximation=("ideal direct preparation of the selected normalized tensor" if decomposition is None
            else "selected layered MPS approximation; circuit fidelity not evaluated"),
        work_law="selected direct or layered MPS construction size",items=len(target),payload_bytes=known_bytes,work=work)
    base = direct_preparation_cx_bound(q, complex_phases=bool(np.any(target.imag))) if decomposition is None else 3*layers*max(0,q-1)
    laws = [ResourceLaw(metric="cx", basis="cx", value=base,
        interpretation="upper_bound" if decomposition is None else "estimate",
        evidence=Evidence(kind="proved_relation" if decomposition is None else "numerical_estimate", source=METHOD),
        adjoint=adjoint, assumptions=("selected PREP recipe before routing/optimization",)) for adjoint in (False, True)]
    if decomposition is None:
        laws.extend(ResourceLaw(metric="cx", basis="cx", value=direct_preparation_controlled_cx_bound(q),
            interpretation="upper_bound", evidence=Evidence(kind="proved_relation", source=METHOD),
            controlled=True, adjoint=adjoint, assumptions=("one extra control on the selected direct PREP extension",))
            for adjoint in (False, True))
    record = SelectedDefinition(signature=signature, implementation=signature.target,
        semantics=BlockSemantics(kind="preparation", input=input_ref,
            basis=basis,preparation=preparation,
            relation="U|0> is the selected normalized tensor, or its selected layered MPS approximation",
            input_projector="all zero", output_projector="prepared tensor", success="deterministic PREP from zero input",
            workspace=0, restoration="no additional workspace", epsilon=0. if decomposition is None else None,
            approximation_metric="state L2 on zero input", approximation_evidence="direct ideal circuit relation; MPS circuit fidelity requires explicit validation",
            inverse_legal=True, control_legal=True, phase="the chosen full extension preserves physical phase under inverse/control"),
        choice="direct" if decomposition is None else "mps_circuit", decomposition=None, cost_law=None,
        coefficient_selection_id=preparation_identity(identity,decomposition,layers),
        construction_work=work, resource_laws=tuple(laws),
        cost_parameters=() if decomposition is None else (
            Binding(parameter="mps_layers",value=layers),
            Binding(parameter="mps_max_stored_bond",value=max(decomposition.bond_dimensions)),
            Binding(parameter="mps_core_bytes",value=decomposition.core_bytes)),
        cost_context="actual coefficient/input tensor and selected PREP, not a full-unitary error certificate",
        workspace=(Workspace(location="host", purpose="preparation", bytes=known_bytes, source=METHOD),))
    return SelectedBlock.bind(record, payload=(target, decomposition, layers, method.max_bytes, method.max_svd_work), constructor=construct_preparation)


def observable_terms(problem, output, *, max_bytes):
    """Return the observable's real Pauli terms in the encoded coordinates.

    Exact readout contracts the stored real Pauli table on the physical
    slice and sampled readout measures its qubit-wise commuting groups, so
    the observable must be a real Pauli sum. Pauli input is used as supplied. A dense
    observable is admitted only up to dimension AUTO_DENSE_PAULI_DIMENSION
    (16), where a full Pauli transform of the zero-padded matrix is cheap,
    and must be exactly Hermitian so that no symmetrization changes the
    requested quantity.
    """
    if not isinstance(output, (NormalizedExpectation, QuadraticForm)):
        return ()
    observable = output.observable
    if observable.manifest.basis != problem.basis:
        raise ValueError("observable must use the original physical coordinates")
    if observable.reference.representation == "pauli":
        values = observable.pauli_terms().labels(max_bytes=max_bytes)
    elif observable.reference.representation == "dense" and problem.dimension <= AUTO_DENSE_PAULI_DIMENSION:
        # The small explicit-dense convenience limit shared with Lanczos,
        # GCiM and Expectation (registered in ENGINEERING_CONSTANTS).
        from nwqlib.operators._pauli import pauli_coefficients
        matrix = observable.dense_array()
        if not np.array_equal(matrix, matrix.conj().T):
            raise ValueError("observable must be exactly Hermitian; explicitly symmetrize if that changes the intended problem")
        d = 1 << max(1,(problem.dimension-1).bit_length())
        # Envelope of 64*(q+2) bytes per matrix entry (q = log2 d). It covers
        # the two active complex128 levels of the Pauli block transform and
        # up to d**2 output labels of q characters with their coefficients.
        _check_bytes(64*d*d*(d.bit_length()+1), max_bytes, "small dense observable Pauli access")
        if d != problem.dimension:
            padded = np.zeros((d,d), dtype=complex)
            padded[:problem.dimension,:problem.dimension] = matrix
            matrix = padded
        values = pauli_coefficients(matrix)
    else:
        raise ValueError("LCHS quantum observable needs finite real Pauli access or an explicitly dense "
                         f"dimension <={AUTO_DENSE_PAULI_DIMENSION}")
    if any(value.imag != 0 for _,value in values):
        raise ValueError("Pauli observable coefficients must be real")
    return tuple((label,float(value.real)) for label,value in values)


def plan_quantum(method, problem, data, *, output, shots, rng):
    """Select the phase-weighted LCU body and recovery for the original physical dynamics.

    The success block of PREP, SELECT and inverse PREP is sum_j c_j U_j/alpha
    with alpha = sum_j |c_j| (ACL arXiv:2312.03916v2, Lemma 24, Eq. (178)),
    acting on the unit initial direction. The physical solution is therefore
    alpha times the QSP amplitude recovery (one when QSP is not selected)
    times ||u0|| times the success-projected system amplitudes. With a
    constant source the input norms are already inside the branch
    coefficients, so ||u0|| is not applied again. Qubits are ordered as
    SELECT ancillas, coefficient address, then system, and every non-system
    qubit must read zero for success.
    """
    from .parameters import select_parameters, selected_error_facts, selected_identity, construction_work
    from .time_independent_terms import lchs_quadrature_summary
    from .native import construct_select
    data, extra = select_parameters(method, problem, data)
    coefficients = data.coefficient_plan.coefficients if data.source_layout is None else data.source_layout.coefficients
    a = (len(coefficients)-1).bit_length()
    q = len(data.initial).bit_length()-1
    ancillas = extra["select_ancillas"]
    width = ancillas+a+q
    alpha = fsum(abs(c) for c in coefficients)
    # Recover coefficient normalization and any QSP factor once. Source
    # branches already carry their own input amplitudes and scale. When the
    # PSD recovery exp(shift*T) exceeds binary64, the coefficients hold it
    # divided by 2**e and the recovery carries 2**e.
    growth = psd_recovery_scale(psd_recovery_exponent(data.quadrature.conversion['psd_shift'], problem.elapsed_time))
    recovery = compose_recovery(alpha, extra["qsp_recovery"],
        *(() if data.source is not None else (problem.initial_state.preparation.physical_scale,)),
        *(() if growth is None else (growth,)))
    coordinates, success = tuple(range(ancillas+a,width)), tuple(range(ancillas+a))
    terms = observable_terms(problem, output, max_bytes=method.max_bytes)
    rec = LCHSReconstruction(elapsed_time=problem.elapsed_time, dimension=problem.dimension, encoded_dimension=1<<q,
        mode='unitary' if data.quadrature.l_norm==0 else 'quadrature',
        has_source=data.source is not None, initial_scale=problem.initial_state.preparation.physical_scale,
        source_scale=None if problem.source is None else problem.source.preparation.physical_scale,
        recovery=recovery, success_bits=success, system_bits=coordinates, terms=terms,
        coefficient_l1_norm=alpha, physical_branches=extra["physical_branches"], padded_branches=1<<a,
        selected_backend=method.hamiltonian_evolution_backend, selected_select=data.selected_select,
        psd_premise="numerical" if data.quadrature.numerical_psd_premise_satisfied else "unavailable",
        psd_shift=data.quadrature.conversion["psd_shift"], l_norm=data.quadrature.l_norm,
        kernel_approximation_bound=data.quadrature.coefficient_plan.quadrature.approximate_lchs_error_bound,
        quadrature_bound=data.quadrature.coefficient_plan.quadrature.quadrature_error_bound,
        source_nodes=extra.get("source_nodes",()), source_weights=extra.get("source_weights",()), step_counts=extra.get("step_counts",()))
    identity = selected_identity(data, problem, method)
    work, counts = construction_work(data, extra, method=method, elapsed=problem.elapsed_time)
    if work > method.max_select_work:
        raise ValueError('selected LCHS native construction exceeds max_select_work')
    # The outer QSP gate's classification and the product-formula
    # decomposition and selection draw on the same ledger.
    select_work = (work+method.max_select_work-extra.get("select_work_remaining",method.max_select_work)
                   +extra.get("select_work_charged",0))
    signature = _signature("lchs_select", "lchs.select", tuple((f"bit_{j}",1) for j in range(width)))
    select_record = SelectedDefinition(signature=signature, implementation=signature.target,
        semantics=BlockSemantics(kind="unknown", input=InputRef(identity=identity, representation="selected_lchs_parameters", source=METHOD),
            basis=problem.basis, relation="selected phase-weighted evolution branches; source branches include their own input PREP",
            input_projector="SELECT ancillas zero; arbitrary coefficient address", output_projector="SELECT ancillas zero",
            success="algorithm SELECT block; LCU success also requires inverse coefficient PREP",
            workspace=0, restoration="selected source and SELECT workspace obligations; coefficient address is preserved",
            epsilon=None, approximation_metric="selected evolution operator norm", approximation_evidence="component laws with their stated premises",
            inverse_legal=True, control_legal=True, phase="complex coefficient and identity evolution phases kept"),
        choice=data.selected_select, coefficient_selection_id=identity, decomposition=None, cost_law=None,
        construction_work=work, cost_context="actual SELECT recipe; no coefficient PREP pair in this leaf's cost",
        resource_laws=tuple(ResourceLaw(metric="cx", basis="cx", value=counts["select_cx"],
            interpretation="estimate", evidence=Evidence(kind="numerical_estimate", source=METHOD), adjoint=adjoint,
            assumptions=("selected unoptimized SELECT projection; routing/native synthesis may differ",)) for adjoint in (False,True)),
        workspace=(Workspace(location="host", purpose="workspace", bytes=counts["known_peak_bytes"], source=METHOD),
                   Workspace(location="host", purpose="workspace", bytes=None, source=METHOD)))
    # Homogeneous evolution prepares u0 outside SELECT. Source evolution
    # includes each branch's own input preparation inside the selected action.
    blocks = []
    definitions, calls = [], []
    if data.source is None:
        initial = select_vector_preparation("initial_prep",data.initial_direction,method=method,
            decomposition=data.initial_mps,layers=method.initial_state_mps_num_layers)
        blocks.append(initial)
        definitions.append(Definition(id="initial_prep",node=BlockCall(signature="initial_prep",
            ports=tuple(PortMap(port=f"system_{j}",wire=f"bit_{bit}") for j,bit in enumerate(coordinates)))))
        calls.append("initial_prep")
    # PREP and inverse PREP surround SELECT on the coefficient address only.
    # Successful readout also projects the separate SELECT work ancillas.
    address = tuple(range(ancillas,ancillas+a))
    if a:
        prep = select_vector_preparation("coefficient_prep",extra["amplitudes"],method=method,
            decomposition=data.coefficient_mps,layers=method.mps_num_layers)
        inverse = transform_block("coefficient_inverse",prep,adjoint=True)
        blocks.extend((prep,inverse))
        for name in ("coefficient_prep","coefficient_inverse"):
            definitions.append(Definition(id=name,node=BlockCall(signature=name,
                ports=tuple(PortMap(port=f"system_{j}",wire=f"bit_{bit}") for j,bit in enumerate(address)))))
        calls.append("coefficient_prep")
    blocks.append(SelectedBlock.bind(select_record,payload=(data,problem.elapsed_time),constructor=construct_select))
    definitions.append(Definition(id="select",node=BlockCall(signature=signature.name,
        ports=tuple(PortMap(port=f"bit_{j}",wire=f"bit_{j}") for j in range(width)))))
    calls.append("select")
    if a:
        calls.append("coefficient_inverse")
    raw = lchs_quadrature_summary(data.quadrature,method=data.method,
        he_backend=method.hamiltonian_evolution_backend,final_time=problem.elapsed_time)
    facts = selected_error_facts(data,method=method,problem=problem,output=output,raw=raw)
    return finish_quantum_plan(method,problem,rec,output=output,shots=shots,rng=rng,
        blocks=blocks,definitions=definitions,calls=calls,facts=facts,select_work=select_work,native_data=data,
        coefficient_plan=data.coefficient_plan if data.source is None else None)


def lchs_readout(output, terms, coordinates, *, dimension, shots, method):
    """Select the LCHS readout acquisitions and charge QWC grouping to max_readout_work.

    Returns ``(settings, groups, comparisons)``. A vector output reads its
    amplitudes and Samples one counts setting. With ``shots=None`` a scalar
    output has one exact ``projected_moments`` reduction. With shots,
    NormSquared measures its physical mass; a Pauli observable measures one
    counts setting per first-fit qubit-wise commuting group of its nonzero
    nonidentity labels, whose label is the group's accumulated basis, and a
    padded normalized output adds the unrotated ``physical_mass`` setting.
    ``groups`` lists each group setting's member labels. The grouping
    comparisons are admitted tile by tile against max_readout_work, and
    comparisons is the actual evaluated count.
    """
    from nwqlib._quantum_readout import PROJECTED_MOMENTS, ReadoutSetting
    identity = "I"*len(coordinates) or "I"
    projection = dimension != 1 << len(coordinates)
    if isinstance(output,(Solution,StateVector,Samples)):
        return (ReadoutSetting(name="setting_0",label=identity),),(),0
    if shots is None:
        return (ReadoutSetting(name=PROJECTED_MOMENTS,label=identity),),(),0
    if isinstance(output,NormSquared):
        return (ReadoutSetting(name="physical_mass",label=identity,physical_projection=projection),),(),0
    groups, bases, comparisons = qwc_groups(
        terms, len(coordinates), max_comparisons=method.max_readout_work,
        max_bytes=method.max_bytes, limit_name="LCHS.max_readout_work")
    if not groups:
        groups,bases = ((),),(identity,)
    settings = tuple(ReadoutSetting(name=f"group_{index}",label=basis) for index,basis in enumerate(bases))
    if isinstance(output,NormalizedExpectation) and projection:
        settings += (ReadoutSetting(name="physical_mass",label=identity,physical_projection=True),)
    return settings,groups,comparisons


def observed_bits(rec, output, setting):
    """Return the qubits measured for one counts setting, in classical-bit order.

    Every counts setting observes all success and system qubits, since the
    Program's classical layout is shared by all counts batches. A group
    setting decodes each label's parity from the coordinates of its support
    and ignores the other measured coordinates; the physical-mass setting
    and Samples use every coordinate.
    """
    return rec.success_bits+rec.system_bits


def _rotations(rec, setting, encoded_basis, blocks, definitions, rotations):
    """Return the selected one-site basis changes of a group setting, selecting each once per Plan.

    A group basis rotates each X coordinate by H and each Y coordinate by S†
    then H, once, with no parity network.
    """
    q = len(rec.system_bits)
    names = []
    for bit,axis in enumerate(reversed(setting.label)):
        if axis not in "XY":
            continue
        name = f"rotate_{axis.lower()}_{bit}"
        if name not in rotations:
            label = "".join(axis if j==bit else "I" for j in reversed(range(q)))
            block = select_pauli_parity(name,label,basis=encoded_basis)
            blocks.append(block)
            definitions.append(Definition(id=name,node=BlockCall(signature=block.record.signature.name,
                ports=tuple(PortMap(port=f"system_{j}",wire=f"bit_{wire}") for j,wire in enumerate(rec.system_bits)))))
            rotations.add(name)
        names.append(name)
    return names


def finish_quantum_plan(method,problem,rec,*,output,shots,rng,blocks,definitions,calls,facts=(),select_work=0,**native):
    """Share the selected coherent body across the selected readout acquisitions.

    A vector output reads its amplitudes. With ``shots=None`` a scalar output
    has one experiment: the coherent body and one ``projected_moments``
    reduction at its end, bound to the success bits, coordinates, original
    dimension and stored observable; its bytes and work are admitted against
    ``max_bytes`` and the remaining ``max_readout_work`` before native
    acquisition (``LCHS.reduction_allowance``). With shots,
    each counts setting rotates its group basis and measures all success
    and system qubits. ``select_work`` is the max_select_work already
    charged by the construction.
    """
    from nwqlib._quantum_readout import PROJECTED_MOMENTS, projected_parameters
    from nwqlib.core.planning import ObservationPoint
    from .analysis import error_model
    settings,groups,comparisons = lchs_readout(output,rec.terms,rec.system_bits,dimension=rec.dimension,shots=shots,
        method=method)
    rec = rec.revise(settings=settings, groups=groups,
        grouping_comparisons=comparisons, select_work=select_work)
    width = len(rec.success_bits)+len(rec.system_bits)
    vector = isinstance(output,(Solution,StateVector))
    reduction = shots is None and not vector
    declarations = [Register(name=f"bit_{j}",width=1,role="clean_ancilla" if j in rec.success_bits else "system") for j in range(width)]
    allocate,release = [],[]
    for j in range(width):
        allocate.append(f"allocate_{j}")
        release.append(f"release_{j}")
        definitions.extend((Definition(id=allocate[-1],node=Allocate(wire=f"bit_{j}")),Definition(id=release[-1],node=Release(wire=f"bit_{j}"))))
    classical,measure = (),[]
    if shots is not None:
        # Every counts setting measures all success and system qubits into
        # the shared classical layout, so one measurement per bit serves all.
        classical = tuple(ClassicalValue(name=f"readout_{j}",dtype="bits",width=1) for j in range(width))
        for j,bit in enumerate(rec.success_bits+rec.system_bits):
            measure.append(f"measure_{j}")
            definitions.append(Definition(id=measure[-1],node=Measure(wire=f"bit_{bit}",result=f"readout_{j}")))
    experiments,batches = [],[]
    encoded_basis = Basis(identity="lchs-encoded",dimension=rec.encoded_dimension,ordering="little-endian original index then zero dummy coordinates")
    rotations = set()
    for index,setting in enumerate(rec.settings):
        body = [*allocate,*calls]
        if shots is not None:
            body.extend(_rotations(rec,setting,encoded_basis,blocks,definitions,rotations))
            body.extend(measure)
        if not vector and not reduction:
            body.extend(release)
        definitions.append(Definition(id=f"body_{index}",node=Sequence(children=tuple(body))))
        if vector or reduction:
            continue
        name = f"batch_{index}"
        batches.append(name)
        definitions.append(Definition(id=name,node=MeasurementBatch(body=f"body_{index}",
            settings=(Setting(label=setting.name,metadata=MetadataRef(format=METHOD,data=InputRef(identity=problem.content_id,representation="scientific_input",source=METHOD))),),
            repetitions=shots,observation_kind="counts")))
        experiments.append(Experiment(name=setting.name,batch=name,setting_index=0,readout=ReadoutDetails()))
    root = "body_0" if vector or reduction else "settings"
    if root == "settings":
        definitions.append(Definition(id=root,node=Sequence(children=tuple(batches))))
    program = admitted_program("LCHS.max_admission_steps",method.max_admission_steps,root=root,definitions=tuple(definitions),registers=tuple(declarations),classical=classical,
        signatures=tuple(block.record.signature for block in blocks))
    construction = SelectedConstruction(program=program,selections=tuple(block.record for block in blocks))
    if vector:
        array = ArrayOutput(name="solution" if isinstance(output,Solution) else "state_vector",kind="vector",basis=problem.basis,
            frame="physical" if isinstance(output,Solution) else output.normalization,
            global_phase="physical" if isinstance(output,Solution) else output.global_phase)
        readout = AmplitudeReadout(construction_id=construction.content_id,source=METHOD,width=width,coordinates=rec.system_bits,
            success=tuple((bit,0) for bit in rec.success_bits),output=array,recovery=rec.recovery)
        experiments = [Experiment(name=rec.settings[0].name,setting=rec.settings[0].name,observation=ObservationSpec(kind="amplitudes",amplitudes=readout))]
    elif reduction:
        parameters = projected_parameters(coordinates=rec.system_bits,success=tuple((bit,0) for bit in rec.success_bits),
            dimension=rec.dimension,terms=rec.terms,moment=not isinstance(output,NormSquared))
        point = ObservationPoint(id="end",kind="reduction",reducer=PROJECTED_MOMENTS,parameters=parameters)
        experiments = [Experiment(name=PROJECTED_MOMENTS,setting=PROJECTED_MOMENTS,
            observation=ObservationSpec(kind="trajectory",positions=(point,)))]
    plan = Plan(problem=problem,method=method,output=output,execution="quantum",shots=shots,randomness=rng.snapshot(),
        construction=construction,experiments=tuple(experiments),reconstruction=rec,facts=tuple(facts),
        error_model=error_model(problem,output,construction,facts,shots=shots),
        assumptions=("construction tolerance is not a total physical output error bound",))
    return plan._bind(blocks=tuple(blocks),**native)
