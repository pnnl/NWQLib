"""Immutable scientific Method configuration and direct execution behavior."""

import inspect
from typing import ClassVar

from nwqlib.core.records import Record, Source, Text


class AlgorithmDescriptor(Record):
    """Declared method scope. Applicability belongs to its actual planner.

    Attributes:
        method: Registered method name, such as ``lchs``.
        version: Method version. The (method, version) pair identifies the implementation.
        problem_families: Problem kinds the Method declares, such as ``linear_dynamics``.
        output_families: Output kinds it declares, such as ``solution``.
        access_families: Input representations it declares, such as ``dense``.
        resource_coverage: What its resource estimates cover.
        evidence_coverage: What its error evidence covers.
        limitations: Known limitations stated to users.
        references: Sources of the implemented method.
        maintenance: Maintainer or status label, ``experimental research method`` by default.
    """

    method: Text
    version: Text
    problem_families: tuple[Text, ...] = ()
    output_families: tuple[Text, ...] = ()
    access_families: tuple[Text, ...] = ()
    resource_coverage: tuple[Text, ...] = ()
    evidence_coverage: tuple[Text, ...] = ()
    limitations: tuple[Text, ...] = ()
    references: tuple[Source, ...] = ()
    maintenance: Text = "experimental research method"

    @property
    def source(self):
        return Source(name=self.method, version=self.version,
                      domain="selected scientific method", reference=self.method)


class Method(Record):
    """One immutable configuration with method-owned scientific fields.

    Static methods call execute_static; adaptive methods call their actual
    controller. No shared family-name dispatch decides scientific behavior.

    A Method is kept separate from the Problem it solves, so one scientific
    target can be approached with different initializations, trial spaces,
    degrees or estimators and compared on equal terms. Because a Method is an
    immutable Record, its content identity enters every Plan it selects, and
    changing a configuration field yields a different Plan. The shared
    lifecycle (Run, preparation, submission, storage, reports) calls the
    hooks below and never branches on the method family, so a new Method
    adds science without a common switch. The extension contract, including
    the protected hooks on Plan and Result, is docs/algorithm_protocol.md.

    Hooks a concrete Method provides or may override:

    - ``plan``: select and bind the Plan. Required.
    - ``analyze``: interpret existing RunData into the Result. Required.
    - ``prepare`` and ``execute``: static defaults, overridden by adaptive controllers.
    - ``prepare_all_refusal``: why ``settings="all"`` cannot be served, asked before a Run exists.
    - ``error_model``: framed known and unavailable error sources of the selection.
    - ``validate_point``, ``specialize_experiment`` and ``selected_kernels``:
      admit and derive one realized point without reselecting.
    - ``before_submit``: admit an acquisition before its submission intent is recorded.
    - ``reduction_allowance``: admit a reduction point's workspace and return the Method's remaining work allowance.
    - ``verify`` and ``recover_analysis``: explicit extra operations, unavailable by default.
    """

    descriptor: ClassVar[AlgorithmDescriptor]

    def plan(self, problem, *, output, execution, shots, rng, accuracy=None):
        """Select and bind a Plan for the admitted input and requested quantity.

        Preserve the original problem, configured Method, selected execution mode,
        output and final named RNG states. accuracy is a requested criterion;
        selection must not record it as an achieved error guarantee.
        """
        raise NotImplementedError("a Method must select its scientific experiment")

    def prepare(self, plan, *, run, settings="first"):
        """Prepare the first static setting, or every one for ``settings="all"``, without submitting.

        Controllers override this to prepare their next query. The hook
        receives ``"all"`` only after ``prepare_all_refusal`` returned None.
        """
        from nwqlib._prepared_execution import prepare_static
        return prepare_static(plan, run=run, settings=settings)

    def prepare_all_refusal(self):
        """Why ``prepare(plan, settings="all")`` cannot serve this Method, or None when it can.

        ``nwqlib.prepare`` asks before it creates a Run, so a refusal leaves
        no Run folder. A Method whose later settings depend on earlier outcomes
        overrides this with its reason and what can be prepared instead. A
        ``prepare`` override without the ``settings`` keyword never receives
        ``"all"``, so the default refuses for it.
        """
        parameters = inspect.signature(self.prepare).parameters.values()
        if any(item.name == "settings" or item.kind is item.VAR_KEYWORD for item in parameters):
            return None
        return (
            f"{type(self).__name__}.prepare does not accept the settings keyword, so it can prepare "
            "only the setting it submits first. prepare(plan) with the default settings='first' prepares "
            "that setting. An override that accepts settings and passes it to Method.prepare also "
            "serves settings='all'"
        )

    def execute(self, plan, *, run):
        """Advance the Run's original work; return the Method Result or pending None."""
        from nwqlib._prepared_execution import execute_static
        return execute_static(plan, run=run)

    def analyze(self, plan, data, *, settings):
        """Interpret existing RunData with explicit settings, without acquiring new data.

        Return the concrete result_type with the selected Plan and exact acquisition
        lineage. Result._attach performs the supported Plan/data validation hooks.
        """
        raise NotImplementedError("a Method must interpret its acquired observations")

    def recover_analysis(self, plan, *, run):
        """Only a method with a saved interrupted analysis may recover it."""
        raise ValueError("this Method has no interrupted controller analysis to recover")

    def before_submit(self, plan, prepared, *, run):
        """Admit a tuple of original acquisitions before their submission intent."""

    def error_model(self, plan):
        """Return the selection's framed error sources. The default returns the Plan's stored model."""
        return plan.error_model

    def validate_point(self, plan, experiment, values):
        """Admit method-specific argument relations after generic IR admission."""

    def specialize_experiment(self, plan, experiment, values):
        """Resolve an adaptive readout from its actual admitted arguments."""
        return experiment

    def selected_kernels(self, plan, experiment, program):
        """Resolve selected host declarations without replacing native inputs."""
        return plan.construction.kernels

    def verify(self, plan, result, *, checks):
        """Run explicitly selected checks at their own cost. A Method without checks refuses."""
        raise ValueError("this method has no selected verification operation")


class ApplicabilityError(ValueError):
    """A configured method cannot solve the selected scientific problem/output."""
