# QLS API

Use `LinearSystem(A=..., b=...)` with `QLS`. Its `solver` chooses the inverse polynomial or an explicitly configured shortcut model. The [QLS guide](../../algorithms/qls.md) explains physical outputs, normalization, spectral assumptions and shortcut restrictions.

`solve(problem, method=QLS())` defaults to quantum execution of `qsvt_inverse` with `epsilon_inv=0.01`, auto-selected encoding scale and condition estimate, and physical solution output. This default performs no reference solve. The inverse target controls the selected approximation and does not establish a total-output error guarantee.

::: nwqlib.algorithms.qls.method.QLS

::: nwqlib.algorithms.qls.primary_records.QLSAnalysis

`x` is available only for a selected physical solution output. Unit-vector shortcut results do not supply the physical solution magnitude. Verification is an explicit `result.verify(checks=QLSVerification(...))` operation.

::: nwqlib.algorithms.qls.verification.QLSVerification
