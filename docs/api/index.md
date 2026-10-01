# API Reference

Start with the [quickstart](../quickstart.md) for a complete example. This reference describes callable signatures, records and return values.

| Task | Reference |
| --- | --- |
| Plan, compare, estimate, execute and inspect results | [Shared workflow](workflow.md) |
| Compute a Pauli expectation | [Expectation](algorithms/expectation.md) |
| Estimate energy or phase | [Lanczos](algorithms/lanczos.md), [QPE](algorithms/qpe.md), [GCiM and ADAPT](algorithms/gcim.md) |
| Evolve a linear system, solve Ax=b, or optimize an objective | [LCHS](algorithms/lchs.md), [QLS](algorithms/qls.md), [QHD](algorithms/qhd.md) |
| Build reusable circuits | Subroutine pages in the navigation |

The root `nwqlib` operations accept bound methods and explicit inputs. Each immutable Method defines scientific configuration and returns the shared Plan and its scientific Result. See [method authoring](../algorithm_protocol.md) for a working Method extension example. The algorithm guides, which the [home page](../index.md) lists by scientific problem, explain which input representations and scientific claims each method supports. Resource estimation and verification are explicit operations with their own work limits.
