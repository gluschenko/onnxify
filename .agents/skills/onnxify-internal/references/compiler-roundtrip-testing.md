# Compiler Roundtrip Test Methodology

Use this page whenever adding or changing compiler operators or roundtrip behavior under OXY-019 and OXY-024—OXY-027.

## Test Project

Create and maintain `src/Onnxify.Compiler.Tests/Onnxify.Compiler.Tests.csproj`, add it to `src/Onnxify.slnx`, and reference `Onnxify.Compiler` from it. This is the authoritative home for verified roundtrip tests of every supported ONNX <=> TorchSharp operator. Keep package-specific API smoke tests in their owning projects, but keep shared operator semantics and directional parity tests here.

## Required Directions

Every supported operator requires both directions:

- `ONNX -> TorchSharp`: load or construct an ONNX fixture, create the compiler tree, generate TorchSharp C# source, compile/execute it where possible, and compare with the original ONNX Runtime result.
- `TorchSharp -> ONNX`: execute the TorchSharp source fixture, create the compiler tree, emit ONNX, create an ONNX Runtime session, and compare deterministic outputs and relevant graph semantics.

An operator is incomplete if only one direction is tested. If the reverse direction is genuinely impossible or lossy, classify it explicitly and add a test proving the diagnostic or fallback. Never silently omit the reverse test.

## Upstream Source Order

Use repository tests as the semantic source of truth in this order:

1. First inspect `third_party/onnxscript/tests`. Use the exact converter/operator tests and parameterized cases to determine intended Torch-to-ONNX behavior, accepted spellings, attributes, shapes, dtypes, and numerical expectations.
2. Then inspect `third_party/onnxruntime/onnxruntime/test` for ONNX operator semantics, type/shape constraints, edge cases, backend behavior, and runtime-kernel coverage.
3. Only then inspect current Onnxify implementation details. Existing behavior is not proof of complete or intended coverage.

## Mandatory Provenance Comments

Every operator test in `Onnxify.Compiler.Tests` must contain comments linking to the exact upstream tests that motivated it. Link to the repository-relative paths, identify the test function or test group, and include both sources when both were consulted:

```csharp
// Source: third_party/onnxscript/tests/converter/test_activations.py (test_relu).
// Runtime semantics: third_party/onnxruntime/onnxruntime/test/...
```

If no direct upstream counterpart exists, say so in the comment and cite the closest specification/runtime tests used to derive the fixture.

## Required Coverage Record

For each operator or family, record in tests or adjacent documentation:

- ONNX operator and TorchSharp/API spellings;
- ranks, dtypes, attributes, defaults, broadcasting, shape behavior, and limitations;
- exact ONNXScript and ONNX Runtime source-test links;
- fixtures for both directions;
- graph assertions, ONNX Runtime session creation, and numerical parity where executable;
- an explicit reason for every skipped case.

## Complete Solutions Over Point Fixes

Do not fix only one failing overload or fixture when the issue belongs to a shared semantic family. Before implementation:

1. Enumerate the complete relevant ONNXScript and ONNX Runtime case set.
2. Identify the common abstraction, normalization rule, registry entry, shape/dtype rule, or lowering path.
3. Implement the reusable solution in `Onnxify.Compiler` for the complete declared family, including equivalent TorchSharp spellings and attributes.
4. Add the full paired verification matrix before marking the work complete.

Adding only ReLU is not sufficient when the supported activation family has not been audited. Apply the same standard to matmul variants, shape operations, factories, reductions, casts, selection operations, and neural-network layers. A fixture-specific special case is acceptable only when upstream semantics are genuinely special and the reason is documented in the mapping and tests.

## Definition Of Done

Compiler operator work is complete only when it has a common implementation path, complete declared family coverage, upstream-linked tests, paired `ONNX -> TorchSharp` and `TorchSharp -> ONNX` verification, runtime/numerical parity where executable, and explicit diagnostics for unsupported cases.
