## 0.4.0

- Introduced the shared compiler package boundary for bidirectional C# TorchSharp and ONNX transformations.
- Added compiler-owned source, sink, and session contracts with acyclic dependency direction through the core `Onnxify` package.
- Added immutable compiler IR types for computation trees, value types, dimensions, literals, state members, operations, module calls, reusable blocks, and ordered C# syntax.
- Added normalized operator descriptors, capability classification, source spans, structured diagnostics, and `CompilerResult<T>` results.
- Added internal tree builders with duplicate-name and reference validation plus structural-equality coverage for representative IR fixtures.
- Added the ONNX frontend/backend with in-memory, file, stream, and asynchronous entry points.
- Added model envelopes, graph metadata, quantization annotations, sparse tensors, typed attributes, nested graph literals, outer-scope captures, and generic unsupported-operator preservation.
- Added ONNX Runtime-backed compiler round-trip tests for `net8.0` and `net10.0`.
- Added the compiler-owned C# TorchSharp frontend with source-text scanning, method-token decompilation, immutable syntax IR, source spans, and diagnostics for unsupported dynamic syntax.
- Added `CompilerTorchSharpModuleDescriptor` and the thin `Onnxify.TorchSharp` adapter for runtime module metadata, state members, child modules, and helper methods.
- Added `Compiler.GenerateCSharp(...)` with deterministic TorchSharp source generation for declarations, assignments, returns, arrays, tuples, indexers, static control flow, helper blocks, and module calls.
- Added Roslyn compilation and minimal executable TorchSharp generated-module tests for `net8.0` and `net10.0`.
