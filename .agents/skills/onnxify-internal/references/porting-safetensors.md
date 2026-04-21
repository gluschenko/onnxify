# Porting `third_party/safetensors` To `Onnxify.Safetensors`

Use this reference when porting the upstream safetensors implementation from `third_party/safetensors` into `src/Onnxify.Safetensors`.

## Goal

Build a fully managed C# implementation of safetensors behavior in `Onnxify.Safetensors` while preserving upstream compatibility and reusing existing Onnxify numeric primitives where possible.

## 1. Treat Upstream As The Behavioral Source Of Truth

Start from:

- `third_party/safetensors/safetensors/src`

When behavior is ambiguous, prefer matching the native safetensors implementation rather than inventing repo-local semantics.

This includes:

- parsing and validation rules
- metadata interpretation
- error conditions and rejection cases
- tensor layout and slicing behavior
- compatibility-sensitive edge cases

## 2. Port The Runtime As Pure C#

Implement the feature set directly in `src/Onnxify.Safetensors`.

- Do not wrap the Rust crate.
- Do not depend on a native bridge for core functionality.
- Keep the implementation self-contained and idiomatic for .NET runtime and packaging expectations.

The target is feature parity through C# source, not interop.

## 3. Preserve Native Behavior Patterns

The C# port should follow the original implementation's behavior as closely as practical.

Aim to keep parity for:

- validation order
- accepted and rejected inputs
- offset and length calculations
- tensor view semantics
- observable failure modes

If you need to diverge for safety or CLR constraints, keep the divergence explicit and narrow.

## 4. Mirror The Upstream API With .NET Naming

Keep the public API easy to map from the original safetensors surface so upstream docs and examples remain useful during porting.

At the same time:

- use standard C# naming conventions
- prefer `PascalCase` for methods, properties, and public members
- keep the conceptual API shape close to the original even when names change to match .NET style
- prefer `DataType` in the public C# API rather than `Dtype`; reserve `dtype` for the upstream terminology and the on-disk/header field name

The goal is "familiar to upstream readers, natural to C# consumers."

## 5. Reuse Onnxify Numeric Types First

Before introducing new numeric or storage helper types in `Onnxify.Safetensors`, inspect `src/Onnxify` for an existing equivalent.

Prefer reusing Onnxify-provided low-precision and quantized representations such as:

- Float8 variants
- `Int4`
- `Int2`

If Onnxify already models the required storage type, use that type instead of creating a duplicate safetensors-local version.

## 6. Port Upstream Tests Into `Onnxify.Tests`

The best way to guarantee behavior parity is to translate existing tests from the original implementation into C#.

Prefer adding them under:

- `src/Onnxify.Tests`

Use upstream Rust tests as the baseline for:

- success cases
- malformed file handling
- boundary conditions
- slicing/view behavior
- dtype coverage

Translating tests is not just coverage work; it is the main mechanism for locking in parity with the original implementation.

## 7. Document Ported Entities With XML Comments

When a `third_party/safetensors` entity is ported into `src/Onnxify.Safetensors`, add XML documentation to the C# entity instead of leaving provenance only in commit history or review context.

For each public or internal ported entity:

- add a short `summary` that explains what the type/member is for
- describe the key behavior that matters to a C# consumer, especially validation, ownership, and view/slicing semantics
- include the original Rust file path
- include the original Rust entity name

When the C# code introduces a helper type or helper method that does not exist as a standalone Rust entity:

- still add XML docs
- state that it is a local C# helper extracted from a specific Rust function, match arm, tuple, or logic block

Prefer putting upstream provenance in `remarks`, for example:

- `Original Rust file: third_party/safetensors/safetensors/src/tensor.rs`
- `Original Rust entity: SafeTensors::read_metadata`

This rule exists so IntelliSense and code review can answer two questions immediately:

- what does this managed entity do
- which upstream Rust behavior it is supposed to track

## 8. Suggested Porting Order

When the scope is large, port in this order:

1. Core runtime and format parsing from `third_party/safetensors/safetensors/src`.
2. Validation and metadata rules that determine compatibility.
3. Tensor slicing and view behavior.
4. Data type coverage using existing Onnxify numeric primitives where available.
5. Upstream test cases rewritten in C# to prove equivalence.

## 9. Read Release Notes Before Porting New Features

Before starting work on newly announced safetensors functionality, review the upstream release notes:

- [safetensors releases](https://github.com/huggingface/safetensors/releases)

Use the release notes to understand:

- which features or API changes have been announced upstream
- whether a behavior change is expected to land in the Rust sources soon
- which tests or edge cases may need to be added once the submodule is updated

Important restriction:

- Do not implement features solely because they appear in the release notes.
- Only port functionality that is already present in the checked-in upstream source under `third_party/safetensors`.
- If a feature is mentioned in release notes but the corresponding code has not yet been pulled into the submodule, treat it as future work and wait until the submodule update lands.

Release notes are a planning input, not the implementation source of truth. The implementation source of truth remains the vendored upstream code in `third_party/safetensors`.

## Heuristics

- Prefer behavioral equivalence over line-for-line translation.
- Reuse Onnxify infrastructure when it preserves semantics; avoid duplicating existing primitives.
- Keep public API mapping obvious for developers comparing C# code to the upstream safetensors sources.
- If an upstream behavior is intentionally unsupported in the first pass, fail clearly and cover that choice with tests.
- Use release notes to anticipate upcoming work, but gate actual implementation on the submodule contents in `third_party/safetensors`.
- Keep XML docs current when the port changes. If behavior, naming, or ownership semantics move, update the summary and the upstream provenance note in the same change.

## Porting Notes From The First C# Slice

- Configure header decoding to use strict UTF-8 validation. In C#, this means using a `UTF8Encoding` instance that throws on invalid bytes so malformed headers fail the same way as upstream.
- Preserve upstream serialization ordering exactly. For byte-exact parity, tensor entries must be emitted in the same effective order as upstream, including sorting by dtype alignment and then by tensor name.
- Be careful when porting slice iteration logic. The upstream Rust code relies on a specific chunk ordering pattern; a seemingly equivalent C# translation can still return the right ranges in the wrong order.
- Prefer small buffer-builder helpers in C# tests for malformed or edge-case fixtures. Hand-writing full byte arrays for safetensors headers is brittle and can accidentally test the wrong failure mode because of an incorrect encoded header length.
- Keep the public C# type/property name fixed as `DataType`; use `dtype` only where the upstream format or release/test language requires that exact term.
- When consuming the API from other projects, watch for namespace/type-name collisions around `Onnxify.Safetensors` and `Safetensors`. Use explicit qualification or aliases when needed so examples stay unambiguous.
