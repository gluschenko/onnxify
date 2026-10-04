# Onnxify.Compiler

`Onnxify.Compiler` is the shared compiler layer for bidirectional transformations between C# TorchSharp computation and ONNX graphs.

## Install

```bash
dotnet add package Onnxify.Compiler
```

The compiler package is also brought in transitively by the Onnxify consumer packages that integrate with the shared compiler boundary.

## Current Scope

The package provides the compiler-owned source, sink, and session contracts plus the immutable computation-tree IR used by both directions. The IR includes typed tensor/value metadata, dimensions, literals, state members, normalized operator attributes, nested ONNX graphs, model metadata, quantization annotations, module calls, reusable blocks, ordered C# syntax, capability classifications, and stage-aware diagnostics.

The ONNX frontend and backend are available through `Compiler.CreateTreeFromOnnx(...)`, `Compiler.GenerateOnnx(...)`, and `Compiler.GenerateOnnxGraph(...)`. File, stream, asynchronous, in-memory, sparse initializer, typed attribute, and nested graph round-trips are represented through compiler-owned IR and return `CompilerResult<T>` diagnostics.

```csharp
var imported = Compiler.CreateTreeFromOnnx("model.onnx");
if (!imported.IsSuccess)
{
    foreach (var diagnostic in imported.Diagnostics)
    {
        Console.WriteLine($"{diagnostic.Code}: {diagnostic.Message}");
    }
}

var emitted = Compiler.GenerateOnnx(imported.Value!);
emitted.Value!.Save("roundtripped.onnx", overwrite: true);
```

Unknown operators can be preserved as generic ONNX operations with an `Unsupported` warning. Full bidirectional operator mappings remain the scope of OXY-024.

The C# TorchSharp frontend accepts either source text or a compiler-neutral descriptor created by
`Onnxify.TorchSharp`. The compiler decompiles only the requested method, scans the supported syntax
subset into immutable IR, and reports unsupported dynamic syntax as diagnostics:

```csharp
using Onnxify.Compiler;
using Onnxify.TorchSharp;

CSharpTorchSharpSource source = module.CreateCompilerSource();
CompilerResult<CompilerComputationTree> tree =
    Compiler.CreateTreeFromTorchSharp(source);

if (tree.IsSuccess)
{
    CompilerResult<string> generated = Compiler.GenerateCSharp(tree.Value!);
    File.WriteAllText("GeneratedTorchModule.cs", generated.Value!);
}
```

The OXY-023 backend emits regular TorchSharp module code, including ordered declarations,
assignments, returns, arrays, tuples, indexers, static `if`/`foreach`, helper blocks, and reusable
module calls. Full bidirectional operator mappings are intentionally handled by OXY-024.

The dependency direction is intentionally one-way:

```text
Onnxify.Compiler -> Onnxify
Onnxify.ModelGenerator -> Onnxify.Compiler
Onnxify.TorchSharp -> Onnxify.Compiler
```

This keeps compiler semantics in one project and prevents the two consumer packages from growing parallel compiler implementations.

## Repository

- Source: <https://github.com/gluschenko/onnxify>
