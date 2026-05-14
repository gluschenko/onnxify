# Onnxify.ModelGenerator

## Purpose

Use `Onnxify.ModelGenerator` when the user already has one or more `.onnx` files and wants compile-time generated, strongly typed `Microsoft.ML.OnnxRuntime` wrappers instead of hand-written `InferenceSession` plumbing.

This package is a Roslyn source generator. It inspects ONNX files during build and emits wrapper types that:

- expose typed input and output contracts
- preserve real ONNX input and output names
- surface model metadata through `Onnxify.OnnxValue` descriptors
- provide `Run(...)` overloads over `InferenceSession`

## When To Use It

Prefer `Onnxify.ModelGenerator` when:

- the user wants to run inference against an existing ONNX model from .NET
- the model contract is stable enough that compile-time wrappers are helpful
- stringly typed `NamedOnnxValue` code is getting repetitive or error-prone
- multiple app components need the same model contract and should share one generated wrapper

Do not default to it when:

- the task is about editing or creating ONNX graphs themselves
  Use the main `Onnxify` library for graph authoring and model mutation.
- the task is about exporting TorchSharp modules to ONNX
  Use `Onnxify.TorchSharp`.
- the user only needs a one-off manual inference snippet for a single model and explicitly does not want generated code

## Why Use It

Compared with hand-written `OnnxRuntime` glue, `Onnxify.ModelGenerator` gives you:

- generated input DTOs with tensor types and ONNX-derived property names
- generated output wrappers with typed accessors
- fewer repeated string literals for model input and output names
- a default model path strategy that keeps the `.onnx` file beside the application output
- discoverable metadata through static `Inputs`, `Outputs`, and `OutputNames`

It is especially useful for app code that wants a small, ergonomic runtime surface while still respecting the real ONNX contract.

## Recommended Package Setup

The generator package is meant to be used together with:

- `Onnxify` for metadata types referenced by generated code
- `Microsoft.ML.OnnxRuntime` for runtime execution

Recommended `csproj` shape:

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net10.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="Onnxify" Version="0.0.0.11" />
    <PackageReference Include="Onnxify.ModelGenerator" Version="0.0.0.11" />
    <PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.23.2" />
  </ItemGroup>

  <ItemGroup>
    <OnnxModel Include="Models\sample-classifier.onnx" />
  </ItemGroup>
</Project>
```

Use the custom `OnnxModel` item instead of wiring `AdditionalFiles` manually unless the user has a specific build reason not to. The packaged `.targets` file forwards `OnnxModel` items into `AdditionalFiles` and also keeps the model copied to output by default.

## Naming And Namespace Overrides

If the generated type name or namespace should differ from the file name or project root namespace, set metadata on the `OnnxModel` item:

```xml
<ItemGroup>
  <OnnxModel Include="Models\sample-classifier.onnx"
             OnnxifyModelClassName="SampleClassifier"
             OnnxifyModelNamespace="MyApp.Models" />
</ItemGroup>
```

This produces a generated wrapper named `SampleClassifierModel` in namespace `MyApp.Models`.

## Typical Runtime Usage

Assuming the model declares one required input `input_ids` and one output `logits`, the generated API shape will look like this:

```csharp
using Microsoft.ML.OnnxRuntime.Tensors;
using MyApp.Models;

var model = new SampleClassifierModel();

var inputIds = new DenseTensor<long>(
    values: new long[] { 101, 2023, 2003, 1037, 3231, 102 },
    dimensions: new[] { 1, 6 }
);

using var outputs = model.Run(inputIds);
Tensor<float> logits = outputs.Logits;
```

If the user prefers the object-style input contract instead of positional overloads:

```csharp
using Microsoft.ML.OnnxRuntime.Tensors;
using MyApp.Models;

var model = new SampleClassifierModel();

var inputs = new SampleClassifierModelInputs
{
    InputIds = new DenseTensor<long>(
        values: new long[] { 101, 7592, 2088, 102 },
        dimensions: new[] { 1, 4 }
    )
};

using var outputs = model.Run(inputs);
Tensor<float> logits = outputs.Logits;
```

## Build And Deployment Notes

- The default wrapper constructor loads the model from `DefaultModelPath`, which resolves relative to the application output directory.
- If the ONNX model uses external tensor data, the generated wrapper can still be emitted, but deployment must include the sibling external-data files too.
- Optional ONNX inputs become nullable tensor properties and nullable tensor parameters in generated `Run(...)` overloads.
- The generator only supports tensor inputs and outputs backed by `Microsoft.ML.OnnxRuntime`-compatible CLR element types.

## Related Repo Entry Points

- `src/Onnxify.ModelGenerator/OnnxModelGenerator.cs`
- `src/Onnxify.ModelGenerator/Onnxify.ModelGenerator.props`
- `src/Onnxify.ModelGenerator/Onnxify.ModelGenerator.targets`
- `src/Onnxify.Tests/OnnxModelGeneratorTests.cs`

## Heuristics

- If the user asks for "typed wrappers around an existing ONNX file", default to this package.
- If the user asks to inspect an ONNX model before writing inference code, combine this page with [inference-from-onnx.md](inference-from-onnx.md).
- If the user needs to author or mutate ONNX graphs, switch back to the main `Onnxify` object model rather than forcing everything through generated inference wrappers.
