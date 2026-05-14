# Onnxify.ModelGenerator

`Onnxify.ModelGenerator` is a Roslyn source generator that turns `.onnx` files in your project into typed `Microsoft.ML.OnnxRuntime` wrapper classes.

## Install

```bash
dotnet add package Onnxify.ModelGenerator
```

In a real consumer project you will typically also reference `Onnxify` and `Microsoft.ML.OnnxRuntime`, because the generated code uses `Onnxify` metadata types and executes through ONNX Runtime.

## When To Use It

Use `Onnxify.ModelGenerator` when you already have an ONNX model and want:

- strongly typed wrapper classes instead of hand-written `InferenceSession` plumbing
- generated input and output contracts based on the real ONNX signature
- fewer hard-coded input and output names in application code
- a small, reusable inference surface for app or service code

Use the main `Onnxify` package instead when your goal is to build, inspect, or edit ONNX graphs directly.

## What It Provides

- Detect `.onnx` files added to the consuming project through the `OnnxModel` MSBuild item.
- Generate typed input and output contracts from the model signature.
- Emit a thin `InferenceSession` wrapper with `Run(...)` overloads for `Microsoft.ML.OnnxRuntime`.
- Surface model input and output metadata in generated code for runtime inspection.

## Recommended `csproj` Setup

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

The packaged `.targets` file forwards `OnnxModel` items to Roslyn as additional files and keeps the model copied to the output directory by default.

## Naming Overrides

If you want a custom namespace or class name, set metadata on the `OnnxModel` item:

```xml
<ItemGroup>
  <OnnxModel Include="Models\sample-classifier.onnx"
             OnnxifyModelNamespace="MyApp.Models"
             OnnxifyModelClassName="SampleClassifier" />
</ItemGroup>
```

This generates a wrapper named `SampleClassifierModel` in namespace `MyApp.Models`.

## Runtime Example

Assuming the ONNX model exposes an input named `input_ids` and an output named `logits`:

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

You can also use the generated input contract object:

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

## Notes

- The default constructor loads the model from `DefaultModelPath`, which resolves relative to the application output folder.
- Optional ONNX inputs become nullable tensor properties and nullable tensor parameters in generated `Run(...)` overloads.
- Models that use external tensor data still require their sibling external-data files at deployment time.
- If you need to wire ONNX files manually, the generator ultimately reads Roslyn additional files, but the recommended project-facing entry point is `OnnxModel`.

## Repository

- Source: <https://github.com/gluschenko/onnxify>
