# Onnxify.ModelGenerator

`Onnxify.ModelGenerator` is a Roslyn source generator that turns `.onnx` files in your project into typed `Microsoft.ML.OnnxRuntime` wrapper classes.

## Why This Package Exists

Using ONNX Runtime directly is powerful, but repetitive:

- you have to load the model manually with `InferenceSession`
- you have to keep string-based input and output names in sync with the real ONNX signature
- you have to map tensors into `NamedOnnxValue` collections by hand
- you have to remember which outputs should be disposed and when

`Onnxify.ModelGenerator` removes that plumbing. You add an ONNX model to your project, and the generator emits a small typed API around it:

- a model wrapper such as `SampleClassifierModel`
- an input contract such as `SampleClassifierModelInputs`
- an output contract such as `SampleClassifierModelOutputs`
- typed `Run(...)` overloads with optional `RunOptions`

Use the main `Onnxify` package instead when your goal is to inspect, build, or edit ONNX graphs themselves.

## Install

```bash
dotnet add package Onnxify
dotnet add package Onnxify.ModelGenerator
dotnet add package Microsoft.ML.OnnxRuntime
```

## Recommended `csproj` Setup

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net10.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="Onnxify" Version="0.1.0" />
    <PackageReference Include="Onnxify.ModelGenerator" Version="0.1.0" />
    <PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.24.4" />
  </ItemGroup>

  <ItemGroup>
    <OnnxModel Include="Models\sample-classifier.onnx"
               OnnxifyModelNamespace="MyApp.Models"
               OnnxifyModelClassName="SampleClassifier"
               OnnxifyModelImportType="OnnxRuntimeInference" />
  </ItemGroup>
</Project>
```

The packaged `.targets` file forwards `OnnxModel` items to Roslyn as additional files and keeps the model copied to the output directory by default.

## Import Type

`OnnxifyModelImportType` controls which generated source shape is emitted for each `OnnxModel` item. The metadata value is case-insensitive and accepts one or more comma-separated values.

There are two import types:

- `OnnxRuntimeInference` generates a typed ONNX Runtime inference wrapper.
- `TorchModule` generates a graph-shaped TorchSharp module that reconstructs supported ONNX graphs as idiomatic `torch.nn.Module` code.

If `OnnxifyModelImportType` is omitted or empty, the generator uses `OnnxRuntimeInference`.

### OnnxRuntimeInference

Use `OnnxRuntimeInference` when you want the default typed `Microsoft.ML.OnnxRuntime` inference wrapper:

```xml
<OnnxModel Include="Models\sample-classifier.onnx"
           OnnxifyModelImportType="OnnxRuntimeInference" />
```

This mode is the right choice for production inference paths that should stay close to ONNX Runtime. It generates:

- a disposable model wrapper such as `SampleClassifierModel`
- input and output contract types
- typed `Run(...)` overloads
- model signature metadata through `Inputs`, `Outputs`, and `OutputNames`
- constructors for loading the model from the default project-relative path, a custom file path, or raw model bytes

`OnnxRuntimeInference` requires `Microsoft.ML.OnnxRuntime` in the consuming project. It does not try to reinterpret the graph as C# model code; it keeps ONNX Runtime as the execution engine and removes the repetitive session/input/output plumbing around it.

### TorchModule

Use `TorchModule` when you want a graph-shaped TorchSharp module reconstructed from supported ONNX operators:

```xml
<OnnxModel Include="Models\sample-classifier.onnx"
           OnnxifyModelImportType="TorchModule" />
```

This is the most distinctive import mode in `Onnxify.ModelGenerator`: it turns a compatible ONNX graph into a real TorchSharp module. Instead of treating the ONNX file only as an opaque runtime artifact, the generator rebuilds the supported graph structure as `torch.nn.Module<Tensor, Tensor>` code that can live naturally inside a TorchSharp application.

That is a big deal for .NET ML workflows. `TorchModule` gives you a bridge back from an exported ONNX artifact to an editable, composable TorchSharp model surface: you can inspect the generated module, compose it with handwritten TorchSharp code, move weights across compatible graph variants, and keep working in the TorchSharp ecosystem without manually rebuilding the model layer by layer.

`TorchModule` generates a `<ModelName>TorchModule` type with:

- private TorchSharp module fields for recognized module-shaped operators such as convolutions, linear layers, recurrent layers, normalization layers, pooling, and activations
- inline TorchSharp tensor expressions for supported primitive ONNX operators
- registered TorchSharp parameters for trainable float32 initializers
- registered buffers for non-parameter constants
- a TorchSharp `forward(...)` method that follows the ONNX graph order
- `LoadWeightsFromOnnx(string modelPath)` and `LoadWeightsFromOnnx(OnnxModel model)` helpers for copying initializer values into the generated module

The weight loader is designed for practical model surgery. It canonicalizes the source graph before loading weights and can fall back from initializer names to deterministic canonical initializer indexes. That means structurally identical models can still transfer weights even when node or value names differ.

`TorchModule` requires TorchSharp in the consuming project. It currently supports ONNX graphs that can be reconstructed from the generator's supported operator set, with exactly one non-initializer runtime input and one graph output for the main module shape. Unsupported graphs report an `OMG006` diagnostic during generation, so failures are visible at build time instead of surfacing later during inference.

This mode is especially useful when you want to:

- start from an ONNX model but continue working in TorchSharp
- inspect or adapt the model as C# module code
- load compatible ONNX weights into a TorchSharp module
- compose generated modules with handwritten TorchSharp layers or training pipelines
- keep a path from exported ONNX artifacts back into a trainable or editable .NET model representation

### Generating Both APIs

Use both values when the same model should generate both APIs:

```xml
<OnnxModel Include="Models\sample-classifier.onnx"
           OnnxifyModelImportType="OnnxRuntimeInference,TorchModule" />
```

This is a strong default while evaluating a model integration: use the ONNX Runtime wrapper for deployment-style inference and the TorchSharp module for experiments, adaptation, debugging, or weight transfer.

With the configuration above, the generator emits types like:

- `MyApp.Models.SampleClassifierModel`
- `MyApp.Models.SampleClassifierModelInputs`
- `MyApp.Models.SampleClassifierModelOutputs`

When `TorchModule` is enabled, the generator also emits `SampleClassifierModelTorchModule`.

## Example: Run Inference With `SessionOptions` And `RunOptions`

This is the most copy-paste-friendly shape for a consumer project:

```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MyApp.Models;

using var sessionOptions = new SessionOptions();
using var model = new SampleClassifierModel(sessionOptions);

var image = new DenseTensor<float>(
    new float[1 * 3 * 16 * 16],
    new[] { 1, 3, 16, 16 }
);

using var runOptions = new RunOptions();
runOptions.LogId = "image-classification";

using var outputs = model.Run(
    input: image,
    runOptions: runOptions
);

Tensor<float> prediction = outputs.Output;
Console.WriteLine($"Output tensor length: {prediction.Length}");
```

This example shows the most important runtime lifetimes:

- `SessionOptions` is disposable
- the generated model wrapper is disposable because it owns `InferenceSession`
- `RunOptions` is disposable
- the generated outputs wrapper is disposable because it owns ONNX Runtime output values

## Example: Use The Generated Input Contract

If you prefer an object that mirrors the model signature, use the generated input class:

```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MyApp.Models;

using var model = new SampleClassifierModel();
using var runOptions = new RunOptions();
runOptions.LogId = "typed-inputs";

var inputs = new SampleClassifierModelInputs
{
    Input = new DenseTensor<float>(
        new float[1 * 3 * 16 * 16],
        new[] { 1, 3, 16, 16 }
    ),
};

using var outputs = model.Run(
    inputs: inputs,
    runOptions: runOptions
);

Tensor<float> prediction = outputs.Output;
Console.WriteLine($"Output tensor length: {prediction.Length}");
```

This style becomes especially useful when the ONNX model has multiple inputs or optional inputs.

## Disposal Guidance

Treat these objects as scoped resources:

- `SessionOptions`
- `RunOptions`
- the generated model wrapper, for example `SampleClassifierModel`
- the generated outputs wrapper, for example `SampleClassifierModelOutputs`

Prefer `using var` for all of them.

Also note that `outputs.Output` and `outputs.Raw` are tied to the lifetime of the output wrapper. Read or copy the data you need before leaving the `using` scope.

## Generated API Notes

- The default constructor loads the model from `DefaultModelPath`, resolved relative to the application output folder.
- You can also construct the wrapper from a custom file path or raw model bytes.
- The generated wrapper exposes both `Run(<generated inputs>)` and `Run(..., RunOptions? runOptions)` overloads.
- Optional ONNX inputs become nullable properties on the generated input type and nullable parameters on the generated `Run(...)` overloads.
- The generated wrapper also exposes `Inputs`, `Outputs`, and `OutputNames` metadata for runtime inspection.
- Models that use external tensor data still require their sibling external-data files at deployment time.

## Repository

- Source: <https://github.com/gluschenko/onnxify>
