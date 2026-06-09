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
               OnnxifyModelClassName="SampleClassifier" />
  </ItemGroup>
</Project>
```

The packaged `.targets` file forwards `OnnxModel` items to Roslyn as additional files and keeps the model copied to the output directory by default.

`OnnxifyModelImportType` controls which source shape is emitted. Omit it for the default `OnnxRuntimeInference` wrapper, or set a comma-separated value such as `OnnxRuntimeInference,TorchModule` when you also want a graph-shaped TorchSharp module.

With the configuration above, the generator emits types like:

- `MyApp.Models.SampleClassifierModel`
- `MyApp.Models.SampleClassifierModelInputs`
- `MyApp.Models.SampleClassifierModelOutputs`

When `TorchModule` is enabled, the generator also emits `SampleClassifierModelTorchModule`. The MVP TorchSharp backend reconstructs a single-input/single-output float32 ONNX graph from supported primitive operators, stores float32 initializers as registered parameters, stores int64 constants as buffers, and provides `LoadWeightsFromOnnx(string modelPath)` to copy initializer values from a compatible ONNX file.

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
