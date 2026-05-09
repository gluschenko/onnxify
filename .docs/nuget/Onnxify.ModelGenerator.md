# Onnxify.ModelGenerator

`Onnxify.ModelGenerator` is a Roslyn source generator that turns `.onnx` files from your project into typed `Microsoft.ML.OnnxRuntime` wrapper classes.

## Install

```bash
dotnet add package Onnxify.ModelGenerator
```

## What It Provides

- Detect `.onnx` files added to the consuming project as `AdditionalFiles`.
- Generate typed input and output contracts from the model signature.
- Emit a thin `InferenceSession` wrapper with `Run(...)` overloads for `Microsoft.ML.OnnxRuntime`.
- Surface model input and output metadata in generated code for runtime inspection.

## Usage

Mark ONNX files as `AdditionalFiles` in the consuming project:

```xml
<ItemGroup>
  <AdditionalFiles Include="Models\sample-classifier.onnx" />
</ItemGroup>
```

Optional metadata can override the generated namespace and class name:

```xml
<ItemGroup>
  <AdditionalFiles Include="Models\sample-classifier.onnx"
                   OnnxifyModelNamespace="MyApp.Models"
                   OnnxifyModelClassName="SampleClassifier" />
</ItemGroup>
```

## Repository

- Source: <https://github.com/gluschenko/onnxify>
