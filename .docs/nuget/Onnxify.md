# Onnxify

`Onnxify` is for programmatic ONNX model work in .NET: open an existing `.onnx`, inspect the graph, modify it, save it back, or build a model from scratch in C#.

## Install

```bash
dotnet add package Onnxify
```

## ONNX Version Baseline

New models created with `OnnxModel.Create()` use standard ONNX opset 25 and IR version 11 by default. These are the package's baseline ONNX versions: opset 25 comes from the bundled standard-domain operator schemas, and IR version 11 is the repository's current compatibility target for generated models.

Set `OnnxModelCreationOptions.Opset` and `IrVersion` explicitly when you need to target an older runtime or a deployment profile with stricter ONNX support.

## Why This Package Exists

ONNX is often used as a model interchange format, but in .NET there is usually an awkward gap between two extremes:

- an inference runtime can execute a model, but usually does very little to help you rewrite it;
- the protobuf layer lets you do almost anything, but it is too low-level for day-to-day engineering work.

`Onnxify` is meant to fill exactly that space. The package exists for cases where an ONNX model is not just an artifact you "run and forget", but something you need to read, understand, version, patch, and generate programmatically.

In practice, it helps with scenarios like these:

- open an existing ONNX model and quickly understand its inputs, outputs, initializers, and nodes;
- patch a graph in C# without dropping down into raw protobuf manipulation;
- generate ONNX as the output of your own tool, exporter, converter, or build pipeline;
- keep graph structure, value types, shapes, and opset versions in normal .NET code instead of manual protobuf plumbing.

In short, `Onnxify` is for teams that need control, transparency, and editability around ONNX in .NET.

## What The Package Provides

- `OnnxModel.FromFile(...)`, `FromFileAsync(...)`, `FromStream(...)`, `FromStreamAsync(...)`, `Save(...)`, and `SaveAsync(...)` for reading and writing `.onnx` models from files or streams.
- `OnnxModel.Create(...)` for creating a new model from scratch. By default it writes standard ONNX opset 25 and IR version 11.
- `OnnxGraph` for working with inputs, outputs, intermediate values, initializers, loose edges, and nodes.
- Typed value and tensor descriptions through `OnnxValue`, `OnnxTensor<T>`, and `OnnxTensorType`.
- Explicit operator construction through `AddNode(...)` and typed operator wrappers when they are available.
- Direct graph editing through `AddInput(OnnxValue)`, `AddOutput(OnnxValue)`, `RemoveInput(...)`, `RemoveOutput(...)`, `RemoveNode(...)`, `ReplaceNode(...)`, `RemoveValue(...)`, `ReplaceValue(...)`, `RemoveTensor(...)`, and `RemoveEdge(...)`.
- `ValidateCompatibility(...)` for structural compatibility checks.

## Quick Start

- If you want the first copy-paste example to run as-is, start with `Build A Small ONNX Model Manually`.
- If you already have a `.onnx` file and want to inspect it, use `Open A Model And Inspect The Graph`.

## Example: Open A Model And Inspect The Graph

This snippet assumes you already have an ONNX file on disk. Replace `"model.onnx"` with the path to a real model in your project or local workspace.

```csharp
using System.Linq;
using Onnxify;

var model = OnnxModel.FromFile("model.onnx");

Console.WriteLine($"Producer: {model.ProducerName}");
Console.WriteLine($"IR version: {model.IrVersion}");
Console.WriteLine($"Opsets: {string.Join(", ", model.OpsetImport.Select(x => $"{x.Domain}:{x.Version}"))}");
Console.WriteLine($"Inputs: {model.Graph.Inputs.Count}");
Console.WriteLine($"Outputs: {model.Graph.Outputs.Count}");
Console.WriteLine($"Nodes: {model.Graph.Nodes.Count}");

foreach (var input in model.Graph.Inputs)
{
    Console.WriteLine($"Input: {input.Name} -> {input}");
}

var weights = model.Graph.Initializers
    .OfType<OnnxTensor<float>>()
    .FirstOrDefault(x => x.Name == "weights");

if (weights is not null)
{
    Console.WriteLine($"Weights shape: [{string.Join(", ", weights.Shape)}]");
    Console.WriteLine($"Preview: {weights}");
}
```

This is useful when a model comes from somewhere else and the first step is understanding it before changing anything.
Use `await OnnxModel.FromFileAsync("model.onnx")` when model I/O should not block the caller, or `OnnxModel.FromStream(stream)` when the model already lives in memory, a network response, or another stream source.

## Example: Build A Small ONNX Model Manually

```csharp
using Onnxify;

var model = OnnxModel.Create(new OnnxModelCreationOptions
{
    ProducerName = "demo",
    Opset = 25,
    IrVersion = 11,
});

var graph = model.Graph;
graph.Name = "bias_add";

var input = graph.AddInput(
    name: "input",
    type: OnnxTensorType.Create<float>(["batch", 4])
);

var bias = graph.AddTensor(
    name: "bias",
    shape: [4],
    value: [0.1f, 0.2f, 0.3f, 0.4f]
);

var hidden = graph.Add(
    name: "add_bias",
    options: new AddInputOptions
    {
        A = input,
        B = bias,
    }
);

var outputEdge = graph.AddEdge("output");

graph.Identity(
    name: "publish_output",
    options: new IdentityInputOutputOptions
    {
        Input = hidden,
        Output = outputEdge,
    }
);

graph.AddOutput(
    name: "output",
    type: OnnxTensorType.Create<float>(["batch", 4])
);

model.AddMetadataProps("author", "onnxify-demo");
model.Save("bias_add.onnx", overwrite: true);
```

This approach is useful when the ONNX model is generated by your own code instead of being exported from an external framework. It is also the best first smoke test if you want to verify that the package is installed and working in a new project.
Use `await model.SaveAsync("bias_add.onnx", overwrite: true)` when saving from an async workflow, or `model.Save(stream)` / `await model.SaveAsync(stream)` when you need to write to an existing stream.

## Example: Edit An Existing Graph

```csharp
using Onnxify;

var model = OnnxModel.FromFile("classifier.onnx");
var graph = model.Graph;

var logits = graph.GetValue("logits")
    ?? throw new InvalidOperationException("Expected logits.");

var probabilities = graph.Softmax(
    name: graph.NextName("probabilities"),
    options: new SoftmaxInputOptions
    {
        Input = logits,
        Axis = 1,
    }
);

var output = graph.AddValue(
    "probabilities",
    OnnxTensorType.Create<float>(["batch", "classes"])
);

graph.Identity(
    name: graph.NextName("probabilities_out"),
    options: new IdentityInputOutputOptions
    {
        Input = probabilities,
        Output = output,
    }
);

graph.AddOutput(output);
model.Save("classifier_patched.onnx", overwrite: true);
```

For targeted rewrites, use `ReplaceNode(...)` or `ReplaceValue(...)`. For deletion, `RemoveNode(...)`, `RemoveValue(...)`, `RemoveTensor(...)`, and `RemoveEdge(...)` clear matching node input/output references and prune unused loose edges so edited graphs do not retain dangling graph pieces.

## Recommendations

- Use `OnnxModel.FromFile(...)` when you are adapting an existing ONNX model and want to make targeted changes without working directly with protobuf APIs.
- Use `OnnxModel.Create(...)` when the ONNX graph is the end product of your own generator, exporter, or toolchain.
- The default creation profile is standard ONNX opset 25 and IR version 11. Set `Opset` or `IrVersion` explicitly, and use `SetOpsetImport(...)` when needed, if you are working beyond the default `ai.onnx` domain or targeting a specific runtime.
- Run `ValidateCompatibility(...)` before publishing or handing models off to other systems, especially when graphs are generated or modified automatically.
- If the source model uses external tensor data, configure `DataLocation` and `DataReader`. The current `Onnxify` serialization flow writes tensors back as embedded data by default.
- Use the base `Onnxify` package when your main need is direct control over the ONNX graph itself. If the task is more specialized, look at the neighboring packages:
- `Onnxify.TorchSharp` for exporting TorchSharp models into a controllable ONNX graph.
- `Onnxify.ModelGenerator` for generating typed wrappers from an existing `.onnx`.
- `Microsoft.ML.OnnxRuntime` if you only need inference and not model editing.

## Repository

- Source: <https://github.com/gluschenko/onnxify>
