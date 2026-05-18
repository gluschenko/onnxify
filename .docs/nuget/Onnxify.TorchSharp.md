# Onnxify.TorchSharp

`Onnxify.TorchSharp` exists for cases where a model is already written in TorchSharp, but the end result needs to be an explicit and controllable ONNX graph in .NET.

## Install

```bash
dotnet add package Onnxify.TorchSharp
```

## Why This Package Exists

`TorchSharp` is a good fit for describing and training models in a PyTorch-like style, while `Onnxify` is a good fit for explicitly building, reading, and editing ONNX models. `Onnxify.TorchSharp` sits between those two worlds.

This package solves several practical problems at once:

- It translates supported `TorchSharp` modules into explicit `Onnxify` operations instead of hiding export behind an opaque black box.
- It moves weights and constants into ONNX initializers so the model can be saved and handled like a normal ONNX model afterward.
- It lets you export a model in pieces and embed TorchSharp blocks into a larger graph that you assemble manually.
- It helps keep weights separate in `safetensors` when that is a better fit for how your project stores and moves model state.

In short, this package is not just for "saving a model to ONNX". It is for building a controllable and extensible bridge between a TorchSharp model and an ONNX graph that you want to inspect, modify, or generate programmatically afterward.

## What It Provides

- `Export(...)` for supported `TorchSharp` modules and sequential containers.
- A set of helpers for tensor-style operations when you want to build an ONNX graph in terms that are close to Torch.
- `SaveStateAsSafetensors(...)` and `LoadStateFromSafetensors(...)` for saving and loading `state_dict()`.
- `SafetensorsExternalDataProvider` for scenarios where ONNX external data should be stored in `safetensors` format.

## Example: Exporting a Sequential Model to ONNX

```csharp
using Onnxify.TorchSharp;
using static TorchSharp.torch.nn;

var features = Sequential(
    ("conv1", Conv2d(3, 8, kernel_size: 3)),
    ("gelu", GELU()),
    ("pool", AvgPool2d(kernel_size: 2, stride: 2)),
    ("flatten", Flatten()),
    ("fc", Linear(392, 10))
);

var model = OnnxModel.Create(new OnnxModelCreationOptions
{
    Opset = 22,
});

var graph = model.Graph;
var input = graph.AddInput(
    name: "input",
    type: OnnxTensorType.Create<float>(["batch", 3, 16, 16])
);

var output = features.Export(graph, input);

var outputEdge = graph.AddEdge("output");

graph.Identity(
    name: "output_identity",
    options: new IdentityInputOutputOptions
    {
        Input = output,
        Output = outputEdge,
    }
);

graph.AddOutput(
    name: "output",
    type: OnnxTensorType.Create<float>(["batch", 10])
);

model.Save("model.onnx", overwrite: true);
```

This is useful when the model architecture lives in TorchSharp, but you want the result to be saved as a normal ONNX model that you can later open, inspect, or refine through `Onnxify`.

## Example: Saving TorchSharp Weights to safetensors

```csharp
using Onnxify.TorchSharp;
using static TorchSharp.torch.nn;

var model = Sequential(
    ("fc1", Linear(128, 64)),
    ("relu", ReLU()),
    ("fc2", Linear(64, 10))
);

model.SaveStateAsSafetensors("model.safetensors");

var restored = Sequential(
    ("fc1", Linear(128, 64)),
    ("relu", ReLU()),
    ("fc2", Linear(64, 10))
);

restored.LoadStateFromSafetensors("model.safetensors");
```

This is useful when the ONNX graph and the weights should be stored separately, when you want to reuse state across experiments, or when `safetensors` is part of your model delivery pipeline.

## How safetensors Save and Load Works

`SaveStateAsSafetensors(...)` and `LoadStateFromSafetensors(...)` operate on the module `state_dict()`.

That means the safetensors file stores the model state that TorchSharp exposes as named tensors:

- trainable weights;
- registered buffers;
- other serializable tensor state that is part of `state_dict()`.

It does not store the full TorchSharp object graph, constructor arguments, or arbitrary custom runtime logic. In practice, the expected workflow is:

1. Recreate the same module shape in code.
2. Load the tensor state from the `.safetensors` file.
3. Continue training, evaluation, export, or inference from that restored module instance.

## Example: Saving with Metadata and Restoring Later

```csharp
using Onnxify.TorchSharp;
using static TorchSharp.torch.nn;

var model = Sequential(
    ("fc1", Linear(128, 64)),
    ("relu", ReLU()),
    ("fc2", Linear(64, 10))
);

model.SaveStateAsSafetensors(
    path: "checkpoints/classifier.safetensors",
    metadata: new Dictionary<string, string>
    {
        ["epoch"] = "12",
        ["dataset"] = "demo",
        ["format_version"] = "1",
    }
);

var restored = Sequential(
    ("fc1", Linear(128, 64)),
    ("relu", ReLU()),
    ("fc2", Linear(64, 10))
);

restored.LoadStateFromSafetensors("checkpoints/classifier.safetensors");
```

During save, tensors are copied through CPU contiguous buffers before they are serialized. This makes the produced file independent of whether the live module currently resides on CPU or GPU.

During load, tensors from the file are matched by name against the target module `state_dict()`. The loader validates shape and dtype compatibility before copying values into the target tensors.

## Strict vs Non-Strict Loading

By default, loading is strict:

```csharp
restored.LoadStateFromSafetensors("model.safetensors", strict: true);
```

With strict loading enabled:

- extra tensors in the file cause an error;
- missing tensors in the target module cause an error;
- shape mismatches cause an error;
- unsupported dtype mappings cause an error.

This is the safer default when you expect the file and the module architecture to match exactly.

If you intentionally want a more permissive restore, you can opt into:

```csharp
restored.LoadStateFromSafetensors("model.safetensors", strict: false);
```

That can be useful during migrations, partial warm starts, or experiments where the target module evolved but you still want to reuse the compatible subset of saved tensors.

## Separating ONNX Graph and Weights

One practical pattern in this repository is:

- export the model structure to `.onnx`;
- save TorchSharp state to `.safetensors`;
- keep architecture and parameter artifacts versioned separately.

This is especially useful when:

- you want to compare several weight checkpoints against the same exported structure;
- you want checkpoint files to participate in a safetensors-based artifact pipeline;
- you want a clean separation between graph definition and learned parameters.

## safetensors for ONNX External Data

`SafetensorsExternalDataProvider` serves a related but different purpose.

`SaveStateAsSafetensors(...)` and `LoadStateFromSafetensors(...)` are about TorchSharp module state. `SafetensorsExternalDataProvider` is about storing an `Onnxify` tensor payload in a safetensors file instead of a raw binary external-data sidecar.

Use that provider when you are already working at the ONNX tensor level and want ONNX external data integration backed by `safetensors` rather than by plain binary blobs.

## Usage Recommendations

- Use `Onnxify.TorchSharp` when your model is already written in `TorchSharp` and you need a controllable ONNX export path from C# without switching to Python.
- Use it when you plan to keep editing the graph after export, add nodes, change inputs or outputs, or assemble a larger composite ONNX model manually.
- Choose it when transparency matters: export is expressed as explicit `Onnxify` operations, and unsupported semantics fail explicitly instead of degrading silently.
- Store weights with `SaveStateAsSafetensors(...)` when you want to separate the graph from the parameters or use `safetensors` as your main artifact format.
- Check coverage for the specific `TorchSharp` modules you need up front. The package already covers a meaningful set of practical layers and tensor operations, but it does not aim to be a complete mirror of the entire TorchSharp API.
- If you do not need the TorchSharp bridge and only need to read, write, or edit ONNX, the base `Onnxify` package is usually enough.
- If your model contains complex dynamic logic, branching, or custom modules, plan for manual export refinement or adding your own `Export(...)` coverage.

## Repository

- Source: <https://github.com/gluschenko/onnxify>
