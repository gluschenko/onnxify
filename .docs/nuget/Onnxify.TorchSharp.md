# Onnxify.TorchSharp

`Onnxify.TorchSharp` exists for cases where a model is already written in TorchSharp, but the end result needs to be an explicit and controllable ONNX graph in .NET.

## Install

```bash
dotnet add package Onnxify.TorchSharp
dotnet add package TorchSharp-cpu
```

`Onnxify.TorchSharp` gives you the export and safetensors integration layer, but a TorchSharp runtime package is typically still needed to instantiate and run TorchSharp modules in a real application.

For local CPU execution, `TorchSharp-cpu` is the simplest starting point. For GPU execution, install the appropriate TorchSharp CUDA runtime package for your environment instead.

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
- Experimental `ExportOnnxModel(...)` for deep-exporting supported single-input `Module<Tensor, Tensor>` models directly from their decompiled `forward(Tensor)` method.
- A set of helpers for tensor-style operations when you want to build an ONNX graph in terms that are close to Torch.
- `SaveStateAsSafetensors(...)` and `LoadStateFromSafetensors(...)` for saving and loading `state_dict()`.
- `SafetensorsExternalDataProvider` for scenarios where ONNX external data should be stored in `safetensors` format.

Naming note: `Export(...)` is still the low-level module/operator export API used while you are manually building a graph, for example `_features.Export(graph, input)`. The experimental whole-model API is named `ExportOnnxModel(...)` because it tries to export the model by decompiling `forward(Tensor)` instead of following a hand-written export method.

## Example: A Realistic TorchSharp Model Class

```csharp
using System.Collections.Generic;
using Onnxify;
using Onnxify.TorchSharp;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public sealed class MyModel : torch.nn.Module<Tensor, Tensor>
{
    private readonly torch.nn.Module<Tensor, Tensor> _features;
    private readonly torch.nn.Module<Tensor, Tensor> _classifier;

    public MyModel(string name = "my_model")
        : base(name)
    {
        _features = Sequential(
            ("conv1", Conv2d(3, 8, kernel_size: 3)),
            ("gelu", GELU()),
            ("pool", AvgPool2d(kernel_size: 2, stride: 2)),
            ("flatten", Flatten())
        );

        _classifier = Sequential(
            ("fc1", Linear(392, 64)),
            ("relu", ReLU()),
            ("fc2", Linear(64, 10))
        );

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = _features.forward(input);
        x = _classifier.forward(x);
        return x;
    }

    public OnnxModel Export()
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions
        {
            Opset = 22,
        });

        var graph = model.Graph;
        var input = graph.AddInput(
            name: "input",
            type: OnnxTensorType.Create<float>(["batch", 3, 16, 16])
        );

        var x = _features.Export(graph, input);
        x = _classifier.Export(graph, x);

        var outputEdge = graph.AddEdge("output");
        graph.Identity(
            name: "output_identity",
            options: new IdentityInputOutputOptions
            {
                Input = x,
                Output = outputEdge,
            }
        );

        graph.AddOutput(
            name: "output",
            type: OnnxTensorType.Create<float>(["batch", 10])
        );

        return model;
    }

    public void SaveCheckpoint(
        string path,
        IReadOnlyDictionary<string, string>? metadata = null
    )
    {
        this.SaveStateAsSafetensors(path, metadata);
    }

    public void LoadCheckpoint(
        string path,
        bool strict = true
    )
    {
        this.LoadStateFromSafetensors(path, strict);
    }
}
```

This shape is closer to how consumers usually structure a real application: the architecture lives in a reusable TorchSharp model class, while ONNX export and checkpoint persistence are exposed as explicit model methods.

The sample above assumes a TorchSharp runtime package such as `TorchSharp-cpu` is installed. Without a runtime package, the project may compile but fail at runtime when TorchSharp tries to create modules.

## Example: Exporting the Model to ONNX

```csharp
var model = new MyModel();
var onnxModel = model.Export();
onnxModel.Save("model.onnx", overwrite: true);
```

This example uses a regular model-owned `Export()` method that manually constructs the ONNX graph. It keeps the export path attached to the same class that defines the TorchSharp architecture, which makes the code easier to discover and reuse. The experimental `ExportOnnxModel(...)` API is shown separately below.

## Experimental: Deep Export from forward(Tensor)

> Spoiler: `ExportOnnxModel(...)` is experimental, but it is already useful for trying real TorchSharp architectures without writing a manual `Export()` method, including transformer-style and convolution-style `forward(Tensor)` bodies. It is not a complete C# or TorchSharp compiler.

`ExportOnnxModel(...)` decompiles a model's `forward(Tensor)` method, walks the supported syntax tree, and lowers the recognized data flow into an `OnnxModel`. It can handle supported module calls, recursively deep-export some user-defined child modules, and lower a focused set of tensor operations and statically resolvable branches.

```csharp
using Onnxify;
using Onnxify.TorchSharp;
using TorchSharp;
using static TorchSharp.torch;

var model = new MyModel();
model.eval();

var onnxModel = model.ExportOnnxModel(
    input: OnnxTensorType.Create<float>(["batch", 3, 16, 16]),
    output: OnnxTensorType.Create<float>(["batch", 10]),
    options: new OnnxModelCreationOptions
    {
        Opset = 22,
        ProducerName = "my-app",
    }
);

onnxModel.Save("model-deep-export.onnx", overwrite: true);
```

This path is a good fit for quick smoke exports and for models whose `forward` is mostly a composition of supported TorchSharp modules and tensor operations:

```csharp
public override Tensor forward(Tensor input)
{
    var x = _features.forward(input);
    x = _classifier.forward(x);
    return x;
}
```

It is also able to lower more involved model code when the control flow is statically understandable. A transformer-like model can use validation helpers, shape reads, helper methods that return tensors, disposable temporaries, tied weights, and recursive user-defined child modules:

```csharp
public override Tensor forward(Tensor tokens)
{
    ValidateInputShape(tokens);

    using var positions = CreatePositionIds(tokens.shape[0], tokens.device);

    var x = _tokenEmbedding.forward(tokens) + _positionEmbedding.forward(positions);
    x = _block.forward(x);
    x = _outputNorm.forward(x);

    return ComputeLogits(x);
}

private Tensor ComputeLogits(Tensor hiddenStates)
{
    using var tiedWeight = _tokenEmbedding.weight!.transpose(0, 1);
    return torch.matmul(hiddenStates, tiedWeight);
}

private Tensor CreatePositionIds(long batchSize, Device device)
{
    return torch.arange(_maxSequenceLength, dtype: ScalarType.Int64, device: device)
        .unsqueeze(0)
        .expand(batchSize, _maxSequenceLength);
}

private static void ValidateInputShape(Tensor tokens)
{
    if (tokens.shape.Length != 2)
    {
        throw new ArgumentException("Expected token ids with rank 2.", nameof(tokens));
    }
}
```

A recursively exported attention child can then use tensor indexing, view/permute chains, scalar tensor math, generated masks, slicing, `matmul`, and `softmax`:

```csharp
public override Tensor forward(Tensor x)
{
    var batchSize = x.shape[0];
    var sequenceLength = x.shape[1];

    var qkv = _attention.forward(x)
        .view([batchSize, sequenceLength, 3, _headCount, _headDimension])
        .permute(2, 0, 3, 1, 4);

    var query = qkv[0];
    var key = qkv[1];
    var value = qkv[2];

    using var causalMask =
        torch.triu(
            torch.full(
                [_maxSequenceLength, _maxSequenceLength],
                -10_000f,
                dtype: x.dtype,
                device: x.device
            ),
            diagonal: 1
        )
        .slice(0, 0, sequenceLength, 1)
        .slice(1, 0, sequenceLength, 1)
        .unsqueeze(0)
        .unsqueeze(0);

    var scores = torch.matmul(query, key.transpose(2, 3)) * _scale;
    var weights = torch.softmax(scores + causalMask, dim: -1);
    var context = torch.matmul(weights, value)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view([batchSize, sequenceLength, _attentionDimensions]);

    return _projection.forward(context);
}
```

Convolutional models can also use recursively exported user-defined child modules, inline tensor arrays, `torch.cat(...)`, scalar residual scaling, nearest-neighbor interpolation helpers, and configuration-time conditional expressions such as:

```csharp
var x = _pixelUnshuffle is null
    ? input
    : _pixelUnshuffle.forward(input);

var x2 = _activation.forward(_conv2.forward(torch.cat(new[] { input, x1 }, 1)));
return (x5 * ResidualScale) + input;
```

It can also fold configuration-time choices when the condition can be evaluated from the already-created module instance:

```csharp
public override Tensor forward(Tensor input)
{
    var x = _optionalProjection == null
        ? input
        : _optionalProjection.forward(input);

    return _head.forward(x);
}
```

It is not intended to silently guess semantics for arbitrary runtime control flow. If a branch, loop, operator, tensor method, or helper call cannot be represented safely, the exporter throws `NotSupportedException` so that the model can be adjusted or exported manually.

## Example: Saving and Loading a safetensors Checkpoint

```csharp
var model = new MyModel();
model.SaveCheckpoint("model.safetensors");

var restored = new MyModel();
restored.LoadCheckpoint("model.safetensors");
```

This is useful when the ONNX graph and the weights should be stored separately, when you want to reuse state across experiments, or when `safetensors` is part of your model delivery pipeline.

This example also requires a TorchSharp runtime package because creating `Conv2d(...)`, `Linear(...)`, `ReLU()`, and other TorchSharp modules loads the underlying TorchSharp native backend.

## How safetensors Save and Load Works

In the class pattern above, `SaveCheckpoint(...)` and `LoadCheckpoint(...)` are thin wrappers over `SaveStateAsSafetensors(...)` and `LoadStateFromSafetensors(...)`, and those APIs operate on the module `state_dict()`.

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
var model = new MyModel();
model.SaveCheckpoint(
    path: "checkpoints/classifier.safetensors",
    metadata: new Dictionary<string, string>
    {
        ["epoch"] = "12",
        ["dataset"] = "demo",
        ["format_version"] = "1",
    }
);

var restored = new MyModel();
restored.LoadCheckpoint("checkpoints/classifier.safetensors");
```

During save, tensors are copied through CPU contiguous buffers before they are serialized. This makes the produced file independent of whether the live module currently resides on CPU or GPU.

During load, tensors from the file are matched by name against the target module `state_dict()`. The loader validates shape and dtype compatibility before copying values into the target tensors.

## Ad-Hoc safetensors Values

For smaller named values that are not part of a TorchSharp `state_dict()`, use the generic `SafeTensors` API from `Onnxify.Safetensors`. The generic methods handle the byte marshaling for supported CLR element types, so callers do not need to build `TensorView` instances by hand.

```csharp
using Onnxify.Safetensors;

var archive = new SafeTensors();

archive.Set("scores", [1.2f, 3.4f, 5.6f]);
archive.Set("thresholds", [1.2, 3.4, 5.6]);
archive.Set("labels", ["cat", "dog", "bird"]);
archive.Set("matrix", [1, 2, 3, 4], 2, 2);

File.WriteAllBytes("values.safetensors", archive.Serialize());

var loaded = SafeTensors.Deserialize(File.ReadAllBytes("values.safetensors"));
var scores = loaded.Get<float>("scores");
var labels = loaded.Get<string>("labels");

loaded.Remove("thresholds");
```

Supported `T` values for `Set<T>(...)` and `Get<T>(...)` are `bool`, `byte`, `sbyte`, `short`, `ushort`, `Half`, `int`, `uint`, `float`, `double`, `long`, `ulong`, and `string`.

## Strict vs Non-Strict Loading

By default, loading is strict:

```csharp
restored.LoadCheckpoint("model.safetensors", strict: true);
```

With strict loading enabled:

- extra tensors in the file cause an error;
- missing tensors in the target module cause an error;
- shape mismatches cause an error;
- unsupported dtype mappings cause an error.

This is the safer default when you expect the file and the module architecture to match exactly.

If you intentionally want a more permissive restore, you can opt into:

```csharp
restored.LoadCheckpoint("model.safetensors", strict: false);
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
- If your model contains complex dynamic logic, runtime branching, or unsupported custom modules, prefer a manual model `Export()` method or add the missing module/operator `Export(...)` coverage instead of relying on experimental deep export alone.

## Repository

- Source: <https://github.com/gluschenko/onnxify>
