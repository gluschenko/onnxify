# Onnxify Graph Examples

## Purpose

Use this page when the task is about authoring or editing ONNX graphs directly with `Onnxify`, without going through `Onnxify.TorchSharp`.

This reference focuses on three things:

- building a model from scratch with the core `Onnxify` graph API
- making practical graph edits to an existing model
- preferring generated strongly typed operator wrappers over raw `AddNode(...)` calls whenever the operator surface already exists

## Core Mindset

When working directly with `Onnxify`, think in three layers:

1. `OnnxModel` owns model-level metadata, opset imports, and save/load.
2. `OnnxGraph` owns inputs, outputs, initializers, intermediate values, loose edges, and nodes.
3. typed operator wrappers such as `graph.Conv(...)`, `graph.BatchNormalization(...)`, `graph.Concat(...)`, and `graph.Split(...)` are the preferred way to express known ONNX operators.

Reach for raw `graph.AddNode(...)` only when:

- there is no generated typed wrapper for the operator you need
- you are dealing with an experimental/custom domain that is intentionally outside the typed surface
- you are reproducing an exact pre-existing node shape and the typed surface is not available yet

## Example 1: Create A MobileNet-Like Model From Scratch

The following pattern builds a lightweight depthwise-separable CNN directly with `Onnxify`. It is not meant to be a byte-for-byte clone of the TorchSharp MobileNet example; it is meant to show how to author the same kind of architecture directly as an ONNX graph.

```csharp
using Onnxify;

public static class MobileNetLikeAuthoring
{
    public static OnnxModel CreateMobileNetLike(int numClasses = 10)
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions
        {
            ProducerName = "onnxify-skill",
            IrVersion = 10,
            Opset = 22,
        });

        var graph = model.Graph;
        graph.Name = "mobilenet_v1_like_direct";

        var input = graph.AddInput(
            name: "input",
            type: OnnxTensorType.Create<float>(["batch", 3, 96, 96])
        );

        var x = AddConvBnRelu(graph, "stem", input, inputChannels: 3, outputChannels: 32, stride: 2);
        x = AddDepthwiseSeparableBlock(graph, "block1", x, inputChannels: 32, outputChannels: 64, stride: 1);
        x = AddDepthwiseSeparableBlock(graph, "block2", x, inputChannels: 64, outputChannels: 128, stride: 2);
        x = AddDepthwiseSeparableBlock(graph, "block3", x, inputChannels: 128, outputChannels: 128, stride: 1);

        x = graph.GlobalAveragePool(
            name: "global_pool",
            options: new GlobalAveragePoolInputOptions
            {
                X = x,
            }
        );

        x = graph.Flatten(
            name: "flatten",
            options: new FlattenInputOptions
            {
                Input = x,
                Axis = 1,
            }
        );

        var classifierWeight = graph.AddTensor(
            name: "classifier_w",
            shape: [numClasses, 128],
            value: new float[numClasses * 128]
        );

        var classifierBias = graph.AddTensor(
            name: "classifier_b",
            shape: [numClasses],
            value: new float[numClasses]
        );

        x = graph.Gemm(
            name: "classifier",
            options: new GemmInputOptions
            {
                A = x,
                B = classifierWeight,
                C = classifierBias,
                TransB = 1,
            }
        );

        var output = graph.AddOutput(
            name: "logits",
            type: OnnxTensorType.Create<float>(["batch", numClasses])
        );

        graph.Identity(
            name: "logits_out",
            options: new IdentityInputOutputOptions
            {
                Input = x,
                Output = output,
            }
        );

        return model;
    }

    private static IOnnxGraphEdge AddDepthwiseSeparableBlock(
        OnnxGraph graph,
        string prefix,
        IOnnxGraphEdge input,
        int inputChannels,
        int outputChannels,
        long stride
    )
    {
        var x = AddConvBnRelu(
            graph,
            prefix: $"{prefix}_depthwise",
            input: input,
            inputChannels: inputChannels,
            outputChannels: inputChannels,
            stride: stride,
            kernelSize: 3,
            groups: inputChannels
        );

        return AddConvBnRelu(
            graph,
            prefix: $"{prefix}_pointwise",
            input: x,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            stride: 1,
            kernelSize: 1,
            padding: [0, 0, 0, 0],
            groups: 1
        );
    }

    private static IOnnxGraphEdge AddConvBnRelu(
        OnnxGraph graph,
        string prefix,
        IOnnxGraphEdge input,
        int inputChannels,
        int outputChannels,
        long stride,
        long kernelSize = 3,
        long[]? padding = null,
        long groups = 1
    )
    {
        padding ??= kernelSize == 3 ? [1, 1, 1, 1] : [0, 0, 0, 0];

        var weight = graph.AddTensor(
            name: $"{prefix}_w",
            shape: [outputChannels, inputChannels / groups, kernelSize, kernelSize],
            value: new float[outputChannels * (inputChannels / (int)groups) * (int)kernelSize * (int)kernelSize]
        );

        var bias = graph.AddTensor(
            name: $"{prefix}_b",
            shape: [outputChannels],
            value: new float[outputChannels]
        );

        var conv = graph.Conv(
            name: $"{prefix}_conv",
            options: new ConvInputOptions
            {
                X = input,
                W = weight,
                B = bias,
                Group = groups,
                KernelShape = [kernelSize, kernelSize],
                Pads = padding,
                Strides = [stride, stride],
            }
        );

        var scale = graph.AddTensor($"{prefix}_bn_scale", [outputChannels], Enumerable.Repeat(1f, outputChannels).ToArray());
        var bnBias = graph.AddTensor($"{prefix}_bn_bias", [outputChannels], new float[outputChannels]);
        var mean = graph.AddTensor($"{prefix}_bn_mean", [outputChannels], new float[outputChannels]);
        var variance = graph.AddTensor($"{prefix}_bn_var", [outputChannels], Enumerable.Repeat(1f, outputChannels).ToArray());

        var bn = graph.BatchNormalization(
            name: $"{prefix}_bn",
            options: new BatchNormalizationInputOptions
            {
                X = conv,
                Scale = scale,
                B = bnBias,
                InputMean = mean,
                InputVar = variance,
                Epsilon = 1e-5f,
            }
        ).Y;

        return graph.Relu(
            name: $"{prefix}_relu",
            options: new ReluInputOptions
            {
                X = bn,
            }
        );
    }
}
```

Why this pattern works well:

- graph inputs, outputs, and initializers are all explicit
- every operator is added through a typed wrapper instead of a stringly typed raw node
- repeated architectural motifs are factored into helper methods
- depthwise convolution is just `Conv` with `Group = inputChannels`
- the classification head uses `GlobalAveragePool -> Flatten -> Gemm`, which is easy to inspect and easy to regenerate

If you want stricter MobileNetV1 semantics with ReLU6, use `graph.Clip(...)` after batch normalization instead of `graph.Relu(...)`. The generated operator docs show `Clip` as the ONNX-side primitive that corresponds to ReLU6-style clamping.

## Example 2: Prefer Typed Operators Over Raw `AddNode(...)`

When a typed wrapper exists, this is the preferred style:

```csharp
var y = graph.Conv(
    name: "conv",
    options: new ConvInputOutputOptions
    {
        X = input,
        W = weights,
        B = bias,
        Y = output,
        KernelShape = [3, 3],
        Pads = [1, 1, 1, 1],
        Strides = [1, 1],
        Group = 1,
        AutoPad = "NOTSET",
    }
);
```

The raw-node fallback is more verbose and easier to get wrong:

```csharp
graph.AddNode(
    name: "conv_raw",
    opType: "Conv",
    domain: "",
    docString: "",
    inputs: [input, weights, bias],
    outputs: [output],
    attributes:
    [
        new OnnxAttribute<long[]>("kernel_shape", [3L, 3L]),
        new OnnxAttribute<long[]>("pads", [1L, 1L, 1L, 1L]),
        new OnnxAttribute<long[]>("strides", [1L, 1L]),
        new OnnxAttribute<long>("group", 1L),
        new OnnxAttribute<string>("auto_pad", "NOTSET"),
    ]
);
```

Prefer the typed form because it gives you:

- operator-specific options types such as `ConvInputOptions` and `ConvInputOutputOptions`
- schema-aligned property names such as `KernelShape`, `Pads`, `Strides`, and `Group`
- typed node rehydration after load when `OnnxNode.FromProto(...)` recognizes the node
- easier project generation, as shown in `src/Onnxify.Tests/OnnxProjectGeneratorTests.cs`

## Example 3: Variadic And Multi-Output Typed Operators

`Concat` and `Split` are good examples because they exercise variadic inputs and variadic outputs without falling back to raw node wiring.

```csharp
var left = graph.AddInput("left", OnnxTensorType.Create<float>([1, 16]));
var right = graph.AddInput("right", OnnxTensorType.Create<float>([1, 16]));
var merged = graph.AddEdge("merged");

var concat = new Concat(
    "concat_features",
    new ConcatInputOutputOptions
    {
        In = [left, right],
        Axis = 1,
        ConcatResult = merged,
    }
);

graph.AddNode(concat);

var splitLeft = graph.AddEdge("split_left");
var splitRight = graph.AddEdge("split_right");

var outputs = graph.Split(
    "split_features",
    new SplitInputOutputOptions
    {
        Input = merged,
        Axis = 1,
        Out = [splitLeft, splitRight],
    }
);
```

Why this is useful:

- you keep the operator as a real typed `Concat`/`Split` node instead of a generic `OnnxNode`
- variadic input and output shapes stay explicit
- the test suite already verifies this round-trip pattern in `src/Onnxify.Tests/OnnxModelTests.cs`

## Example 4: Practical Graph Edits On An Existing Model

The current public `OnnxGraph` API is strongest at:

- looking up existing nodes and values
- appending new values, nodes, and outputs
- rebuilding or extending a suffix of the graph

It does not currently expose a rich public "remove node" or "replace node in place" helper surface, so the safest edit pattern is often additive: branch from an existing value, append corrected logic, and publish that branch as a new output.

### Add A Softmax View Of Existing Logits

```csharp
var model = OnnxModel.FromFile("classifier.onnx");
var graph = model.Graph;

var logits = graph.GetValue("logits")
    ?? throw new InvalidOperationException("Expected a graph value named 'logits'.");

var probabilities = graph.Softmax(
    name: graph.NextName("probabilities"),
    options: new SoftmaxInputOptions
    {
        Input = logits,
        Axis = 1,
    }
);

var probabilitiesOutput = graph.AddOutput(
    name: "probabilities",
    type: OnnxTensorType.Create<float>(["batch", "classes"])
);

graph.Identity(
    name: graph.NextName("probabilities_out"),
    options: new IdentityInputOutputOptions
    {
        Input = probabilities,
        Output = probabilitiesOutput,
    }
);
```

This is often the cleanest patch for inspection or serving needs because it leaves the old output path intact and adds a new public output.

### Inspect Existing Structure Before Editing

```csharp
var classifier = graph.GetNode("classifier");
var logits = graph.GetValue("logits");

if (classifier is not null)
{
    classifier.DocString = "Dense classifier head";
}

Console.WriteLine(graph);
```

Useful public helpers:

- `graph.GetNode(name)` to find an existing node by name
- `graph.GetValue(name)` to find any input, output, placeholder, initializer, or loose edge by name
- `graph.NextName(prefix)` when you are patching a graph and need collision-free names

## Example 5: Rebuild A Patched Suffix Instead Of Mutating In Place

When the exact old path is wrong and the graph API does not yet provide the remove/replace helper you want, rebuild the affected suffix from the last trustworthy edge.

```csharp
var logitsInput = graph.GetValue("classifier_input")
    ?? throw new InvalidOperationException();

var patchedLogits = graph.Gemm(
    name: graph.NextName("classifier_patched"),
    options: new GemmInputOptions
    {
        A = logitsInput,
        B = patchedWeight,
        C = patchedBias,
        TransB = 1,
    }
);

var patchedOutput = graph.AddOutput(
    name: "logits_patched",
    type: OnnxTensorType.Create<float>(["batch", "classes"])
);

graph.Identity(
    name: graph.NextName("logits_patched_out"),
    options: new IdentityInputOutputOptions
    {
        Input = patchedLogits,
        Output = patchedOutput,
    }
);
```

This "append a corrected suffix" strategy is often easier to validate than trying to surgically rewrite an already-loaded graph in place.

## Example 6: Use Typed Wrappers First, Raw Nodes As Fallback

A good practical rule is:

- default to `graph.Conv(...)`, `graph.BatchNormalization(...)`, `graph.Add(...)`, `graph.MatMul(...)`, `graph.Split(...)`, and similar typed wrappers
- use `new Concat(...)` plus `graph.AddNode(...)` when you want to construct a typed node instance explicitly before adding it
- use raw `graph.AddNode(...)` only when the operator is genuinely uncovered or custom

That rule gives you the best balance of readability, schema safety, and compatibility with project generation.

## Current Limitations To Remember

- Graph creation and additive editing are strong today.
- Public remove/replace helpers at the graph level are still limited, so many edits are best modeled as "append a fixed branch/suffix".
- When you need a heavy rewrite, it is often clearer to construct a fresh model or regenerate the affected subgraph than to force a deep in-place mutation strategy.

## Best Files To Read First

- `src/Onnxify.ConsoleTest/Program.cs`
  - compact examples of `Conv`, `Relu`, `MaxPool`, `Flatten`, and `Gemm`
- `src/Onnxify.Tests/OnnxModelTests.cs`
  - raw and typed round-trip coverage, including `Concat` and `Split`
- `src/Onnxify.Tests/OnnxProjectGeneratorTests.cs`
  - confirms that typed wrappers are preferred and preserved in generated code
- `references/operators/index.md`
  - use this when you know the ONNX operator name and want the exact wrapper/options surface

## Practical Heuristic

If the operator already exists in `references/operators/index.md`, use the typed wrapper first.

If the graph edit is mostly additive, extend the current graph in place.

If the graph edit is structurally invasive, rebuild the affected suffix or regenerate the model from a cleaner source of truth.
