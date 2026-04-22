# Onnxify.Safetensors

## Purpose

Use this page when the task is about reading, writing, inspecting, slicing, or round-tripping `.safetensors` files with the repository's managed `Onnxify.Safetensors` API, or when a TorchSharp model in this repo needs its state saved to or restored from safetensors.

This reference focuses on two layers:

- the low-level `Onnxify.Safetensors.SafeTensors` API for named tensor archives
- the higher-level TorchSharp integration in `Onnxify.TorchSharp.TorchModuleSafetensorsExtensions`

## Core Entry Points

- `src/Onnxify.Safetensors/SafeTensors.cs`
  - archive-level read, write, metadata parsing, tensor lookup, and formatting
- `src/Onnxify.Safetensors/TensorView.cs`
  - validated tensor payload view with `DataType`, `Shape`, `Data`, and `Slice(...)`
- `src/Onnxify.TorchSharp/TorchModuleSafetensorsExtensions.cs`
  - `SaveStateAsSafetensors(...)` and `LoadStateFromSafetensors(...)` for TorchSharp modules
- `src/Onnxify.ConsoleTest/Program.cs`
  - compact low-level safetensors round-trip example
- `src/Onnxify.Examples/Program.cs`
  - end-to-end examples that save model weights as `.safetensors`

## Core Mindset

When working with `Onnxify.Safetensors`, separate these concerns:

1. `SafeTensors` owns archive structure, metadata validation, tensor name lookup, and serialization.
2. `TensorView` owns one tensor's logical shape, data type, and raw bytes.
3. TorchSharp integration owns converting `state_dict()` tensors to and from `TensorView`; do not duplicate that conversion logic in unrelated callers.

Prefer the low-level API when:

- you are reading or writing a `.safetensors` file outside TorchSharp
- you need to inspect names, metadata, or individual tensors
- you want zero-copy views over an already loaded buffer

Prefer the TorchSharp extension methods when:

- the task is about saving or restoring a TorchSharp model's weights
- the examples already use `SaveStateAsSafetensors(...)` or `LoadStateFromSafetensors(...)`
- the caller should work in terms of model state rather than raw named tensor archives

## Example 1: Low-Level Round Trip With `SafeTensors`

`src/Onnxify.ConsoleTest/Program.cs` shows the smallest repository-owned round-trip:

```csharp
using Onnxify.Safetensors;

var values = new float[] { 1.0f, 2.0f, 3.5f, 4.5f };
var data = values
    .SelectMany(BitConverter.GetBytes)
    .ToArray();

var tensor = new TensorView(
    dtype: DataType.F32,
    shape: [2, 2],
    data: data);

SafeTensors.SerializeToFile(
    data: [new KeyValuePair<string, TensorView>("weights", tensor)],
    metadata: new Dictionary<string, string>
    {
        ["framework"] = "onnxify-console-test",
        ["purpose"] = "roundtrip-demo",
    },
    path: outputPath);

var raw = File.ReadAllBytes(outputPath);
var safetensors = SafeTensors.Deserialize(raw);
var loadedTensor = safetensors.Tensor("weights");

var loadedValues = loadedTensor.Data.ToArray()
    .Chunk(sizeof(float))
    .Select(chunk => BitConverter.ToSingle(chunk))
    .ToArray();
```

What this pattern is good for:

- writing a named archive from scratch
- attaching top-level `__metadata__` entries
- loading the archive back and resolving a tensor by name
- validating a small deterministic round trip in tests or playground code

## Example 2: Inspect A `.safetensors` Archive

For programmatic inspection, the usual flow is:

```csharp
using Onnxify.Safetensors;

var raw = File.ReadAllBytes(path);
var safetensors = SafeTensors.Deserialize(raw);

Console.WriteLine($"Tensor names: {string.Join(", ", safetensors.Names())}");
Console.WriteLine($"Tensor count: {safetensors.Length}");

foreach (var name in safetensors.Names())
{
    var tensor = safetensors.Tensor(name);
    Console.WriteLine($"{name}: dtype={tensor.DataType.ToWireName()} shape=[{string.Join(", ", tensor.Shape)}]");
}

var metadata = safetensors.Metadata.MetadataEntries;
if (metadata is not null)
{
    foreach (var pair in metadata)
    {
        Console.WriteLine($"{pair.Key}={pair.Value}");
    }
}
```

Useful helpers:

- `SafeTensors.Deserialize(raw)` for a validated archive view over file bytes
- `safetensors.Names()` to list tensors in metadata order
- `safetensors.Tensor(name)` to resolve one tensor
- `safetensors.Tensors()` or `safetensors.Iter()` when you want named enumeration
- `safetensors.Metadata.MetadataEntries` for top-level `__metadata__`
- `safetensors.ToString()` for a formatted preview of archive contents

If the task is inspection from the terminal only, use `references/cli.md` instead.

## Example 3: Save TorchSharp Weights As `.safetensors`

`src/Onnxify.Examples/Program.cs` uses the TorchSharp extension method repeatedly:

```csharp
var model = new TorchSharpExportShowcase();
model.eval();

var weightOutputPath = Path.Combine(outputDirectory, "torchsharp-export-showcase.safetensors");
model.SaveStateAsSafetensors(weightOutputPath);
```

The same pattern is used for:

- `mini-gpt2-like.safetensors`
- `lang-lstm.safetensors`
- `tiny-yolo-like.safetensors`
- `mobilenet-v1-like.safetensors`
- `alexnet.safetensors`

Why this is the preferred repo pattern:

- it enumerates `state_dict()` instead of making each caller rebuild tensor-name mapping manually
- it converts TorchSharp dtypes into safetensors-compatible `DataType` values in one place
- it automatically writes baseline metadata such as `format=pt` and `module=<type>`
- it keeps repo examples focused on model behavior rather than byte packing

When the task is "save my TorchSharp model weights", start with `SaveStateAsSafetensors(...)` instead of direct `SafeTensors.SerializeToFile(...)`.

## Example 4: Restore TorchSharp Weights From `.safetensors`

`src/Onnxify.Examples/Program.cs` also shows load-before-train / resume-style usage:

```csharp
var weightOutputPath = Path.Combine(outputDirectory, "alexnet.safetensors");
var model = new AlexNet("alexnet", trainDataset.LabelNames.Count, device);

if (File.Exists(weightOutputPath))
{
    model.LoadStateFromSafetensors(weightOutputPath);
}
```

What `LoadStateFromSafetensors(...)` does for you:

- reads the file
- validates that each named tensor exists when `strict: true`
- checks shape compatibility
- checks data type compatibility
- copies data into the target TorchSharp tensors

Use `strict: false` only when the task explicitly wants a partial or forward-compatible load. Otherwise prefer the default strict mode so missing or unexpected weights fail clearly.

## Example 5: Work With Raw Tensor Payloads

At the low level, `TensorView` is the core building block:

```csharp
var tensor = safetensors.Tensor("weights");

Console.WriteLine(tensor.DataType.ToWireName());
Console.WriteLine($"[{string.Join(", ", tensor.Shape)}]");

var bytes = tensor.Data.Span;
var values = MemoryMarshal.Cast<byte, float>(bytes).ToArray();
```

Important rules:

- `TensorView` validates that `shape x dtype size == data length`
- `TensorView.Data` is raw bytes, so callers must interpret it according to `DataType`
- prefer `TensorView` over ad hoc tuples because it preserves repository validation rules

If the task needs a logical sub-range rather than the whole tensor, inspect `TensorView.Slice(...)`, `TensorIndexer`, and `SliceIterator`.

## Metadata And Validation Notes

- `SafeTensors.ReadMetadata(buffer)` parses and validates just the header information plus header length.
- `SafeTensors.Deserialize(buffer)` validates the whole archive view and exposes the payload section.
- `Metadata` enforces contiguous tensor offsets and byte-size agreement for every tensor entry.
- `TensorView` enforces that the raw byte length exactly matches the declared shape and dtype.
- invalid archive structure throws `SafeTensorException`; invalid slicing throws `InvalidSliceException`.

This means you should usually trust `SafeTensors.Deserialize(...)` as the first validation boundary instead of rechecking offsets and byte counts in every caller.

## Common Task Mapping

- Need to create a safetensors archive from scratch: start with `SafeTensors.Serialize(...)` or `SafeTensors.SerializeToFile(...)`.
- Need to inspect tensor names, shapes, or metadata in a `.safetensors` file: start with `SafeTensors.Deserialize(...)`.
- Need to save TorchSharp model weights: start with `TorchModuleSafetensorsExtensions.SaveStateAsSafetensors(...)`.
- Need to restore TorchSharp model weights: start with `TorchModuleSafetensorsExtensions.LoadStateFromSafetensors(...)`.
- Need examples that actually run in this repo: inspect `src/Onnxify.ConsoleTest/Program.cs` and `src/Onnxify.Examples/Program.cs`.
- Need terminal-only inspection instead of C# code: read `references/cli.md`.

## Practical Heuristic

If the task is about model state, use the TorchSharp safetensors extensions first.

If the task is about raw archive structure, metadata, or named tensor payloads, use `Onnxify.Safetensors.SafeTensors` first.
