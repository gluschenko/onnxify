# Onnxify.Safetensors

`Onnxify.Safetensors` is the safetensors-focused package in this repository for reading metadata, indexing tensors, and working with safetensors-backed storage from .NET.

## Install

```bash
dotnet add package Onnxify.Safetensors
```

## Why This Package Exists

Modern ML workflows often keep model weights in `.safetensors` because the format is deterministic, self-describing, and easier to validate than ad-hoc binary blobs. The problem on the .NET side is usually not "how do I read bytes from a file", but:

- how to safely parse safetensors headers and validate offsets;
- how to inspect tensor names, dtypes, and shapes without inventing a custom parser;
- how to work with tensor payloads as structured views instead of manual byte slicing;
- how to produce valid `.safetensors` archives from .NET code for model export and tooling scenarios.

`Onnxify.Safetensors` exists to give .NET a dedicated safetensors surface for those tasks. It lets you read metadata, deserialize archives into zero-copy tensor views, slice tensor payloads, and serialize new safetensors files in a format-compatible way.

## What It Provides

- Read safetensors metadata and tensor layout information.
- Work with tensor slices and tensor-backed storage abstractions.
- Support safetensors-related flows that integrate with the broader Onnxify toolchain.

## Code Examples

### Create and Round-Trip a Safetensors File

```csharp
using System.Globalization;
using System.Runtime.InteropServices;
using Onnxify.Safetensors;

var values = new float[] { 1.0f, 2.0f, 3.5f, 4.5f };
var bytes = new byte[values.Length * sizeof(float)];
Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);

var tensor = new TensorView(
    dtype: DataType.F32,
    shape: new ulong[] { 2, 2 },
    data: bytes
);

SafeTensors.SerializeToFile(
    data:
    [
        new KeyValuePair<string, TensorView>("weights", tensor),
    ],
    metadata: new Dictionary<string, string>
    {
        ["framework"] = "onnxify",
        ["purpose"] = "roundtrip-demo",
    },
    path: "weights.safetensors"
);

var archive = SafeTensors.Deserialize(File.ReadAllBytes("weights.safetensors"));
var loaded = archive.Tensor("weights");
var loadedValues = MemoryMarshal.Cast<byte, float>(loaded.Data.Span).ToArray();

Console.WriteLine(string.Join(", ", archive.Names()));
Console.WriteLine($"Shape: [{string.Join(", ", loaded.Shape)}]");
Console.WriteLine(string.Join(", ", loadedValues.Select(x => x.ToString(CultureInfo.InvariantCulture))));
```

### Read Only Metadata Before Touching Tensor Payloads

```csharp
using Onnxify.Safetensors;

var buffer = File.ReadAllBytes("weights.safetensors");
var metadata = SafeTensors.ReadMetadata(buffer).Metadata;

foreach (var name in metadata.OffsetKeys())
{
    var info = metadata.Info(name)!;
    Console.WriteLine($"{name}: {info.DataType} [{string.Join(", ", info.Shape)}]");
}

if (metadata.MetadataEntries is not null)
{
    foreach (var pair in metadata.MetadataEntries)
    {
        Console.WriteLine($"{pair.Key} = {pair.Value}");
    }
}
```

### Slice a Tensor View Without Rebuilding the Whole Archive

```csharp
using Onnxify.Safetensors;

var archive = SafeTensors.Deserialize(File.ReadAllBytes("weights.safetensors"));
var tensor = archive.Tensor("weights");

var rowSlice = tensor.Slice(
    new NarrowTensorIndexer(TensorBounds.Included(0), TensorBounds.Excluded(1)),
    new NarrowTensorIndexer(TensorBounds.Unbounded(), TensorBounds.Unbounded())
);

foreach (var chunk in rowSlice)
{
    Console.WriteLine($"Chunk bytes: {chunk.Length}");
}
```

## Recommendations

- Use `Onnxify.Safetensors` when your source of truth is a `.safetensors` file and you need a low-level, format-aware .NET API instead of a framework-specific wrapper.
- Prefer `SafeTensors.ReadMetadata(...)` when you only need names, shapes, dtypes, or archive metadata. It keeps the workflow explicit and is a good fit for validation, indexing, and diagnostics.
- Prefer `SafeTensors.Deserialize(...)` plus `TensorView` when you need to inspect or slice payload bytes without copying the whole archive into new tensor objects.
- Use `SerializeToFile(...)` when you are generating weights from .NET tooling and want deterministic safetensors output rather than hand-building headers and offsets.
- If your real task is saving or loading TorchSharp module state, reach for the higher-level `Onnxify.TorchSharp` extensions such as `SaveStateAsSafetensors(...)` and `LoadStateFromSafetensors(...)`, which build on top of this package.
- Keep this package as the safetensors boundary and do dtype conversion at the edge of your app. In practice, that usually means storing bytes in `TensorView` and materializing `float[]`, `Half[]`, or other typed values only where they are actually needed.

## Status

This package is still evolving, but it is the dedicated safetensors surface in the repo.

## Repository

- Source: <https://github.com/gluschenko/onnxify>
