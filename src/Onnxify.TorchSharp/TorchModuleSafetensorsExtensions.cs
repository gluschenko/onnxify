using System.Globalization;
using Onnxify.Safetensors;
using TorchSharp;
using static TorchSharp.torch;

namespace Onnxify.TorchSharp;

public static class TorchModuleSafetensorsExtensions
{
    public static void SaveStateAsSafetensors(
        this TorchModule module,
        string path,
        IReadOnlyDictionary<string, string>? metadata = null)
    {
        ArgumentNullException.ThrowIfNull(module);
        ArgumentNullException.ThrowIfNull(path);

        var directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var state = EnumerateStateTensors(module)
            .Select(entry => new KeyValuePair<string, TensorView>(entry.Name, CreateTensorView(entry.Tensor)))
            .ToArray();

        if (state.Length == 0)
        {
            throw new InvalidOperationException(
                $"No state tensors were discovered for '{module.GetType().FullName}'. " +
                "This usually means TorchSharp state_dict() did not expose any serializable state.");
        }

        var mergedMetadata = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["format"] = "pt",
            ["module"] = module.GetType().Name,
        };

        if (metadata is not null)
        {
            foreach (var entry in metadata)
            {
                mergedMetadata[entry.Key] = entry.Value;
            }
        }

        global::Onnxify.Safetensors.Safetensors.SerializeToFile(
            data: state,
            metadata: mergedMetadata,
            path: path);
    }

    public static void LoadStateFromSafetensors(
        this TorchModule module,
        string path,
        bool strict = true)
    {
        ArgumentNullException.ThrowIfNull(module);
        ArgumentNullException.ThrowIfNull(path);

        var raw = File.ReadAllBytes(path);
        var safetensors = global::Onnxify.Safetensors.Safetensors.Deserialize(raw);
        var stateByName = EnumerateStateTensors(module)
            .ToDictionary(x => x.Name, x => x.Tensor, StringComparer.Ordinal);

        foreach (var name in safetensors.Names())
        {
            if (!stateByName.TryGetValue(name, out var target))
            {
                if (strict)
                {
                    throw new InvalidOperationException(
                        $"Tensor '{name}' exists in the safetensors file but not in {module.GetType().Name}.");
                }

                continue;
            }

            var source = safetensors.Tensor(name);
            ValidateStateTensorShape(name, target, source);
            CopyTensorData(target, source);
            stateByName.Remove(name);
        }

        if (strict && stateByName.Count > 0)
        {
            throw new InvalidOperationException(
                $"Missing tensors in safetensors file: {string.Join(", ", stateByName.Keys.OrderBy(x => x, StringComparer.Ordinal))}");
        }
    }

    private static IEnumerable<StateTensorEntry> EnumerateStateTensors(TorchModule module)
    {
        foreach (var (name, tensor) in module.state_dict())
        {
            if (tensor is not null && !tensor.IsInvalid)
            {
                yield return new StateTensorEntry(name, tensor);
            }
        }
    }

    private static TensorView CreateTensorView(Tensor tensor)
    {
        using var detached = tensor.detach();
        using var cpuTensor = detached.cpu();
        using var contiguousTensor = cpuTensor.contiguous();

        var dataType = MapTorchDataType(contiguousTensor);
        var shape = contiguousTensor.shape.Select(static x => checked((ulong)x)).ToArray();

        return dataType switch
        {
            DataType.F32 => new TensorView(dataType, shape, ToBytes(contiguousTensor.data<float>().ToArray())),
            _ => throw new NotSupportedException(
                $"Unsupported Torch tensor data type for safetensors export: {contiguousTensor.dtype}."),
        };
    }

    private static void CopyTensorData(Tensor target, TensorView source)
    {
        using var detached = target.detach();
        var targetShape = detached.shape.Select(static x => checked((long)x)).ToArray();

        switch (source.DataType)
        {
            case DataType.F32:
            {
                var values = ToSingleArray(source.Data.Span);
                using var sourceTensor = torch.tensor(values, targetShape, device: detached.device);
                detached.copy_(sourceTensor);
                break;
            }
            default:
                throw new NotSupportedException(
                    $"Unsupported safetensors data type for Torch state import: {source.DataType.ToWireName()}.");
        }
    }

    private static void ValidateStateTensorShape(string name, Tensor target, TensorView source)
    {
        var targetShape = target.shape.Select(static x => checked((ulong)x)).ToArray();
        if (!targetShape.SequenceEqual(source.Shape))
        {
            throw new InvalidOperationException(
                $"Shape mismatch for tensor '{name}'. Model expects [{string.Join(", ", targetShape)}] but file contains [{string.Join(", ", source.Shape)}].");
        }
    }

    private static DataType MapTorchDataType(Tensor tensor)
    {
        var dataTypeName = Convert.ToString(tensor.dtype, CultureInfo.InvariantCulture);
        return dataTypeName switch
        {
            "Float32" => DataType.F32,
            "Float" => DataType.F32,
            _ => throw new NotSupportedException(
                $"Unsupported Torch tensor data type for safetensors export: {dataTypeName}."),
        };
    }

    private static byte[] ToBytes(float[] values)
    {
        var bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static float[] ToSingleArray(ReadOnlySpan<byte> bytes)
    {
        var byteArray = bytes.ToArray();
        var values = new float[byteArray.Length / sizeof(float)];
        Buffer.BlockCopy(byteArray, 0, values, 0, byteArray.Length);
        return values;
    }

    private readonly record struct StateTensorEntry(string Name, Tensor Tensor);
}
