using System.Runtime.InteropServices;
using Onnxify.Safetensors;
using TorchSharp;
using static TorchSharp.torch;

namespace Onnxify.TorchSharp;

/// <summary>
/// Provides safetensors save/load helpers for TorchSharp module state dictionaries.
/// </summary>
/// <remarks>
/// These helpers are intended for model weights and buffers exposed by <c>state_dict()</c>. They copy tensors through CPU contiguous buffers so the resulting files are independent of the device that owns the live TorchSharp module.
/// </remarks>
public static class TorchModuleSafetensorsExtensions
{
    /// <summary>
    /// Saves all serializable tensors from a TorchSharp module state dictionary to a safetensors file.
    /// </summary>
    /// <param name="module">Module whose <c>state_dict()</c> should be exported.</param>
    /// <param name="path">Destination safetensors path.</param>
    /// <param name="metadata">Optional metadata entries to merge with the default format and module metadata.</param>
    /// <exception cref="InvalidOperationException">Thrown when the module exposes no serializable state tensors.</exception>
    /// <exception cref="NotSupportedException">Thrown when a state tensor uses a Torch dtype that cannot be represented by the current safetensors mapping.</exception>
    public static void SaveStateAsSafetensors(
        this TorchModule module,
        string path,
        IReadOnlyDictionary<string, string>? metadata = null
    )
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

        global::Onnxify.Safetensors.SafeTensors.SerializeToFile(
            data: state,
            metadata: mergedMetadata,
            path: path
        );
    }

    /// <summary>
    /// Loads tensors from a safetensors file into matching entries of a TorchSharp module state dictionary.
    /// </summary>
    /// <param name="module">Module whose existing state tensors should receive the loaded values.</param>
    /// <param name="path">Source safetensors path.</param>
    /// <param name="strict">When <see langword="true"/>, fail on extra file tensors or missing module tensors.</param>
    /// <exception cref="InvalidOperationException">Thrown when strict matching fails or a tensor shape or dtype does not match the target state tensor.</exception>
    /// <exception cref="NotSupportedException">Thrown when a target tensor dtype cannot be loaded by the current mapping.</exception>
    public static void LoadStateFromSafetensors(
        this TorchModule module,
        string path,
        bool strict = true
    )
    {
        ArgumentNullException.ThrowIfNull(module);
        ArgumentNullException.ThrowIfNull(path);

        var raw = File.ReadAllBytes(path);
        var safetensors = global::Onnxify.Safetensors.SafeTensors.Deserialize(raw);
        var stateByName = EnumerateStateTensors(module)
            .ToDictionary(x => x.Name, x => x.Tensor, StringComparer.Ordinal);

        foreach (var name in safetensors.Names())
        {
            if (!stateByName.TryGetValue(name, out var target))
            {
                if (strict)
                {
                    throw new InvalidOperationException(
                        $"Tensor '{name}' exists in the safetensors file but not in {module.GetType().Name}."
                    );
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
                $"Missing tensors in safetensors file: {string.Join(", ", stateByName.Keys.OrderBy(x => x, StringComparer.Ordinal))}"
            );
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

    internal static TensorView CreateTensorView(Tensor tensor)
    {
        using var detached = tensor.detach();
        using var cpuTensor = detached.cpu();
        using var contiguousTensor = cpuTensor.contiguous();

        var dataType = MapTorchDataType(contiguousTensor);
        var shape = GetSafeTensorShape(contiguousTensor);
        var data = GetTensorBytes(contiguousTensor);

        return new TensorView(dataType, shape, data);
    }

    internal static void CopyTensorData(Tensor target, TensorView source)
    {
        using var detached = target.detach();
        ValidateStateTensorType(detached, source);

        var targetShape = detached.shape.Select(static x => checked((long)x)).ToArray();

        switch (detached.dtype)
        {
            case ScalarType.Byte:
            {
                var values = source.Data.ToArray();
                using var sourceTensor = torch.tensor(values, targetShape, dtype: ScalarType.Byte, device: detached.device);
                detached.copy_(sourceTensor);
                break;
            }
            case ScalarType.Int8:
            {
                var values = ToArray<sbyte>(source.Data.Span);
                using var sourceTensor = torch.tensor(values, targetShape, dtype: ScalarType.Int8, device: detached.device);
                detached.copy_(sourceTensor);
                break;
            }
            case ScalarType.Int16:
            {
                var values = ToArray<short>(source.Data.Span);
                using var sourceTensor = torch.tensor(values, targetShape, dtype: ScalarType.Int16, device: detached.device);
                detached.copy_(sourceTensor);
                break;
            }
            case ScalarType.Int32:
            {
                var values = ToArray<int>(source.Data.Span);
                using var sourceTensor = torch.tensor(values, targetShape, dtype: ScalarType.Int32, device: detached.device);
                detached.copy_(sourceTensor);
                break;
            }
            case ScalarType.Int64:
            {
                var values = ToArray<long>(source.Data.Span);
                using var sourceTensor = torch.tensor(values, targetShape, dtype: ScalarType.Int64, device: detached.device);
                detached.copy_(sourceTensor);
                break;
            }
            case ScalarType.Float16:
            {
                var values = ToSingleArrayFromHalf(source.Data.Span);
                using var sourceTensor = torch.tensor(values, targetShape, dtype: ScalarType.Float32, device: detached.device);
                detached.copy_(sourceTensor);
                break;
            }
            case ScalarType.Float32:
            {
                var values = ToArray<float>(source.Data.Span);
                using var sourceTensor = torch.tensor(values, targetShape, dtype: ScalarType.Float32, device: detached.device);
                detached.copy_(sourceTensor);
                break;
            }
            case ScalarType.Float64:
            {
                var values = ToArray<double>(source.Data.Span);
                using var sourceTensor = torch.tensor(values, targetShape, dtype: ScalarType.Float64, device: detached.device);
                detached.copy_(sourceTensor);
                break;
            }
            case ScalarType.ComplexFloat32:
            {
                var values = ToArray<float>(source.Data.Span);
                var complexShape = AppendComplexComponentDimension(targetShape);
                using var sourceTensor = torch.tensor(values, complexShape, dtype: ScalarType.Float32, device: detached.device);
                using var complexTensor = sourceTensor.view_as_complex();
                detached.copy_(complexTensor);
                break;
            }
            case ScalarType.ComplexFloat64:
            {
                var values = ToArray<double>(source.Data.Span);
                var complexShape = AppendComplexComponentDimension(targetShape);
                using var sourceTensor = torch.tensor(values, complexShape, dtype: ScalarType.Float64, device: detached.device);
                using var complexTensor = sourceTensor.view_as_complex();
                detached.copy_(complexTensor);
                break;
            }
            case ScalarType.Bool:
            {
                var values = ToBooleanArray(source.Data.Span);
                using var sourceTensor = torch.tensor(values, targetShape, dtype: ScalarType.Bool, device: detached.device);
                detached.copy_(sourceTensor);
                break;
            }
            case ScalarType.BFloat16:
            {
                var values = ToSingleArrayFromBFloat16(source.Data.Span);
                using var sourceTensor = torch.tensor(values, targetShape, dtype: ScalarType.Float32, device: detached.device);
                detached.copy_(sourceTensor);
                break;
            }
            default:
                throw new NotSupportedException(
                    $"Unsupported Torch tensor data type for safetensors import: {detached.dtype}."
                );
        }
    }

    private static void ValidateStateTensorShape(string name, Tensor target, TensorView source)
    {
        var targetShape = GetSafeTensorShape(target);
        if (!targetShape.SequenceEqual(source.Shape))
        {
            throw new InvalidOperationException(
                $"Shape mismatch for tensor '{name}'. Model expects [{string.Join(", ", targetShape)}] but file contains [{string.Join(", ", source.Shape)}]."
            );
        }
    }

    private static DataType MapTorchDataType(Tensor tensor)
    {
        return tensor.dtype switch
        {
            ScalarType.Bool => DataType.Bool,
            ScalarType.Byte => DataType.U8,
            ScalarType.Int8 => DataType.I8,
            ScalarType.Int16 => DataType.I16,
            ScalarType.Float16 => DataType.F16,
            ScalarType.BFloat16 => DataType.Bf16,
            ScalarType.Int32 => DataType.I32,
            ScalarType.Float32 => DataType.F32,
            ScalarType.ComplexFloat32 => DataType.C64,
            ScalarType.Float64 => DataType.F64,
            ScalarType.Int64 => DataType.I64,
            // Safetensors has no native C128 dtype, so complex128 tensors are
            // stored as F64 values with an extra trailing [2] dimension.
            ScalarType.ComplexFloat64 => DataType.F64,
            _ => throw new NotSupportedException(
                $"Unsupported Torch tensor data type for safetensors export: {tensor.dtype}."),
        };
    }

    private static void ValidateStateTensorType(Tensor target, TensorView source)
    {
        var expectedType = MapTorchDataType(target);
        if (source.DataType != expectedType)
        {
            throw new InvalidOperationException(
                $"Tensor data type mismatch. Model expects {target.dtype} / {expectedType.ToWireName()} but file contains {source.DataType.ToWireName()}.");
        }
    }

    private static ulong[] GetSafeTensorShape(Tensor tensor)
    {
        var shape = tensor.shape.Select(static x => checked((ulong)x)).ToArray();
        if (tensor.dtype != ScalarType.ComplexFloat64)
        {
            return shape;
        }

        var complexShape = new ulong[shape.Length + 1];
        Array.Copy(shape, complexShape, shape.Length);
        complexShape[^1] = 2;
        return complexShape;
    }

    private static byte[] GetTensorBytes(Tensor tensor)
    {
        return tensor.dtype switch
        {
            ScalarType.Bool => ToBooleanBytes(tensor.data<bool>().ToArray()),
            ScalarType.Byte => tensor.data<byte>().ToArray(),
            ScalarType.Int8 => ToBytes(tensor.data<sbyte>().ToArray()),
            ScalarType.Int16 => ToBytes(tensor.data<short>().ToArray()),
            ScalarType.Int32 => ToBytes(tensor.data<int>().ToArray()),
            ScalarType.Int64 => ToBytes(tensor.data<long>().ToArray()),
            ScalarType.Float16 => ToHalfBytes(tensor.@float().data<float>().ToArray()),
            ScalarType.BFloat16 => ToBFloat16Bytes(tensor.@float().data<float>().ToArray()),
            ScalarType.Float32 => ToBytes(tensor.data<float>().ToArray()),
            ScalarType.ComplexFloat32 => ToComplex64Bytes(tensor),
            ScalarType.Float64 => ToBytes(tensor.data<double>().ToArray()),
            ScalarType.ComplexFloat64 => ToComplex128Bytes(tensor),
            _ => throw new NotSupportedException(
                $"Unsupported Torch tensor data type for safetensors export: {tensor.dtype}."
            ),
        };
    }

    private static long[] AppendComplexComponentDimension(long[] shape)
    {
        var complexShape = new long[shape.Length + 1];
        Array.Copy(shape, complexShape, shape.Length);
        complexShape[^1] = 2;
        return complexShape;
    }

    private static byte[] ToComplex64Bytes(Tensor tensor)
    {
        using var realTensor = tensor.view_as_real().contiguous();
        return ToBytes(realTensor.data<float>().ToArray());
    }

    private static byte[] ToComplex128Bytes(Tensor tensor)
    {
        using var realTensor = tensor.view_as_real().contiguous();
        return ToBytes(realTensor.data<double>().ToArray());
    }

    private static byte[] ToBooleanBytes(bool[] values)
        => [.. values.Select(static value => value ? (byte)1 : (byte)0)];

    private static bool[] ToBooleanArray(ReadOnlySpan<byte> bytes)
        => [.. bytes.ToArray().Select(static value => value != 0)];

    private static byte[] ToHalfBytes(float[] values)
    {
        var bytes = new byte[values.Length * sizeof(ushort)];
        for (var i = 0; i < values.Length; i++)
        {
            var bits = BitConverter.HalfToUInt16Bits((Half)values[i]);
            bytes[i * 2] = (byte)(bits & 0xFF);
            bytes[i * 2 + 1] = (byte)(bits >> 8);
        }

        return bytes;
    }

    private static float[] ToSingleArrayFromHalf(ReadOnlySpan<byte> bytes)
    {
        var ushortValues = MemoryMarshal.Cast<byte, ushort>(bytes);
        var values = new float[ushortValues.Length];
        for (var i = 0; i < ushortValues.Length; i++)
        {
            values[i] = (float)BitConverter.UInt16BitsToHalf(ushortValues[i]);
        }

        return values;
    }

    private static byte[] ToBFloat16Bytes(float[] values)
    {
        var bytes = new byte[values.Length * sizeof(ushort)];
        for (var i = 0; i < values.Length; i++)
        {
            var bits = (uint)BitConverter.SingleToInt32Bits(values[i]);
            var bf16 = (ushort)(bits >> 16);
            bytes[i * 2] = (byte)(bf16 & 0xFF);
            bytes[i * 2 + 1] = (byte)(bf16 >> 8);
        }

        return bytes;
    }

    private static float[] ToSingleArrayFromBFloat16(ReadOnlySpan<byte> bytes)
    {
        var ushortValues = MemoryMarshal.Cast<byte, ushort>(bytes);
        var values = new float[ushortValues.Length];
        for (var i = 0; i < ushortValues.Length; i++)
        {
            values[i] = BitConverter.Int32BitsToSingle(ushortValues[i] << 16);
        }

        return values;
    }

    private static byte[] ToBytes<T>(T[] values)
        where T : unmanaged
    {
        return typeof(T) == typeof(byte)
            ? (byte[])(object)values
            : MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
    }

    private static T[] ToArray<T>(ReadOnlySpan<byte> bytes)
        where T : unmanaged
    {
        if (typeof(T) == typeof(byte))
        {
            return (T[])(object)bytes.ToArray();
        }

        return MemoryMarshal.Cast<byte, T>(bytes).ToArray();
    }

    private readonly record struct StateTensorEntry(string Name, Tensor Tensor);
}
