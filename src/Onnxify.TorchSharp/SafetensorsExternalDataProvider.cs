using Onnxify.Data;
using Onnxify.Data.Numerics;
using Onnxify.Helpers;
using Onnxify.Safetensors;
using SafeTensorsArchive = Onnxify.Safetensors.SafeTensors;

namespace Onnxify.TorchSharp;

/// <summary>
/// Reads and writes single-tensor safetensors files for ONNX external-data integration.
/// </summary>
/// <remarks>
/// Use this provider when an <see cref="OnnxTensor"/> payload should live in a safetensors file instead of a raw binary sidecar. Each file is expected to contain exactly one tensor when reading through ONNX external-data hooks.
/// </remarks>
public sealed class SafetensorsExternalDataProvider : ExternalDataProvider
{
    private const string DefaultTensorName = "tensor";

    /// <summary>
    /// Gets the default singleton provider for safetensors-backed external tensor data.
    /// </summary>
    public static readonly SafetensorsExternalDataProvider Instance = new();

    /// <summary>
    /// Writes an Onnxify tensor as a single-tensor safetensors archive.
    /// </summary>
    /// <param name="tensor">Tensor whose raw payload, shape, and element type should be stored.</param>
    /// <param name="location">Destination safetensors path.</param>
    /// <param name="metadata">Optional archive metadata to include alongside the tensor.</param>
    /// <exception cref="NotSupportedException">Thrown when the tensor element type cannot be represented by safetensors.</exception>
    public void WriteTensor(
        OnnxTensor tensor,
        string location,
        IReadOnlyDictionary<string, string>? metadata = null
    )
    {
        ArgumentNullException.ThrowIfNull(tensor);
        ArgumentNullException.ThrowIfNull(location);

        var directory = Path.GetDirectoryName(location);
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        SafeTensorsArchive.SerializeToFile(
            data:
            [
                new KeyValuePair<string, TensorView>(
                    GetTensorName(tensor),
                    CreateTensorView(tensor)
                ),
            ],
            metadata: metadata,
            path: location
        );
    }

    /// <summary>
    /// Reads the single tensor in a safetensors file as a typed array.
    /// </summary>
    /// <typeparam name="T">Expected tensor element type.</typeparam>
    /// <param name="location">Safetensors file path.</param>
    /// <param name="expectedElementCount">Expected number of decoded elements, or <see langword="null"/> to return the full payload.</param>
    /// <returns>The decoded tensor values.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the file contains a different element count or data type.</exception>
    public T[] ReadTensorArray<T>(string location, long? expectedElementCount = null)
        where T : struct
    {
        var values = (T[])ReadTensorValue(location, offset: 0, length: -1, type: typeof(T));

        if (expectedElementCount is not long count)
        {
            return values;
        }

        if (count < 0 || count > values.LongLength)
        {
            throw new InvalidOperationException($"Tensor payload length mismatch. Expected {count} items, got {values.LongLength}.");
        }

        return values.AsSpan(0, checked((int)count)).ToArray();
    }

    /// <summary>
    /// Always throws because safetensors has no string tensor data type.
    /// </summary>
    /// <param name="location">Ignored safetensors path.</param>
    /// <returns>This method does not return.</returns>
    /// <exception cref="NotSupportedException">Always thrown.</exception>
    public string[] ReadStringArray(string location)
    {
        throw new NotSupportedException("Safetensors does not support string tensors.");
    }

    /// <summary>
    /// Reads a byte slice from the single tensor in a safetensors file and decodes it as an ONNX tensor payload.
    /// </summary>
    /// <param name="location">Safetensors file path.</param>
    /// <param name="offset">Byte offset within the tensor payload.</param>
    /// <param name="length">Number of bytes to read, or a negative value to read to the end of the payload.</param>
    /// <param name="type">Expected CLR tensor element type.</param>
    /// <returns>An array whose element type matches <paramref name="type"/>.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the archive does not contain exactly one tensor, the type differs, or the requested slice is out of range.</exception>
    public override object ReadTensorValue(
        string location,
        long offset,
        long length,
        Type type
    )
    {
        ArgumentNullException.ThrowIfNull(type);

        var tensor = ReadSingleTensor(location);
        ValidateTensorType(tensor, type);

        var payload = SlicePayload(tensor.Data.Span, offset, length);
        return DecodeRawData(payload, type);
    }

    private static TensorView CreateTensorView(OnnxTensor tensor)
    {
        if (tensor.DataType == typeof(string))
        {
            throw new NotSupportedException("Safetensors does not support string tensors.");
        }

        return new TensorView(
            dtype: MapDataType(tensor.DataType),
            shape: tensor.Shape.Select(static x => checked((ulong)x)),
            data: tensor.GetTensorRawData()
        );
    }

    private static TensorView ReadSingleTensor(string location)
    {
        if (!File.Exists(location))
        {
            throw new IOException($"File not found at '{location}'");
        }

        var raw = File.ReadAllBytes(location);
        var safetensors = SafeTensorsArchive.Deserialize(raw);
        if (safetensors.Length != 1)
        {
            throw new InvalidOperationException(
                $"Expected exactly one tensor in '{location}', but found {safetensors.Length}."
            );
        }

        return safetensors.Tensor(safetensors.Names()[0]);
    }

    private static ReadOnlySpan<byte> SlicePayload(
        ReadOnlySpan<byte> payload,
        long offset,
        long length
    )
    {
        if (offset < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(offset), offset, "Offset cannot be negative.");
        }

        var start = checked((int)offset);
        if (start > payload.Length)
        {
            throw new InvalidOperationException(
                $"Tensor payload offset {offset} is beyond the available {payload.Length} bytes."
            );
        }

        var availableLength = payload.Length - start;
        var sliceLength = length < 0
            ? availableLength
            : checked((int)length);

        if (sliceLength > availableLength)
        {
            throw new InvalidOperationException(
                $"Tensor payload length mismatch. Requested {sliceLength} bytes from offset {offset}, but only {availableLength} bytes are available."
            );
        }

        return payload.Slice(start, sliceLength);
    }

    private static void ValidateTensorType(TensorView tensor, Type requestedType)
    {
        var expectedType = MapDataType(requestedType);
        if (tensor.DataType != expectedType)
        {
            throw new InvalidOperationException(
                $"Tensor data type mismatch. Expected {expectedType.ToWireName()} but file contains {tensor.DataType.ToWireName()}."
            );
        }
    }

    private static string GetTensorName(OnnxTensor tensor)
        => string.IsNullOrWhiteSpace(tensor.Name) ? DefaultTensorName : tensor.Name;

    private static DataType MapDataType(Type type)
    {
        ArgumentNullException.ThrowIfNull(type);

        return type == typeof(bool) ? DataType.Bool
            : type == typeof(byte) ? DataType.U8
            : type == typeof(sbyte) ? DataType.I8
            : type == typeof(short) ? DataType.I16
            : type == typeof(ushort) ? DataType.U16
            : type == typeof(Half) ? DataType.F16
            : type == typeof(BFloat16) ? DataType.Bf16
            : type == typeof(int) ? DataType.I32
            : type == typeof(uint) ? DataType.U32
            : type == typeof(float) ? DataType.F32
            : type == typeof(Complex64) ? DataType.C64
            : type == typeof(double) ? DataType.F64
            : type == typeof(long) ? DataType.I64
            : type == typeof(ulong) ? DataType.U64
            : type == typeof(Float8E5M2) ? DataType.F8E5M2
            : type == typeof(Float8E4M3FN) ? DataType.F8E4M3
            : type == typeof(Float8E8M0) ? DataType.F8E8M0
            : type == typeof(Float8E4M3FNUZ) ? DataType.F8E4M3Fnuz
            : type == typeof(Float8E5M2FNUZ) ? DataType.F8E5M2Fnuz
            : type == typeof(Float4E2M1) ? DataType.F4
            : throw new NotSupportedException(
                $"Type '{type.FullName}' cannot be stored in safetensors external data."
            );
    }
}
