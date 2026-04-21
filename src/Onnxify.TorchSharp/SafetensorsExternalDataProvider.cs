using Onnxify.Data;
using Onnxify.Data.Numerics;
using Onnxify.Helpers;
using Onnxify.Safetensors;
using SafeTensorsArchive = Onnxify.Safetensors.SafeTensors;

namespace Onnxify.TorchSharp;

public sealed class SafetensorsExternalDataProvider : ExternalDataProvider
{
    private const string DefaultTensorName = "tensor";

    public static readonly SafetensorsExternalDataProvider Instance = new();

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

    public string[] ReadStringArray(string location)
    {
        throw new NotSupportedException("Safetensors does not support string tensors.");
    }

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
