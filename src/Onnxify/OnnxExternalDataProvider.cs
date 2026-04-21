using System.Text.Json;
using Onnxify.Data;

namespace Onnxify;

public sealed class OnnxExternalDataProvider : ExternalDataProvider
{
    public static readonly OnnxExternalDataProvider Instance = new();

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
        if (!File.Exists(location))
        {
            throw new IOException($"File not found at '{location}'");
        }

        return JsonSerializer.Deserialize<string[]>(File.ReadAllText(location))
            ?? throw new InvalidOperationException($"Could not deserialize tensor data from '{location}'.");
    }

    public override object ReadTensorValue(
        string location,
        long offset,
        long length,
        Type type
    )
    {
        if (!File.Exists(location))
        {
            throw new IOException($"File not found at '{location}'");
        }

        using var fs = File.OpenRead(location);
        fs.Seek(offset, SeekOrigin.Begin);

        if (length < 0)
        {
            length = fs.Length - offset;
        }

        var buffer = new byte[length];
        fs.ReadExactly(buffer);

        var data = DecodeRawData(buffer, type);
        return data;
    }
}
