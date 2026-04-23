using System.Text.Json;
using Onnxify.Data;

namespace Onnxify;

/// <summary>
/// Default filesystem-backed reader for ONNX external tensor data.
/// </summary>
/// <remarks>
/// The provider expects binary tensor payloads to use ONNX raw-data layout. String tensors are read from a JSON string array by the convenience <see cref="ReadStringArray"/> method.
/// </remarks>
public sealed class OnnxExternalDataProvider : ExternalDataProvider
{
    /// <summary>
    /// Gets the default singleton used by model load and creation options.
    /// </summary>
    public static readonly OnnxExternalDataProvider Instance = new();

    /// <summary>
    /// Reads a complete raw tensor file and optionally trims it to an expected element count.
    /// </summary>
    /// <typeparam name="T">Value type matching the tensor element type.</typeparam>
    /// <param name="location">Filesystem path to the external tensor data.</param>
    /// <param name="expectedElementCount">Expected number of decoded elements, or <see langword="null"/> to return all decoded values.</param>
    /// <returns>The decoded tensor values.</returns>
    /// <exception cref="InvalidOperationException">Thrown when <paramref name="expectedElementCount"/> is negative or larger than the decoded payload.</exception>
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
    /// Reads a string tensor payload stored as a JSON string array.
    /// </summary>
    /// <param name="location">Filesystem path to the JSON payload.</param>
    /// <returns>The decoded string values.</returns>
    /// <exception cref="IOException">Thrown when the file does not exist.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the JSON payload cannot be decoded as a string array.</exception>
    public string[] ReadStringArray(string location)
    {
        if (!File.Exists(location))
        {
            throw new IOException($"File not found at '{location}'");
        }

        return JsonSerializer.Deserialize<string[]>(File.ReadAllText(location))
            ?? throw new InvalidOperationException($"Could not deserialize tensor data from '{location}'.");
    }

    /// <summary>
    /// Reads and decodes raw tensor bytes from a filesystem path.
    /// </summary>
    /// <param name="location">Filesystem path to the external tensor data.</param>
    /// <param name="offset">Byte offset where the tensor payload begins.</param>
    /// <param name="length">Number of bytes to read, or a negative value to read to the end of the file.</param>
    /// <param name="type">CLR element type to decode.</param>
    /// <returns>An array whose element type corresponds to <paramref name="type"/>.</returns>
    /// <exception cref="IOException">Thrown when the file does not exist.</exception>
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
