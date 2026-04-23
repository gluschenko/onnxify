using Onnxify.Helpers;

namespace Onnxify.Data;

/// <summary>
/// Defines how Onnxify reads tensor payloads stored outside the ONNX protobuf.
/// </summary>
/// <remarks>
/// Implement this when external data is stored in a custom location, compressed format, object store, or encrypted container. The returned value must be an array whose element type matches the requested CLR tensor type.
/// </remarks>
public abstract class ExternalDataProvider
{
    /// <summary>
    /// Reads an external tensor payload as a typed array represented by <see cref="Type"/>.
    /// </summary>
    /// <param name="location">Resolved external-data location.</param>
    /// <param name="offset">Byte offset where the tensor payload begins.</param>
    /// <param name="length">Number of bytes to read, or a negative value when the provider should read to the end of the payload.</param>
    /// <param name="type">CLR tensor element type expected by Onnxify.</param>
    /// <returns>An array containing decoded tensor elements.</returns>
    public abstract object ReadTensorValue(
        string location,
        long offset,
        long length,
        Type type
    );

    /// <summary>
    /// Reads an external tensor payload as a strongly typed array.
    /// </summary>
    /// <typeparam name="T">CLR tensor element type expected by the caller.</typeparam>
    /// <param name="location">Resolved external-data location.</param>
    /// <param name="offset">Byte offset where the tensor payload begins.</param>
    /// <param name="length">Number of bytes to read, or a negative value when the provider should read to the end of the payload.</param>
    /// <returns>The decoded tensor values.</returns>
    /// <exception cref="InvalidCastException">Thrown when the provider returns an array with a different element type.</exception>
    public virtual T[] ReadTensorValue<T>(
        string location,
        long offset,
        long length
    )
    {
        var type = typeof(T);

        var untypedResult = ReadTensorValue(
            location: location,
            offset: offset,
            length: length,
            type: type
        );

        if (untypedResult is not T[] result)
        {
            throw new InvalidCastException($"Failed to read tesnor value of type '{type.FullName}'. Real type is '{untypedResult.GetType().FullName}.'");
        }

        return result;
    }

    /// <summary>
    /// Decodes raw ONNX tensor bytes for providers that retrieve bytes but want Onnxify's standard element conversion.
    /// </summary>
    /// <param name="span">Raw tensor payload bytes.</param>
    /// <param name="type">CLR tensor element type expected by Onnxify.</param>
    /// <returns>An array containing decoded tensor elements.</returns>
    protected virtual object DecodeRawData(
        ReadOnlySpan<byte> span,
        Type type
    )
    {
        return OnnxHelper.DecodeRawData(span, type);
    }
}
