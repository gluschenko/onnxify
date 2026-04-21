using Onnxify.Helpers;

namespace Onnxify.Data;

public abstract class ExternalDataProvider
{
    public abstract object ReadTensorValue(
        string location,
        long offset,
        long length,
        Type type
    );

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

    protected virtual object DecodeRawData(
        ReadOnlySpan<byte> span,
        Type type
    )
    {
        return OnnxHelper.DecodeRawData(span, type);
    }
}
