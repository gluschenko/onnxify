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

    protected virtual object DecodeRawData(
        ReadOnlySpan<byte> span,
        Type type
    )
    {
        return OnnxHelper.DecodeRawData(span, type);
    }
}

public sealed class OnnxExternalDataProvider : ExternalDataProvider
{
    public static readonly OnnxExternalDataProvider Instance = new();

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
