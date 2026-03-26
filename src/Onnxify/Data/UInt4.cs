namespace Onnxify.Data;

public readonly struct UInt4
{
    public byte Value { get; }

    public UInt4(byte value)
    {
        Value = (byte)(value & 0xF);
    }

    public byte ToByte() => (byte)(Value & 0xF);
}

