namespace Onnxify.Data.Numerics;

public readonly struct UInt2
{
    public byte Value { get; }

    public UInt2(byte value)
    {
        Value = (byte)(value & 0x3);
    }

    public byte ToByte() => (byte)(Value & 0x3);
}

