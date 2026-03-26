namespace Onnxify.Data;

public readonly struct Int2
{
    public byte Value { get; }

    public Int2(sbyte value)
    {
        Value = (byte)(value & 0x3);
    }

    public sbyte ToSByte()
    {
        int v = Value & 0x3;
        if (v >= 2) v -= 4;
        return (sbyte)v;
    }
}

