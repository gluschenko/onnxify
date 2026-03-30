namespace Onnxify.Data.Numerics;

public readonly struct Int4
{
    public byte Value { get; }

    public Int4(sbyte value)
    {
        Value = (byte)(value & 0xF);
    }

    public sbyte ToSByte()
    {
        int v = Value & 0xF;
        if (v >= 8) v -= 16;
        return (sbyte)v;
    }
}

