namespace Onnxify.Data;

public readonly struct BFloat16
{
    public ushort Value { get; }

    public BFloat16(float value)
    {
        uint bits = (uint)BitConverter.SingleToInt32Bits(value);
        Value = (ushort)(bits >> 16);
    }

    public float ToSingle()
    {
        uint bits = (uint)Value << 16;
        return BitConverter.Int32BitsToSingle((int)bits);
    }
}
