using System.Collections.Immutable;
using System.Text;

namespace Onnxify.ModelGenerator;

internal static class OnnxModelMetadataReader
{
    public static ParsedOnnxModel ReadModel(byte[] data)
    {
        if (data is null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        var index = 0;
        ParsedOnnxGraph? graph = null;

        while (TryReadTag(data, ref index, data.Length, out var fieldNumber, out var wireType))
        {
            if (fieldNumber == 7 && wireType == 2)
            {
                var messageEnd = ReadLengthDelimitedEnd(data, ref index, data.Length);
                graph = ReadGraph(data, ref index, messageEnd);
                continue;
            }

            SkipField(data, ref index, data.Length, wireType);
        }

        return new ParsedOnnxModel(graph ?? ParsedOnnxGraph.Empty);
    }

    private static ParsedOnnxGraph ReadGraph(byte[] data, ref int index, int end)
    {
        var initializers = new List<ParsedOnnxTensorInitializer>();
        var inputs = new List<ParsedOnnxValueInfo>();
        var outputs = new List<ParsedOnnxValueInfo>();

        while (TryReadTag(data, ref index, end, out var fieldNumber, out var wireType))
        {
            switch (fieldNumber)
            {
                case 5 when wireType == 2:
                {
                    var messageEnd = ReadLengthDelimitedEnd(data, ref index, end);
                    initializers.Add(ReadTensorInitializer(data, ref index, messageEnd));
                    break;
                }
                case 11 when wireType == 2:
                {
                    var messageEnd = ReadLengthDelimitedEnd(data, ref index, end);
                    inputs.Add(ReadValueInfo(data, ref index, messageEnd));
                    break;
                }
                case 12 when wireType == 2:
                {
                    var messageEnd = ReadLengthDelimitedEnd(data, ref index, end);
                    outputs.Add(ReadValueInfo(data, ref index, messageEnd));
                    break;
                }
                default:
                    SkipField(data, ref index, end, wireType);
                    break;
            }
        }

        return new ParsedOnnxGraph(
            initializers.ToImmutableArray(),
            inputs.ToImmutableArray(),
            outputs.ToImmutableArray()
        );
    }

    private static ParsedOnnxTensorInitializer ReadTensorInitializer(byte[] data, ref int index, int end)
    {
        string name = string.Empty;
        var hasExternalData = false;

        while (TryReadTag(data, ref index, end, out var fieldNumber, out var wireType))
        {
            switch (fieldNumber)
            {
                case 8 when wireType == 2:
                    name = ReadString(data, ref index, end);
                    break;
                case 13 when wireType == 2:
                    hasExternalData = true;
                    SkipField(data, ref index, end, wireType);
                    break;
                case 14 when wireType == 0:
                    hasExternalData = ReadVarint(data, ref index, end) == 1UL || hasExternalData;
                    break;
                default:
                    SkipField(data, ref index, end, wireType);
                    break;
            }
        }

        return new ParsedOnnxTensorInitializer(name, hasExternalData);
    }

    private static ParsedOnnxValueInfo ReadValueInfo(byte[] data, ref int index, int end)
    {
        string name = string.Empty;
        ParsedOnnxType? type = null;

        while (TryReadTag(data, ref index, end, out var fieldNumber, out var wireType))
        {
            switch (fieldNumber)
            {
                case 1 when wireType == 2:
                    name = ReadString(data, ref index, end);
                    break;
                case 2 when wireType == 2:
                {
                    var messageEnd = ReadLengthDelimitedEnd(data, ref index, end);
                    type = ReadType(data, ref index, messageEnd);
                    break;
                }
                default:
                    SkipField(data, ref index, end, wireType);
                    break;
            }
        }

        return new ParsedOnnxValueInfo(name, type ?? ParsedOnnxType.Unknown);
    }

    private static ParsedOnnxType ReadType(byte[] data, ref int index, int end)
    {
        var kind = OnnxValueKind.Unknown;
        ParsedOnnxTensorType? tensorType = null;
        string denotation = string.Empty;

        while (TryReadTag(data, ref index, end, out var fieldNumber, out var wireType))
        {
            switch (fieldNumber)
            {
                case 1 when wireType == 2:
                {
                    kind = OnnxValueKind.Tensor;
                    var messageEnd = ReadLengthDelimitedEnd(data, ref index, end);
                    tensorType = ReadTensorType(data, ref index, messageEnd);
                    break;
                }
                case 4:
                    kind = OnnxValueKind.Sequence;
                    SkipField(data, ref index, end, wireType);
                    break;
                case 5:
                    kind = OnnxValueKind.Map;
                    SkipField(data, ref index, end, wireType);
                    break;
                case 7:
                    kind = OnnxValueKind.Opaque;
                    SkipField(data, ref index, end, wireType);
                    break;
                case 8:
                    kind = OnnxValueKind.SparseTensor;
                    SkipField(data, ref index, end, wireType);
                    break;
                case 9:
                    kind = OnnxValueKind.Optional;
                    SkipField(data, ref index, end, wireType);
                    break;
                case 6 when wireType == 2:
                    denotation = ReadString(data, ref index, end);
                    break;
                default:
                    SkipField(data, ref index, end, wireType);
                    break;
            }
        }

        return new ParsedOnnxType(kind, tensorType, denotation);
    }

    private static ParsedOnnxTensorType ReadTensorType(byte[] data, ref int index, int end)
    {
        var dataType = OnnxTensorDataType.Undefined;
        var dimensions = ImmutableArray<ParsedOnnxDimension>.Empty;

        while (TryReadTag(data, ref index, end, out var fieldNumber, out var wireType))
        {
            switch (fieldNumber)
            {
                case 1 when wireType == 0:
                    dataType = (OnnxTensorDataType)ReadInt32(data, ref index, end);
                    break;
                case 2 when wireType == 2:
                {
                    var messageEnd = ReadLengthDelimitedEnd(data, ref index, end);
                    dimensions = ReadTensorShape(data, ref index, messageEnd);
                    break;
                }
                default:
                    SkipField(data, ref index, end, wireType);
                    break;
            }
        }

        return new ParsedOnnxTensorType(dataType, dimensions);
    }

    private static ImmutableArray<ParsedOnnxDimension> ReadTensorShape(byte[] data, ref int index, int end)
    {
        var dimensions = new List<ParsedOnnxDimension>();

        while (TryReadTag(data, ref index, end, out var fieldNumber, out var wireType))
        {
            if (fieldNumber == 1 && wireType == 2)
            {
                var messageEnd = ReadLengthDelimitedEnd(data, ref index, end);
                dimensions.Add(ReadDimension(data, ref index, messageEnd));
                continue;
            }

            SkipField(data, ref index, end, wireType);
        }

        return dimensions.ToImmutableArray();
    }

    private static ParsedOnnxDimension ReadDimension(byte[] data, ref int index, int end)
    {
        long? numericValue = null;
        string? symbolicName = null;

        while (TryReadTag(data, ref index, end, out var fieldNumber, out var wireType))
        {
            switch (fieldNumber)
            {
                case 1 when wireType == 0:
                    numericValue = ReadInt64(data, ref index, end);
                    break;
                case 2 when wireType == 2:
                    symbolicName = ReadString(data, ref index, end);
                    break;
                default:
                    SkipField(data, ref index, end, wireType);
                    break;
            }
        }

        return new ParsedOnnxDimension(numericValue, symbolicName);
    }

    private static string ReadString(byte[] data, ref int index, int end)
    {
        var messageEnd = ReadLengthDelimitedEnd(data, ref index, end);
        var value = Encoding.UTF8.GetString(data, index, messageEnd - index);
        index = messageEnd;
        return value;
    }

    private static int ReadLengthDelimitedEnd(byte[] data, ref int index, int end)
    {
        var length = ReadInt32(data, ref index, end);
        if (length < 0 || index + length > end)
        {
            throw new InvalidDataException("Invalid protobuf length-delimited payload.");
        }

        return index + length;
    }

    private static bool TryReadTag(byte[] data, ref int index, int end, out int fieldNumber, out int wireType)
    {
        if (index >= end)
        {
            fieldNumber = 0;
            wireType = 0;
            return false;
        }

        var tag = ReadVarint(data, ref index, end);
        if (tag == 0)
        {
            throw new InvalidDataException("Invalid protobuf tag 0.");
        }

        fieldNumber = (int)(tag >> 3);
        wireType = (int)(tag & 0x07);
        return true;
    }

    private static void SkipField(byte[] data, ref int index, int end, int wireType)
    {
        switch (wireType)
        {
            case 0:
                ReadVarint(data, ref index, end);
                return;
            case 1:
                Advance(ref index, 8, end);
                return;
            case 2:
                index = ReadLengthDelimitedEnd(data, ref index, end);
                return;
            case 5:
                Advance(ref index, 4, end);
                return;
            default:
                throw new InvalidDataException($"Unsupported protobuf wire type '{wireType}'.");
        }
    }

    private static void Advance(ref int index, int count, int end)
    {
        if (index + count > end)
        {
            throw new InvalidDataException("Unexpected end of protobuf payload.");
        }

        index += count;
    }

    private static int ReadInt32(byte[] data, ref int index, int end)
    {
        return checked((int)ReadVarint(data, ref index, end));
    }

    private static long ReadInt64(byte[] data, ref int index, int end)
    {
        return checked((long)ReadVarint(data, ref index, end));
    }

    private static ulong ReadVarint(byte[] data, ref int index, int end)
    {
        ulong result = 0;
        var shift = 0;

        while (shift < 64)
        {
            if (index >= end)
            {
                throw new InvalidDataException("Unexpected end of protobuf varint.");
            }

            var current = data[index++];
            result |= ((ulong)(current & 0x7F)) << shift;

            if ((current & 0x80) == 0)
            {
                return result;
            }

            shift += 7;
        }

        throw new InvalidDataException("Protobuf varint is too large.");
    }
}

internal sealed class ParsedOnnxModel
{
    public ParsedOnnxModel(ParsedOnnxGraph graph)
    {
        Graph = graph;
    }

    public ParsedOnnxGraph Graph { get; }
}

internal sealed class ParsedOnnxGraph
{
    public static ParsedOnnxGraph Empty { get; } = new(
        ImmutableArray<ParsedOnnxTensorInitializer>.Empty,
        ImmutableArray<ParsedOnnxValueInfo>.Empty,
        ImmutableArray<ParsedOnnxValueInfo>.Empty);

    public ParsedOnnxGraph(
        ImmutableArray<ParsedOnnxTensorInitializer> initializers,
        ImmutableArray<ParsedOnnxValueInfo> inputs,
        ImmutableArray<ParsedOnnxValueInfo> outputs)
    {
        Initializers = initializers;
        Inputs = inputs;
        Outputs = outputs;
    }

    public ImmutableArray<ParsedOnnxTensorInitializer> Initializers { get; }

    public ImmutableArray<ParsedOnnxValueInfo> Inputs { get; }

    public ImmutableArray<ParsedOnnxValueInfo> Outputs { get; }
}

internal sealed class ParsedOnnxTensorInitializer
{
    public ParsedOnnxTensorInitializer(string name, bool hasExternalData)
    {
        Name = name;
        HasExternalData = hasExternalData;
    }

    public string Name { get; }

    public bool HasExternalData { get; }
}

internal sealed class ParsedOnnxValueInfo
{
    public ParsedOnnxValueInfo(string name, ParsedOnnxType type)
    {
        Name = name;
        Type = type;
    }

    public string Name { get; }

    public ParsedOnnxType Type { get; }
}

internal sealed class ParsedOnnxType
{
    public static ParsedOnnxType Unknown { get; } = new(OnnxValueKind.Unknown, null, string.Empty);

    public ParsedOnnxType(OnnxValueKind kind, ParsedOnnxTensorType? tensorType, string denotation)
    {
        Kind = kind;
        TensorType = tensorType;
        Denotation = denotation;
    }

    public OnnxValueKind Kind { get; }

    public ParsedOnnxTensorType? TensorType { get; }

    public string Denotation { get; }
}

internal sealed class ParsedOnnxTensorType
{
    public ParsedOnnxTensorType(OnnxTensorDataType elementType, ImmutableArray<ParsedOnnxDimension> shape)
    {
        ElementType = elementType;
        Shape = shape;
    }

    public OnnxTensorDataType ElementType { get; }

    public ImmutableArray<ParsedOnnxDimension> Shape { get; }
}

internal sealed class ParsedOnnxDimension
{
    public ParsedOnnxDimension(long? numericValue, string? symbolicName)
    {
        NumericValue = numericValue;
        SymbolicName = symbolicName;
    }

    public long? NumericValue { get; }

    public string? SymbolicName { get; }
}

internal enum OnnxValueKind
{
    Unknown = 0,
    Tensor = 1,
    Sequence = 4,
    Map = 5,
    Opaque = 7,
    SparseTensor = 8,
    Optional = 9,
}

internal enum OnnxTensorDataType
{
    Undefined = 0,
    Float = 1,
    Uint8 = 2,
    Int8 = 3,
    Uint16 = 4,
    Int16 = 5,
    Int32 = 6,
    Int64 = 7,
    String = 8,
    Bool = 9,
    Float16 = 10,
    Double = 11,
    Uint32 = 12,
    Uint64 = 13,
    Complex64 = 14,
    Complex128 = 15,
    BFloat16 = 16,
    Float8E4M3Fn = 17,
    Float8E4M3Fnuz = 18,
    Float8E5M2 = 19,
    Float8E5M2Fnuz = 20,
    Uint4 = 21,
    Int4 = 22,
    Float4E2M1 = 23,
    Float8E8M0 = 24,
    Uint2 = 25,
    Int2 = 26,
}
