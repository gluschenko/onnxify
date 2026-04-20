using System.Buffers;
using System.Buffers.Binary;
using System.Text;
using System.Text.Json;

namespace Onnxify.Safetensors;

public sealed class Safetensors
{
    private const int MaxHeaderSize = 100_000_000;
    private const int LengthPrefixSize = sizeof(ulong);
    private static readonly UTF8Encoding StrictUtf8 = new(false, true);

    private readonly Metadata _metadata;
    private readonly ReadOnlyMemory<byte> _data;

    private Safetensors(Metadata metadata, ReadOnlyMemory<byte> data)
    {
        _metadata = metadata;
        _data = data;
    }

    public Metadata Metadata => _metadata;

    public static MetadataReadResult ReadMetadata(ReadOnlyMemory<byte> buffer)
    {
        if (buffer.Length < LengthPrefixSize)
        {
            throw SafetensorException.HeaderTooSmall();
        }

        var headerLengthRaw = BinaryPrimitives.ReadUInt64LittleEndian(buffer.Span[..LengthPrefixSize]);
        if (headerLengthRaw > MaxHeaderSize)
        {
            throw SafetensorException.HeaderTooLarge();
        }

        int headerLength;
        try
        {
            headerLength = checked((int)headerLengthRaw);
        }
        catch (OverflowException)
        {
            throw SafetensorException.HeaderTooLarge();
        }

        int stop;
        try
        {
            stop = checked(headerLength + LengthPrefixSize);
        }
        catch (OverflowException)
        {
            throw SafetensorException.InvalidHeaderLength();
        }

        if (stop > buffer.Length)
        {
            throw SafetensorException.InvalidHeaderLength();
        }

        var headerBytes = buffer.Span.Slice(LengthPrefixSize, headerLength);

        string headerText;
        try
        {
            headerText = StrictUtf8.GetString(headerBytes);
        }
        catch (DecoderFallbackException ex)
        {
            throw SafetensorException.InvalidHeader(ex);
        }

        JsonDocument document;
        try
        {
            document = JsonDocument.Parse(headerText);
        }
        catch (JsonException ex)
        {
            throw SafetensorException.InvalidHeaderDeserialization(ex);
        }

        using (document)
        {
            Metadata metadata;
            try
            {
                metadata = ParseMetadata(document.RootElement);
            }
            catch (JsonException ex)
            {
                throw SafetensorException.InvalidHeaderDeserialization(ex);
            }
            catch (SafetensorException)
            {
                throw;
            }
            catch (Exception ex)
            {
                throw SafetensorException.JsonError(ex);
            }

            var bufferEnd = metadata.DataLength();
            if (bufferEnd + (ulong)LengthPrefixSize + (ulong)headerLength != (ulong)buffer.Length)
            {
                throw SafetensorException.MetadataIncompleteBuffer();
            }

            return new MetadataReadResult(headerLength, metadata);
        }
    }

    public static Safetensors Deserialize(ReadOnlyMemory<byte> buffer)
    {
        var readResult = ReadMetadata(buffer);
        var data = buffer.Slice(LengthPrefixSize + readResult.HeaderLength);
        return new Safetensors(readResult.Metadata, data);
    }

    public static byte[] Serialize(
        IEnumerable<KeyValuePair<string, TensorView>> data,
        IReadOnlyDictionary<string, string>? metadata = null)
    {
        ArgumentNullException.ThrowIfNull(data);

        var prepared = Prepare(data, metadata);
        if (prepared.HeaderBytes.Length > MaxHeaderSize)
        {
            throw SafetensorException.HeaderTooLarge();
        }

        ulong totalTensorBytes = 0;
        foreach (var tensor in prepared.Tensors)
        {
            totalTensorBytes = checked(totalTensorBytes + (ulong)tensor.Value.Data.Length);
        }

        var totalLength = checked(LengthPrefixSize + prepared.HeaderBytes.Length + CheckedInt(totalTensorBytes));
        var buffer = new byte[totalLength];
        BinaryPrimitives.WriteUInt64LittleEndian(buffer.AsSpan(0, LengthPrefixSize), (ulong)prepared.HeaderBytes.Length);
        prepared.HeaderBytes.CopyTo(buffer.AsSpan(LengthPrefixSize));

        var offset = LengthPrefixSize + prepared.HeaderBytes.Length;
        foreach (var tensor in prepared.Tensors)
        {
            tensor.Value.Data.Span.CopyTo(buffer.AsSpan(offset));
            offset += tensor.Value.Data.Length;
        }

        return buffer;
    }

    public static void SerializeToFile(
        IEnumerable<KeyValuePair<string, TensorView>> data,
        IReadOnlyDictionary<string, string>? metadata,
        string path)
    {
        ArgumentNullException.ThrowIfNull(path);

        var prepared = Prepare(data, metadata);
        if (prepared.HeaderBytes.Length > MaxHeaderSize)
        {
            throw SafetensorException.HeaderTooLarge();
        }

        try
        {
            using var file = File.Create(path);
            Span<byte> headerLength = stackalloc byte[LengthPrefixSize];
            BinaryPrimitives.WriteUInt64LittleEndian(headerLength, (ulong)prepared.HeaderBytes.Length);
            file.Write(headerLength);
            file.Write(prepared.HeaderBytes);

            foreach (var tensor in prepared.Tensors)
            {
                file.Write(tensor.Value.Data.Span);
            }
        }
        catch (IOException ex)
        {
            throw SafetensorException.IoError(ex);
        }
    }

    public IReadOnlyList<KeyValuePair<string, TensorView>> Tensors()
    {
        var tensors = new List<KeyValuePair<string, TensorView>>(_metadata.OffsetKeys().Count);
        foreach (var name in _metadata.OffsetKeys())
        {
            tensors.Add(new KeyValuePair<string, TensorView>(name, Tensor(name)));
        }

        return tensors;
    }

    public IEnumerable<KeyValuePair<string, TensorView>> Iter()
    {
        foreach (var name in _metadata.OffsetKeys())
        {
            yield return new KeyValuePair<string, TensorView>(name, Tensor(name));
        }
    }

    public TensorView Tensor(string tensorName)
    {
        ArgumentNullException.ThrowIfNull(tensorName);

        var info = _metadata.Info(tensorName);
        if (info is null)
        {
            throw SafetensorException.TensorNotFound(tensorName);
        }

        var start = CheckedInt(info.DataOffsets.Start);
        var length = CheckedInt(info.DataOffsets.End - info.DataOffsets.Start);
        return new TensorView(info.DataType, info.Shape, _data.Slice(start, length));
    }

    public IReadOnlyList<string> Names() => _metadata.OffsetKeys();

    public int Length => _metadata.OffsetKeys().Count;

    public bool IsEmpty => Length == 0;

    private static PreparedSafetensorsData Prepare(
        IEnumerable<KeyValuePair<string, TensorView>> data,
        IReadOnlyDictionary<string, string>? metadataEntries)
    {
        var sorted = data
            .OrderByDescending(x => x.Value.DataType)
            .ThenBy(x => x.Key, StringComparer.Ordinal)
            .ToList();

        ulong offset = 0;
        var tensors = new List<KeyValuePair<string, TensorInfo>>(sorted.Count);

        foreach (var pair in sorted)
        {
            var length = (ulong)pair.Value.Data.Length;
            tensors.Add(new KeyValuePair<string, TensorInfo>(
                pair.Key,
                new TensorInfo
                {
                    DataType = pair.Value.DataType,
                    Shape = pair.Value.Shape.ToArray(),
                    DataOffsets = new TensorDataOffsets(offset, checked(offset + length)),
                }));

            offset = checked(offset + length);
        }

        var metadata = new Metadata(metadataEntries, tensors);
        var headerBytes = SerializeMetadata(metadata);
        var originalLength = headerBytes.Length;
        var alignedLength = AlignToEightBytes(originalLength);
        if (alignedLength != originalLength)
        {
            Array.Resize(ref headerBytes, alignedLength);
            Array.Fill(headerBytes, (byte)' ', originalLength, alignedLength - originalLength);
        }

        return new PreparedSafetensorsData(headerBytes, sorted);
    }

    private static Metadata ParseMetadata(JsonElement root)
    {
        if (root.ValueKind != JsonValueKind.Object)
        {
            throw new JsonException("Safetensors header root must be a JSON object.");
        }

        Dictionary<string, string>? metadataEntries = null;
        var tensors = new Dictionary<string, TensorInfo>(StringComparer.Ordinal);

        foreach (var property in root.EnumerateObject())
        {
            if (property.NameEquals("__metadata__"))
            {
                metadataEntries = ParseMetadataEntries(property.Value);
            }
            else
            {
                tensors[property.Name] = ParseTensorInfo(property.Value);
            }
        }

        var ordered = tensors
            .OrderBy(x => x.Value.DataOffsets.Start)
            .ThenBy(x => x.Value.DataOffsets.End)
            .Select(x => new KeyValuePair<string, TensorInfo>(x.Key, x.Value));

        return new Metadata(metadataEntries, ordered);
    }

    private static Dictionary<string, string> ParseMetadataEntries(JsonElement value)
    {
        if (value.ValueKind != JsonValueKind.Object)
        {
            throw new JsonException("__metadata__ must be a JSON object.");
        }

        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var property in value.EnumerateObject())
        {
            result[property.Name] = property.Value.GetString()
                ?? throw new JsonException("Metadata values must be strings.");
        }

        return result;
    }

    private static TensorInfo ParseTensorInfo(JsonElement value)
    {
        if (value.ValueKind != JsonValueKind.Object)
        {
            throw new JsonException("Tensor entries must be JSON objects.");
        }

        string? dtypeName = null;
        ulong[]? shape = null;
        ulong[]? offsets = null;

        foreach (var property in value.EnumerateObject())
        {
            switch (property.Name)
            {
                case "dtype":
                    dtypeName = property.Value.GetString();
                    break;
                case "shape":
                    shape = property.Value.EnumerateArray().Select(x => x.GetUInt64()).ToArray();
                    break;
                case "data_offsets":
                    offsets = property.Value.EnumerateArray().Select(x => x.GetUInt64()).ToArray();
                    break;
            }
        }

        if (dtypeName is null || shape is null || offsets is null || offsets.Length != 2)
        {
            throw new JsonException("Tensor info is missing required fields.");
        }

        return new TensorInfo
        {
            DataType = DataTypeExtensions.ParseWireName(dtypeName),
            Shape = shape,
            DataOffsets = new TensorDataOffsets(offsets[0], offsets[1]),
        };
    }

    private static byte[] SerializeMetadata(Metadata metadata)
    {
        var buffer = new ArrayBufferWriter<byte>();
        using var writer = new Utf8JsonWriter(buffer);

        writer.WriteStartObject();

        if (metadata.MetadataEntries is not null)
        {
            writer.WritePropertyName("__metadata__");
            writer.WriteStartObject();
            foreach (var pair in metadata.MetadataEntries)
            {
                writer.WriteString(pair.Key, pair.Value);
            }
            writer.WriteEndObject();
        }

        foreach (var name in metadata.OffsetKeys())
        {
            var info = metadata.Info(name)!;
            writer.WritePropertyName(name);
            writer.WriteStartObject();
            writer.WriteString("dtype", info.DataType.ToWireName());
            writer.WritePropertyName("shape");
            writer.WriteStartArray();
            foreach (var dim in info.Shape)
            {
                writer.WriteNumberValue(dim);
            }
            writer.WriteEndArray();
            writer.WritePropertyName("data_offsets");
            writer.WriteStartArray();
            writer.WriteNumberValue(info.DataOffsets.Start);
            writer.WriteNumberValue(info.DataOffsets.End);
            writer.WriteEndArray();
            writer.WriteEndObject();
        }

        writer.WriteEndObject();
        writer.Flush();

        return buffer.WrittenMemory.ToArray();
    }

    private static int AlignToEightBytes(int value)
    {
        var remainder = value % LengthPrefixSize;
        return remainder == 0 ? value : checked(value + (LengthPrefixSize - remainder));
    }

    private static int CheckedInt(ulong value)
    {
        if (value > int.MaxValue)
        {
            throw SafetensorException.ValidationOverflow();
        }

        return (int)value;
    }
}
