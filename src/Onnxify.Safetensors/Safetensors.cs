using System.Buffers;
using System.Buffers.Binary;
using System.Text;
using System.Text.Json;

namespace Onnxify.Safetensors;

/// <summary>
/// Represents a validated safetensors archive and exposes zero-copy tensor views over the payload region of the source buffer.
/// </summary>
/// <remarks>
/// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
/// Original Rust entity: <c>SafeTensors</c>.
/// </remarks>
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

    /// <summary>
    /// Gets the validated header metadata that indexes the tensors inside this archive.
    /// </summary>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>SafeTensors.metadata</c>.
    /// </remarks>
    public Metadata Metadata => _metadata;

    /// <summary>
    /// Parses only the header portion of a safetensors buffer and returns both the header byte length and validated metadata.
    /// </summary>
    /// <param name="buffer">The complete safetensors file contents, including the 8-byte length prefix.</param>
    /// <returns>The parsed header length and validated metadata.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>SafeTensors::read_metadata</c>.
    /// </remarks>
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

    /// <summary>
    /// Parses a complete safetensors buffer and returns a managed archive view without copying the tensor payload bytes.
    /// </summary>
    /// <param name="buffer">The complete safetensors file contents.</param>
    /// <returns>A validated archive view over the buffer.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>SafeTensors::deserialize</c>.
    /// </remarks>
    public static Safetensors Deserialize(ReadOnlyMemory<byte> buffer)
    {
        var readResult = ReadMetadata(buffer);
        var data = buffer.Slice(LengthPrefixSize + readResult.HeaderLength);
        return new Safetensors(readResult.Metadata, data);
    }

    /// <summary>
    /// Serializes tensor views and optional archive metadata into a complete safetensors byte buffer.
    /// </summary>
    /// <param name="data">The tensors to serialize, keyed by safetensors tensor name.</param>
    /// <param name="metadata">Optional top-level <c>__metadata__</c> entries.</param>
    /// <returns>A byte buffer containing a complete safetensors file.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>serialize</c>.
    /// </remarks>
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

    /// <summary>
    /// Streams tensor views and optional archive metadata directly to a safetensors file on disk.
    /// </summary>
    /// <param name="data">The tensors to serialize, keyed by safetensors tensor name.</param>
    /// <param name="metadata">Optional top-level <c>__metadata__</c> entries.</param>
    /// <param name="path">The destination file path.</param>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>serialize_to_file</c>.
    /// </remarks>
    public static void SerializeToFile(
        IEnumerable<KeyValuePair<string, TensorView>> data,
        IReadOnlyDictionary<string, string>? metadata,
        string path)
    {
        ArgumentNullException.ThrowIfNull(data);
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

    /// <summary>
    /// Materializes all tensors in archive order as named tensor views over the underlying payload buffer.
    /// </summary>
    /// <returns>A list of tensor name/view pairs.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>SafeTensors::tensors</c>.
    /// </remarks>
    public IReadOnlyList<KeyValuePair<string, TensorView>> Tensors()
    {
        var tensors = new List<KeyValuePair<string, TensorView>>(_metadata.OffsetKeys().Count);
        foreach (var name in _metadata.OffsetKeys())
        {
            tensors.Add(new KeyValuePair<string, TensorView>(name, Tensor(name)));
        }

        return tensors;
    }

    /// <summary>
    /// Lazily enumerates archive tensors in metadata order without copying payload bytes.
    /// </summary>
    /// <returns>An enumerable of tensor name/view pairs.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>SafeTensors::iter</c>.
    /// </remarks>
    public IEnumerable<KeyValuePair<string, TensorView>> Iter()
    {
        foreach (var name in _metadata.OffsetKeys())
        {
            yield return new KeyValuePair<string, TensorView>(name, Tensor(name));
        }
    }

    /// <summary>
    /// Resolves a single tensor by name and returns a view over its payload bytes.
    /// </summary>
    /// <param name="tensorName">The tensor name to resolve.</param>
    /// <returns>A tensor view over the requested payload slice.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>SafeTensors::tensor</c>.
    /// </remarks>
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

    /// <summary>
    /// Returns the tensor names exposed by this archive in metadata order.
    /// </summary>
    /// <returns>An ordered list of tensor names.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>SafeTensors::names</c>.
    /// </remarks>
    public IReadOnlyList<string> Names() => _metadata.OffsetKeys();

    /// <summary>
    /// Gets the number of tensors described by the archive metadata.
    /// </summary>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>SafeTensors::len</c>.
    /// </remarks>
    public int Length => _metadata.OffsetKeys().Count;

    /// <summary>
    /// Gets a value indicating whether the archive contains no tensors.
    /// </summary>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>SafeTensors::is_empty</c>.
    /// </remarks>
    public bool IsEmpty => Length == 0;

    /// <summary>
    /// Prepares the ordered header model and emission order required for deterministic safetensors serialization.
    /// </summary>
    /// <param name="data">The named tensor views to serialize.</param>
    /// <param name="metadataEntries">Optional top-level metadata entries.</param>
    /// <returns>The aligned header bytes and ordered tensor sequence.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>prepare</c>.
    /// </remarks>
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

    /// <summary>
    /// Converts a parsed JSON root object into a validated <see cref="Metadata"/> instance.
    /// </summary>
    /// <param name="root">The root JSON object from the safetensors header.</param>
    /// <returns>A validated metadata model.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>Deserialize for Metadata</c> via <c>HashMetadata</c>.
    /// </remarks>
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

    /// <summary>
    /// Parses the optional top-level <c>__metadata__</c> dictionary from the header JSON.
    /// </summary>
    /// <param name="value">The JSON value stored under <c>__metadata__</c>.</param>
    /// <returns>A string dictionary with ordinal key comparison.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>HashMetadata.metadata</c>.
    /// </remarks>
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

    /// <summary>
    /// Parses a single tensor metadata record from the header JSON.
    /// </summary>
    /// <param name="value">The JSON object describing one tensor.</param>
    /// <returns>A tensor metadata entry suitable for validation and lookup.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>TensorInfo</c>.
    /// </remarks>
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

    /// <summary>
    /// Serializes validated metadata back into the exact JSON header structure required by safetensors.
    /// </summary>
    /// <param name="metadata">The validated metadata model to serialize.</param>
    /// <returns>The unpadded UTF-8 JSON header bytes.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>Serialize for Metadata</c>.
    /// </remarks>
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

    /// <summary>
    /// Rounds a header length up to the next 8-byte boundary to match safetensors header padding rules.
    /// </summary>
    /// <param name="value">The unaligned header length.</param>
    /// <returns>The smallest aligned length greater than or equal to <paramref name="value"/>.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: metadata alignment in <c>prepare</c> using <c>next_multiple_of</c>.
    /// </remarks>
    private static int AlignToEightBytes(int value)
    {
        var remainder = value % LengthPrefixSize;
        return remainder == 0 ? value : checked(value + (LengthPrefixSize - remainder));
    }

    /// <summary>
    /// Converts a validated unsigned byte count into <see cref="int"/> while preserving upstream overflow behavior.
    /// </summary>
    /// <param name="value">The byte count to convert.</param>
    /// <returns>The same value as a signed 32-bit integer.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: checked <c>usize</c> conversions around header and slice sizes.
    /// </remarks>
    private static int CheckedInt(ulong value)
    {
        if (value > int.MaxValue)
        {
            throw SafetensorException.ValidationOverflow();
        }

        return (int)value;
    }
}
