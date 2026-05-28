using System.Buffers;
using System.Buffers.Binary;
using System.Globalization;
using System.Runtime.InteropServices;
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
public sealed class SafeTensors
{
    private const int MAX_HEADER_SIZE = 100_000_000;
    private const int LENGTH_PREFIX_SIZE = sizeof(ulong);
    private static readonly UTF8Encoding _strictUtf8 = new(false, true);

    private readonly Dictionary<string, TensorView> _tensors;
    private readonly Dictionary<string, string> _metadataEntries;

    public SafeTensors(IReadOnlyDictionary<string, string>? metadata = null)
    {
        _tensors = new Dictionary<string, TensorView>(StringComparer.Ordinal);
        _metadataEntries = metadata is null
            ? new Dictionary<string, string>(StringComparer.Ordinal)
            : new Dictionary<string, string>(metadata, StringComparer.Ordinal);
    }

    private SafeTensors(Metadata metadata, ReadOnlyMemory<byte> data)
        : this(metadata.MetadataEntries)
    {
        foreach (var name in metadata.OffsetKeys())
        {
            var info = metadata.Info(name)!;
            var start = CheckedInt(info.DataOffsets.Start);
            var length = CheckedInt(info.DataOffsets.End - info.DataOffsets.Start);
            _tensors.Add(name, new TensorView(info.DataType, info.Shape, data.Slice(start, length)));
        }
    }

    /// <summary>
    /// Gets the validated header metadata that indexes the tensors inside this archive.
    /// </summary>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>SafeTensors.metadata</c>.
    /// </remarks>
    public Metadata Metadata => BuildMetadata(_tensors, MetadataEntriesOrNull());

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
        if (buffer.Length < LENGTH_PREFIX_SIZE)
        {
            throw SafeTensorException.HeaderTooSmall();
        }

        var headerLengthRaw = BinaryPrimitives.ReadUInt64LittleEndian(buffer.Span[..LENGTH_PREFIX_SIZE]);
        if (headerLengthRaw > MAX_HEADER_SIZE)
        {
            throw SafeTensorException.HeaderTooLarge();
        }

        int headerLength;
        try
        {
            headerLength = checked((int)headerLengthRaw);
        }
        catch (OverflowException)
        {
            throw SafeTensorException.HeaderTooLarge();
        }

        int stop;
        try
        {
            stop = checked(headerLength + LENGTH_PREFIX_SIZE);
        }
        catch (OverflowException)
        {
            throw SafeTensorException.InvalidHeaderLength();
        }

        if (stop > buffer.Length)
        {
            throw SafeTensorException.InvalidHeaderLength();
        }

        var headerBytes = buffer.Span.Slice(LENGTH_PREFIX_SIZE, headerLength);

        string headerText;
        try
        {
            headerText = _strictUtf8.GetString(headerBytes);
        }
        catch (DecoderFallbackException ex)
        {
            throw SafeTensorException.InvalidHeader(ex);
        }

        JsonDocument document;
        try
        {
            document = JsonDocument.Parse(headerText);
        }
        catch (JsonException ex)
        {
            throw SafeTensorException.InvalidHeaderDeserialization(ex);
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
                throw SafeTensorException.InvalidHeaderDeserialization(ex);
            }
            catch (SafeTensorException)
            {
                throw;
            }
            catch (Exception ex)
            {
                throw SafeTensorException.JsonError(ex);
            }

            var bufferEnd = metadata.DataLength();
            if (bufferEnd + (ulong)LENGTH_PREFIX_SIZE + (ulong)headerLength != (ulong)buffer.Length)
            {
                throw SafeTensorException.MetadataIncompleteBuffer();
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
    public static SafeTensors Deserialize(ReadOnlyMemory<byte> buffer)
    {
        var readResult = ReadMetadata(buffer);
        var data = buffer.Slice(LENGTH_PREFIX_SIZE + readResult.HeaderLength);
        return new SafeTensors(readResult.Metadata, data);
    }

    /// <summary>
    /// Asynchronously reads a complete safetensors file from disk and parses it into a managed archive.
    /// </summary>
    /// <param name="path">The source file path.</param>
    /// <param name="cancellationToken">A token that can cancel the asynchronous file read.</param>
    /// <returns>A validated archive view over the loaded file contents.</returns>
    public static async Task<SafeTensors> LoadFromFileAsync(
        string path,
        CancellationToken cancellationToken = default
    )
    {
        ArgumentNullException.ThrowIfNull(path);

        try
        {
            var raw = await File.ReadAllBytesAsync(path, cancellationToken);
            return Deserialize(raw);
        }
        catch (IOException ex)
        {
            throw SafeTensorException.IoError(ex);
        }
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
        IReadOnlyDictionary<string, string>? metadata = null
    )
    {
        ArgumentNullException.ThrowIfNull(data);

        var prepared = Prepare(data, metadata);
        if (prepared.HeaderBytes.Length > MAX_HEADER_SIZE)
        {
            throw SafeTensorException.HeaderTooLarge();
        }

        ulong totalTensorBytes = 0;
        foreach (var tensor in prepared.Tensors)
        {
            totalTensorBytes = checked(totalTensorBytes + (ulong)tensor.Value.Data.Length);
        }

        var totalLength = checked(LENGTH_PREFIX_SIZE + prepared.HeaderBytes.Length + CheckedInt(totalTensorBytes));
        var buffer = new byte[totalLength];
        BinaryPrimitives.WriteUInt64LittleEndian(buffer.AsSpan(0, LENGTH_PREFIX_SIZE), (ulong)prepared.HeaderBytes.Length);
        prepared.HeaderBytes.CopyTo(buffer.AsSpan(LENGTH_PREFIX_SIZE));

        var offset = LENGTH_PREFIX_SIZE + prepared.HeaderBytes.Length;
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
        string path
    )
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(path);

        var prepared = Prepare(data, metadata);
        if (prepared.HeaderBytes.Length > MAX_HEADER_SIZE)
        {
            throw SafeTensorException.HeaderTooLarge();
        }

        try
        {
            using var file = File.Create(path);
            Span<byte> headerLength = stackalloc byte[LENGTH_PREFIX_SIZE];
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
            throw SafeTensorException.IoError(ex);
        }
    }

    /// <summary>
    /// Asynchronously streams tensor views and optional archive metadata directly to a safetensors file on disk.
    /// </summary>
    /// <param name="data">The tensors to serialize, keyed by safetensors tensor name.</param>
    /// <param name="metadata">Optional top-level <c>__metadata__</c> entries.</param>
    /// <param name="path">The destination file path.</param>
    /// <param name="cancellationToken">A token that can cancel the asynchronous file writes.</param>
    public static async Task SerializeToFileAsync(
        IEnumerable<KeyValuePair<string, TensorView>> data,
        IReadOnlyDictionary<string, string>? metadata,
        string path,
        CancellationToken cancellationToken = default
    )
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(path);

        var prepared = Prepare(data, metadata);
        if (prepared.HeaderBytes.Length > MAX_HEADER_SIZE)
        {
            throw SafeTensorException.HeaderTooLarge();
        }

        try
        {
            await using var file = File.Create(path);
            var headerLength = new byte[LENGTH_PREFIX_SIZE];
            BinaryPrimitives.WriteUInt64LittleEndian(headerLength, (ulong)prepared.HeaderBytes.Length);
            await file.WriteAsync(headerLength, cancellationToken);
            await file.WriteAsync(prepared.HeaderBytes, cancellationToken);

            foreach (var tensor in prepared.Tensors)
            {
                await file.WriteAsync(tensor.Value.Data, cancellationToken);
            }
        }
        catch (IOException ex)
        {
            throw SafeTensorException.IoError(ex);
        }
    }

    public byte[] Serialize()
        => Serialize(_tensors, MetadataEntriesOrNull());

    public void SerializeToFile(string path)
        => SerializeToFile(_tensors, MetadataEntriesOrNull(), path);

    /// <summary>
    /// Asynchronously saves this archive to a safetensors file on disk.
    /// </summary>
    /// <param name="path">The destination file path.</param>
    /// <param name="cancellationToken">A token that can cancel the asynchronous file writes.</param>
    public async Task SaveToFileAsync(
        string path,
        CancellationToken cancellationToken = default
    )
    {
        await SerializeToFileAsync(_tensors, MetadataEntriesOrNull(), path, cancellationToken);
    }

    /// <summary>
    /// Stores a named value and marshals the supplied CLR array into a safetensors tensor payload.
    /// </summary>
    /// <typeparam name="T">
    /// Supported types: <see cref="bool"/>, <see cref="byte"/>, <see cref="sbyte"/>, <see cref="short"/>,
    /// <see cref="ushort"/>, <see cref="Half"/>, <see cref="int"/>, <see cref="uint"/>, <see cref="float"/>,
    /// <see cref="double"/>, <see cref="long"/>, <see cref="ulong"/>, and <see cref="string"/>.
    /// </typeparam>
    /// <param name="tensorName">The tensor name to create or replace.</param>
    /// <param name="values">The values to store.</param>
    /// <param name="shape">Optional tensor shape. When omitted, the value is stored as a one-dimensional vector.</param>
    public void Set<T>(string tensorName, T[] values, params ulong[] shape)
    {
        ArgumentNullException.ThrowIfNull(tensorName);
        ArgumentNullException.ThrowIfNull(values);

        if (typeof(T) == typeof(string))
        {
            if (shape.Length != 0)
            {
                throw new ArgumentException("String values do not support an explicit safetensors shape.", nameof(shape));
            }

            SetString(tensorName, (string[])(object)values);
            return;
        }

        var type = typeof(T);
        if (type == typeof(bool))
        {
            var typedValues = (bool[])(object)values;
            SetTensor(tensorName, DataType.Bool, ShapeOrVector(typedValues.Length, shape), BooleansToBytes(typedValues));
        }
        else if (type == typeof(byte))
        {
            var typedValues = (byte[])(object)values;
            SetTensor(tensorName, DataType.U8, ShapeOrVector(typedValues.Length, shape), typedValues.ToArray());
        }
        else if (type == typeof(sbyte))
        {
            var typedValues = (sbyte[])(object)values;
            SetTensor(tensorName, DataType.I8, ShapeOrVector(typedValues.Length, shape), CopyBytes<sbyte>(typedValues));
        }
        else if (type == typeof(short))
        {
            var typedValues = (short[])(object)values;
            SetTensor(tensorName, DataType.I16, ShapeOrVector(typedValues.Length, shape), CopyBytes<short>(typedValues));
        }
        else if (type == typeof(ushort))
        {
            var typedValues = (ushort[])(object)values;
            SetTensor(tensorName, DataType.U16, ShapeOrVector(typedValues.Length, shape), CopyBytes<ushort>(typedValues));
        }
        else if (type == typeof(Half))
        {
            var typedValues = (Half[])(object)values;
            SetTensor(tensorName, DataType.F16, ShapeOrVector(typedValues.Length, shape), CopyBytes<Half>(typedValues));
        }
        else if (type == typeof(int))
        {
            var typedValues = (int[])(object)values;
            SetTensor(tensorName, DataType.I32, ShapeOrVector(typedValues.Length, shape), CopyBytes<int>(typedValues));
        }
        else if (type == typeof(uint))
        {
            var typedValues = (uint[])(object)values;
            SetTensor(tensorName, DataType.U32, ShapeOrVector(typedValues.Length, shape), CopyBytes<uint>(typedValues));
        }
        else if (type == typeof(float))
        {
            var typedValues = (float[])(object)values;
            SetTensor(tensorName, DataType.F32, ShapeOrVector(typedValues.Length, shape), CopyBytes<float>(typedValues));
        }
        else if (type == typeof(double))
        {
            var typedValues = (double[])(object)values;
            SetTensor(tensorName, DataType.F64, ShapeOrVector(typedValues.Length, shape), CopyBytes<double>(typedValues));
        }
        else if (type == typeof(long))
        {
            var typedValues = (long[])(object)values;
            SetTensor(tensorName, DataType.I64, ShapeOrVector(typedValues.Length, shape), CopyBytes<long>(typedValues));
        }
        else if (type == typeof(ulong))
        {
            var typedValues = (ulong[])(object)values;
            SetTensor(tensorName, DataType.U64, ShapeOrVector(typedValues.Length, shape), CopyBytes<ulong>(typedValues));
        }
        else
        {
            throw new NotSupportedException($"Safetensors values of type '{type.FullName}' are not supported.");
        }
    }

    /// <summary>
    /// Removes a named value from the archive.
    /// </summary>
    /// <param name="tensorName">The tensor name to remove.</param>
    /// <returns><see langword="true"/> when a value was removed; otherwise <see langword="false"/>.</returns>
    public bool Remove(string tensorName)
    {
        ArgumentNullException.ThrowIfNull(tensorName);

        return _tensors.Remove(tensorName);
    }

    /// <summary>
    /// Reads a named value and marshals the safetensors tensor payload into a CLR array.
    /// </summary>
    /// <typeparam name="T">
    /// Supported types: <see cref="bool"/>, <see cref="byte"/>, <see cref="sbyte"/>, <see cref="short"/>,
    /// <see cref="ushort"/>, <see cref="Half"/>, <see cref="int"/>, <see cref="uint"/>, <see cref="float"/>,
    /// <see cref="double"/>, <see cref="long"/>, <see cref="ulong"/>, and <see cref="string"/>.
    /// </typeparam>
    /// <param name="tensorName">The tensor name to read.</param>
    /// <returns>The tensor payload decoded as an array of <typeparamref name="T"/>.</returns>
    public T[] Get<T>(string tensorName)
    {
        if (typeof(T) == typeof(string))
        {
            return (T[])(object)GetString(tensorName);
        }

        var tensor = Tensor(tensorName);
        var type = typeof(T);
        if (type == typeof(bool))
        {
            EnsureDataType(tensorName, tensor, DataType.Bool);
            return (T[])(object)ToBooleanArray(tensor.Data.Span);
        }

        if (type == typeof(byte)) return (T[])(object)GetUnmanagedValue<byte>(tensorName, tensor, DataType.U8);
        if (type == typeof(sbyte)) return (T[])(object)GetUnmanagedValue<sbyte>(tensorName, tensor, DataType.I8);
        if (type == typeof(short)) return (T[])(object)GetUnmanagedValue<short>(tensorName, tensor, DataType.I16);
        if (type == typeof(ushort)) return (T[])(object)GetUnmanagedValue<ushort>(tensorName, tensor, DataType.U16);
        if (type == typeof(Half)) return (T[])(object)GetUnmanagedValue<Half>(tensorName, tensor, DataType.F16);
        if (type == typeof(int)) return (T[])(object)GetUnmanagedValue<int>(tensorName, tensor, DataType.I32);
        if (type == typeof(uint)) return (T[])(object)GetUnmanagedValue<uint>(tensorName, tensor, DataType.U32);
        if (type == typeof(float)) return (T[])(object)GetUnmanagedValue<float>(tensorName, tensor, DataType.F32);
        if (type == typeof(double)) return (T[])(object)GetUnmanagedValue<double>(tensorName, tensor, DataType.F64);
        if (type == typeof(long)) return (T[])(object)GetUnmanagedValue<long>(tensorName, tensor, DataType.I64);
        if (type == typeof(ulong)) return (T[])(object)GetUnmanagedValue<ulong>(tensorName, tensor, DataType.U64);

        throw new NotSupportedException($"Safetensors values of type '{type.FullName}' are not supported.");
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
        var tensors = new List<KeyValuePair<string, TensorView>>(_tensors.Count);
        foreach (var name in _tensors.Keys)
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
        foreach (var name in _tensors.Keys)
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

        if (!_tensors.TryGetValue(tensorName, out var tensor))
        {
            throw SafeTensorException.TensorNotFound(tensorName);
        }

        return tensor;
    }

    /// <summary>
    /// Returns the tensor names exposed by this archive in metadata order.
    /// </summary>
    /// <returns>An ordered list of tensor names.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>SafeTensors::names</c>.
    /// </remarks>
    public IReadOnlyList<string> Names() => _tensors.Keys.ToArray();

    /// <summary>
    /// Gets the number of tensors described by the archive metadata.
    /// </summary>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>SafeTensors::len</c>.
    /// </remarks>
    public int Length => _tensors.Count;

    /// <summary>
    /// Gets a value indicating whether the archive contains no tensors.
    /// </summary>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>SafeTensors::is_empty</c>.
    /// </remarks>
    public bool IsEmpty => Length == 0;

    public override string ToString()
    {
        return $"""
            Safetensors(
                Metadata={Indent(FormatMetadataEntries(_metadataEntries), 1)},
                Tensors={Indent(FormatTensorCollection(), 1)}
            )
            """;
    }

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
        IReadOnlyDictionary<string, string>? metadataEntries
    )
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
                })
            );

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

    private static Metadata BuildMetadata(
        IEnumerable<KeyValuePair<string, TensorView>> data,
        IReadOnlyDictionary<string, string>? metadataEntries
    )
    {
        ulong offset = 0;
        var tensors = new List<KeyValuePair<string, TensorInfo>>();
        foreach (var pair in data)
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

        return new Metadata(metadataEntries, tensors);
    }

    private void SetTensor(string tensorName, DataType dataType, ulong[] shape, byte[] data)
    {
        ArgumentNullException.ThrowIfNull(tensorName);
        _tensors[tensorName] = new TensorView(dataType, shape, data);
    }

    private void SetString(string tensorName, string[] values)
    {
        var json = JsonSerializer.Serialize(values);
        var data = Encoding.UTF8.GetBytes(json);
        _tensors[tensorName] = new TensorView(DataType.U8, [(ulong)data.Length], data);
    }

    private string[] GetString(string tensorName)
    {
        var tensor = Tensor(tensorName);
        EnsureDataType(tensorName, tensor, DataType.U8);
        var json = Encoding.UTF8.GetString(tensor.Data.Span);
        return JsonSerializer.Deserialize<string[]>(json) ?? [];
    }

    private IReadOnlyDictionary<string, string>? MetadataEntriesOrNull()
        => _metadataEntries.Count == 0 ? null : _metadataEntries;

    private static ulong[] ShapeOrVector(int valueCount, ulong[] shape)
        => shape.Length == 0 ? [(ulong)valueCount] : shape;

    private static byte[] BooleansToBytes(ReadOnlySpan<bool> values)
    {
        var result = new byte[values.Length];
        for (var i = 0; i < values.Length; i++)
        {
            result[i] = values[i] ? (byte)1 : (byte)0;
        }

        return result;
    }

    private static byte[] CopyBytes<T>(ReadOnlySpan<T> values)
        where T : unmanaged
        => MemoryMarshal.AsBytes(values).ToArray();

    private static T[] GetUnmanagedValue<T>(string tensorName, TensorView tensor, DataType expectedType)
        where T : unmanaged
    {
        EnsureDataType(tensorName, tensor, expectedType);
        return MemoryMarshal.Cast<byte, T>(tensor.Data.Span).ToArray();
    }

    private static void EnsureDataType(string tensorName, TensorView tensor, DataType expectedType)
    {
        if (tensor.DataType != expectedType)
        {
            throw new InvalidOperationException(
                $"Tensor '{tensorName}' has dtype {tensor.DataType.ToWireName()}, not {expectedType.ToWireName()}.");
        }
    }

    private static DataType GetDataType<T>()
        where T : unmanaged
    {
        var type = typeof(T);
        if (type == typeof(bool)) return DataType.Bool;
        if (type == typeof(byte)) return DataType.U8;
        if (type == typeof(sbyte)) return DataType.I8;
        if (type == typeof(short)) return DataType.I16;
        if (type == typeof(ushort)) return DataType.U16;
        if (type == typeof(Half)) return DataType.F16;
        if (type == typeof(int)) return DataType.I32;
        if (type == typeof(uint)) return DataType.U32;
        if (type == typeof(float)) return DataType.F32;
        if (type == typeof(double)) return DataType.F64;
        if (type == typeof(long)) return DataType.I64;
        if (type == typeof(ulong)) return DataType.U64;

        throw new NotSupportedException($"Safetensors values of type '{type.FullName}' are not supported.");
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
        var remainder = value % LENGTH_PREFIX_SIZE;
        return remainder == 0 ? value : checked(value + (LENGTH_PREFIX_SIZE - remainder));
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
            throw SafeTensorException.ValidationOverflow();
        }

        return (int)value;
    }

    private string FormatTensorCollection()
    {
        var items = _tensors.Keys.Select(FormatNamedTensor);
        return FormatCollection(items);
    }

    private string FormatNamedTensor(string name)
    {
        var tensor = Tensor(name);
        return $"{name}: {FormatDataTypeName(tensor.DataType)}{FormatShape(tensor.Shape)} = {FormatTensorData(tensor)}";
    }

    private static string FormatMetadataEntries(IReadOnlyDictionary<string, string>? metadataEntries)
    {
        if (metadataEntries is null || metadataEntries.Count == 0)
        {
            return "[]";
        }

        return FormatCollection(metadataEntries
            .OrderBy(static pair => pair.Key, StringComparer.Ordinal)
            .Select(static pair => $"{pair.Key}={pair.Value}"));
    }

    private static string FormatCollection(IEnumerable<string> values)
    {
        var items = values.ToArray();
        if (items.Length == 0)
        {
            return "[]";
        }

        return $"""
            [
                {Indent(string.Join(",\n", items), 1)}
            ]
            """;
    }

    private static string FormatShape(IReadOnlyList<ulong> shape)
        => $"[{string.Join(", ", shape)}]";

    private static string FormatTensorData(TensorView tensor)
    {
        return tensor.DataType switch
        {
            DataType.Bool => FormatPreview(ToBooleanArray(tensor.Data.Span), static x => x ? "true" : "false"),
            DataType.U8 => FormatPreview(tensor.Data.Span.ToArray(), FormatValue),
            DataType.I8 => FormatPreview(MemoryMarshal.Cast<byte, sbyte>(tensor.Data.Span).ToArray(), FormatValue),
            DataType.I16 => FormatPreview(MemoryMarshal.Cast<byte, short>(tensor.Data.Span).ToArray(), FormatValue),
            DataType.U16 => FormatPreview(MemoryMarshal.Cast<byte, ushort>(tensor.Data.Span).ToArray(), FormatValue),
            DataType.F16 => FormatPreview(ToHalfArray(tensor.Data.Span), FormatValue),
            DataType.Bf16 => FormatPreview(ToBFloat16Array(tensor.Data.Span), FormatValue),
            DataType.I32 => FormatPreview(MemoryMarshal.Cast<byte, int>(tensor.Data.Span).ToArray(), FormatValue),
            DataType.U32 => FormatPreview(MemoryMarshal.Cast<byte, uint>(tensor.Data.Span).ToArray(), FormatValue),
            DataType.F32 => FormatPreview(MemoryMarshal.Cast<byte, float>(tensor.Data.Span).ToArray(), FormatValue),
            DataType.C64 => FormatPreview(ToComplex64Array(tensor.Data.Span), static x => $"({FormatValue(x.Real)}, {FormatValue(x.Imaginary)})"),
            DataType.F64 => FormatPreview(MemoryMarshal.Cast<byte, double>(tensor.Data.Span).ToArray(), FormatValue),
            DataType.I64 => FormatPreview(MemoryMarshal.Cast<byte, long>(tensor.Data.Span).ToArray(), FormatValue),
            DataType.U64 => FormatPreview(MemoryMarshal.Cast<byte, ulong>(tensor.Data.Span).ToArray(), FormatValue),
            _ => FormatPackedPreview(tensor.DataType, tensor.Shape, tensor.Data.Span),
        };
    }

    private static string FormatPackedPreview(DataType dataType, IReadOnlyList<ulong> shape, ReadOnlySpan<byte> data)
    {
        ulong elementCount = 1;
        foreach (var dim in shape)
        {
            elementCount = checked(elementCount * dim);
        }

        return $"<packed {dataType.ToWireName()} x {elementCount}: {FormatPreview(data.ToArray(), FormatValue)}>";
    }

    private static string FormatPreview<T>(IReadOnlyList<T> values, Func<T, string> formatter)
    {
        const int PREVIEW_EDGE_COUNT = 3;

        if (values.Count == 0)
        {
            return "[]";
        }

        if (values.Count <= PREVIEW_EDGE_COUNT * 2)
        {
            return $"[{string.Join(", ", values.Select(formatter))}]";
        }

        var head = values.Take(PREVIEW_EDGE_COUNT).Select(formatter);
        var tail = values.Skip(values.Count - PREVIEW_EDGE_COUNT).Select(formatter);
        return $"[{string.Join(", ", head)}, ... {string.Join(", ", tail)}]";
    }

    private static string FormatValue<T>(T value)
    {
        if (value is null)
        {
            return "null";
        }

        return value switch
        {
            string text => text,
            IFormattable formattable => formattable.ToString(null, CultureInfo.InvariantCulture),
            _ => value.ToString() ?? string.Empty,
        };
    }

    private static string FormatDataTypeName(DataType dataType)
    {
        return dataType switch
        {
            DataType.Bool => nameof(Boolean),
            DataType.U8 => nameof(Byte),
            DataType.I8 => nameof(SByte),
            DataType.I16 => nameof(Int16),
            DataType.U16 => nameof(UInt16),
            DataType.F16 => nameof(Half),
            DataType.Bf16 => "BFloat16",
            DataType.I32 => nameof(Int32),
            DataType.U32 => nameof(UInt32),
            DataType.F32 => nameof(Single),
            DataType.C64 => "Complex64",
            DataType.F64 => nameof(Double),
            DataType.I64 => nameof(Int64),
            DataType.U64 => nameof(UInt64),
            _ => dataType.ToWireName(),
        };
    }

    private static bool[] ToBooleanArray(ReadOnlySpan<byte> data)
        => [.. data.ToArray().Select(static x => x != 0)];

    private static Half[] ToHalfArray(ReadOnlySpan<byte> data)
    {
        var ushortValues = MemoryMarshal.Cast<byte, ushort>(data);
        var result = new Half[ushortValues.Length];
        for (var i = 0; i < ushortValues.Length; i++)
        {
            result[i] = BitConverter.UInt16BitsToHalf(ushortValues[i]);
        }

        return result;
    }

    private static float[] ToBFloat16Array(ReadOnlySpan<byte> data)
    {
        var ushortValues = MemoryMarshal.Cast<byte, ushort>(data);
        var result = new float[ushortValues.Length];
        for (var i = 0; i < ushortValues.Length; i++)
        {
            result[i] = BitConverter.Int32BitsToSingle(ushortValues[i] << 16);
        }

        return result;
    }

    private static (float Real, float Imaginary)[] ToComplex64Array(ReadOnlySpan<byte> data)
    {
        var values = MemoryMarshal.Cast<byte, float>(data);
        var result = new (float Real, float Imaginary)[values.Length / 2];
        for (var i = 0; i < result.Length; i++)
        {
            result[i] = (values[i * 2], values[i * 2 + 1]);
        }

        return result;
    }

    private static string Indent(string text, int tabs)
    {
        var indent = new string(' ', tabs * 4);
        return text.Trim().Replace("\n", $"\n{indent}").Trim();
    }
}
