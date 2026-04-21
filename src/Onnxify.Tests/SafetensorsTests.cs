using System.Buffers.Binary;
using System.Text;
using Onnxify.Safetensors;
using SafeTensors = Onnxify.Safetensors.SafeTensors;

namespace Onnxify.Tests;

public class SafetensorsTests
{
    [Fact]
    public void Serialize_F32_MatchesUpstreamBytes()
    {
        var data = FloatsToBytes(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f);
        var metadata = new Dictionary<string, TensorView>
        {
            ["attn.0"] = new(DataType.F32, new ulong[] { 1, 2, 3 }, data),
        };

        var serialized = SafeTensors.Serialize(metadata);

        Assert.Equal(
            new byte[]
            {
                64, 0, 0, 0, 0, 0, 0, 0, 123, 34, 97, 116, 116, 110, 46, 48, 34, 58, 123, 34, 100,
                116, 121, 112, 101, 34, 58, 34, 70, 51, 50, 34, 44, 34, 115, 104, 97, 112, 101, 34,
                58, 91, 49, 44, 50, 44, 51, 93, 44, 34, 100, 97, 116, 97, 95, 111, 102, 102, 115,
                101, 116, 115, 34, 58, 91, 48, 44, 50, 52, 93, 125, 125, 0, 0, 0, 0, 0, 0, 128, 63,
                0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0, 160, 64,
            },
            serialized
        );

        var parsed = SafeTensors.Deserialize(serialized);
        var tensor = parsed.Tensor("attn.0");
        Assert.Equal([1, 2, 3], tensor.Shape);
        Assert.Equal(DataType.F32, tensor.DataType);
        Assert.Equal(data, tensor.Data.ToArray());
    }

    [Fact]
    public void Serialize_F4_MatchesUpstreamBytes()
    {
        var metadata = new Dictionary<string, TensorView>
        {
            ["attn.0"] = new(DataType.F4, new ulong[] { 1, 2 }, new byte[] { 0 }),
        };

        var serialized = SafeTensors.Serialize(metadata);

        Assert.Equal(
            new byte[]
            {
                64, 0, 0, 0, 0, 0, 0, 0, 123, 34, 97, 116, 116, 110, 46, 48, 34, 58, 123, 34, 100,
                116, 121, 112, 101, 34, 58, 34, 70, 52, 34, 44, 34, 115, 104, 97, 112, 101, 34, 58,
                91, 49, 44, 50, 93, 44, 34, 100, 97, 116, 97, 95, 111, 102, 102, 115, 101, 116,
                115, 34, 58, 91, 48, 44, 49, 93, 125, 125, 32, 32, 32, 32, 0,
            },
            serialized
        );

        var parsed = SafeTensors.Deserialize(serialized);
        var tensor = parsed.Tensor("attn.0");
        Assert.Equal([1, 2], tensor.Shape);
        Assert.Equal(DataType.F4, tensor.DataType);
        Assert.Equal(new byte[] { 0 }, tensor.Data.ToArray());
    }

    [Fact]
    public void TensorView_F4_Misaligned_Throws()
    {
        var exception = Assert.Throws<SafeTensorException>(
            () => new TensorView(DataType.F4, [1, 3], new byte[] { 0, 1 })
        );

        Assert.Equal(SafeTensorErrorCode.MisalignedSlice, exception.Code);
    }

    [Fact]
    public void TensorView_F4_InvalidLength_Throws()
    {
        var exception = Assert.Throws<SafeTensorException>(
            () => new TensorView(DataType.F4, new ulong[] { 1, 2 }, new byte[] { 0, 1 }));

        Assert.Equal(SafeTensorErrorCode.InvalidTensorView, exception.Code);
        Assert.Equal(DataType.F4, exception.DataType);
        Assert.Equal([1, 2], exception.Shape);
        Assert.Equal(2UL, exception.ByteLength);
    }

    [Fact]
    public void Serialize_Empty_MatchesUpstreamBytes()
    {
        var tensors = Array.Empty<KeyValuePair<string, TensorView>>();

        var serialized = SafeTensors.Serialize(tensors);
        Assert.Equal(
            new byte[] { 8, 0, 0, 0, 0, 0, 0, 0, 123, 125, 32, 32, 32, 32, 32, 32 },
            serialized
        );
        Assert.Equal(0, SafeTensors.Deserialize(serialized).Length);

        var serializedWithMetadata = SafeTensors.Serialize(
            tensors,
            new Dictionary<string, string> { ["framework"] = "pt" });

        Assert.Equal(
            new byte[]
            {
                40, 0, 0, 0, 0, 0, 0, 0, 123, 34, 95, 95, 109, 101, 116, 97, 100, 97, 116, 97, 95,
                95, 34, 58, 123, 34, 102, 114, 97, 109, 101, 119, 111, 114, 107, 34, 58, 34, 112,
                116, 34, 125, 125, 32, 32, 32, 32, 32,
            },
            serializedWithMetadata
        );

        Assert.Equal(0, SafeTensors.Deserialize(serializedWithMetadata).Length);
    }

    [Fact]
    public void Serialize_ForcedAlignment_MatchesUpstreamBytes()
    {
        var data = FloatsToBytes(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f);
        var metadata = new Dictionary<string, TensorView>
        {
            ["attn0"] = new(DataType.F32, [1, 1, 2, 3], data),
        };

        var serialized = SafeTensors.Serialize(metadata);

        Assert.Equal(
            new byte[]
            {
                72, 0, 0, 0, 0, 0, 0, 0, 123, 34, 97, 116, 116, 110, 48, 34, 58, 123, 34, 100, 116,
                121, 112, 101, 34, 58, 34, 70, 51, 50, 34, 44, 34, 115, 104, 97, 112, 101, 34, 58,
                91, 49, 44, 49, 44, 50, 44, 51, 93, 44, 34, 100, 97, 116, 97, 95, 111, 102, 102,
                115, 101, 116, 115, 34, 58, 91, 48, 44, 50, 52, 93, 125, 125, 32, 32, 32, 32, 32,
                32, 32, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0,
                160, 64,
            },
            serialized
        );

        var parsed = SafeTensors.Deserialize(serialized);
        var tensor = parsed.Tensor("attn0");
        Assert.Equal(0, tensor.Data.Span.Length % (tensor.DataType.Bitsize() / 8));
    }

    [Fact]
    public void SerializeToFile_RoundTrips()
    {
        var data = FloatsToBytes(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f);
        var metadata = new Dictionary<string, TensorView>
        {
            ["attn.0"] = new(DataType.F32, [1, 2, 3], data),
        };

        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.safetensors");
        try
        {
            SafeTensors.SerializeToFile(metadata, null, path);
            var raw = File.ReadAllBytes(path);
            var loaded = SafeTensors.Deserialize(raw);
            Assert.Equal(["attn.0"], loaded.Names());
            Assert.Equal(data, loaded.Tensor("attn.0").Data.ToArray());
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    [Fact]
    public void Slice_FromTensorRsExamples_MatchesUpstreamOutput()
    {
        var data = FloatsToBytes(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f);
        var metadata = new Dictionary<string, TensorView>
        {
            ["attn.0"] = new(DataType.F32, [1, 2, 3], data),
        };

        var parsed = SafeTensors.Deserialize(SafeTensors.Serialize(metadata));
        var tensor = parsed.Tensor("attn.0");

        var outBuffer = FlattenBytes(tensor.Slice(All(), Narrow(0, 1)));
        Assert.Equal(new byte[] { 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 64 }, outBuffer);
        Assert.Equal(FloatsToBytes(0.0f, 1.0f, 2.0f), outBuffer);

        outBuffer = FlattenBytes(tensor.Slice(All(), All(), Narrow(0, 1)));
        Assert.Equal(new byte[] { 0, 0, 0, 0, 0, 0, 64, 64 }, outBuffer);
        Assert.Equal(FloatsToBytes(0.0f, 3.0f), outBuffer);
    }

    [Fact]
    public void Deserialize_EmptyShapesAllowed()
    {
        var serialized = BuildBuffer(
            "{\"test\":{\"dtype\":\"I32\",\"shape\":[],\"data_offsets\":[0,4]}}",
            [0, 0, 0, 0]
        );

        var loaded = SafeTensors.Deserialize(serialized);

        Assert.Equal(["test"], loaded.Names());
        var tensor = loaded.Tensor("test");
        Assert.Empty(tensor.Shape);
        Assert.Equal(DataType.I32, tensor.DataType);
        Assert.Equal(new byte[] { 0, 0, 0, 0 }, tensor.Data.ToArray());
    }

    [Fact]
    public void Deserialize_BasicTensor_MatchesUpstream()
    {
        var serialized = BuildBuffer(
            "{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}",
            new byte[16]);

        var loaded = SafeTensors.Deserialize(serialized);

        Assert.Equal(1, loaded.Length);
        Assert.Equal(["test"], loaded.Names());
        var tensor = loaded.Tensor("test");
        Assert.Equal([2, 2], tensor.Shape);
        Assert.Equal(DataType.I32, tensor.DataType);
        Assert.Equal(new byte[16], tensor.Data.ToArray());
    }

    [Fact]
    public void Deserialize_JsonAttack_ThrowsInvalidOffset()
    {
        var header = string.Join(
            ",",
            Enumerable.Range(0, 10).Select(i =>
                $"\"weight_{i}\":{{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}"));

        var serialized = BuildBuffer($"{{{header}}}", new byte[16]);

        var exception = Assert.Throws<SafeTensorException>(() => SafeTensors.Deserialize(serialized));
        Assert.Equal(SafeTensorErrorCode.InvalidOffset, exception.Code);
        Assert.StartsWith("weight_", exception.TensorName);
    }

    [Fact]
    public void Deserialize_MetadataIncompleteBuffer_Throws()
    {
        var baseBuffer = BuildBuffer(
            "{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}",
            new byte[16]);

        var withExtraData = baseBuffer.Concat(Encoding.ASCII.GetBytes("extra_bogus_data_for_polyglot_file")).ToArray();
        var exception = Assert.Throws<SafeTensorException>(() => SafeTensors.Deserialize(withExtraData));
        Assert.Equal(SafeTensorErrorCode.MetadataIncompleteBuffer, exception.Code);

        var missingData = BuildBuffer(
            "{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}",
            new byte[14]
        );

        exception = Assert.Throws<SafeTensorException>(() => SafeTensors.Deserialize(missingData));
        Assert.Equal(SafeTensorErrorCode.MetadataIncompleteBuffer, exception.Code);
    }

    [Fact]
    public void Deserialize_HeaderTooLarge_Throws()
    {
        var serialized = BuildBufferWithHeaderLength(
            uint.MaxValue,
            Encoding.ASCII.GetBytes("{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}"),
            new byte[16]
        );

        var exception = Assert.Throws<SafeTensorException>(() => SafeTensors.Deserialize(serialized));
        Assert.Equal(SafeTensorErrorCode.HeaderTooLarge, exception.Code);
    }

    [Fact]
    public void Deserialize_HeaderTooSmall_Throws()
    {
        var exception = Assert.Throws<SafeTensorException>(() => SafeTensors.Deserialize(Array.Empty<byte>()));
        Assert.Equal(SafeTensorErrorCode.HeaderTooSmall, exception.Code);
    }

    [Fact]
    public void Deserialize_InvalidHeaderLength_Throws()
    {
        var exception = Assert.Throws<SafeTensorException>(
            () => SafeTensors.Deserialize(new byte[] { 60, 0, 0, 0, 0, 0, 0, 0 })
        );

        Assert.Equal(SafeTensorErrorCode.InvalidHeaderLength, exception.Code);
    }

    [Fact]
    public void Deserialize_InvalidHeaderNonUtf8_Throws()
    {
        var exception = Assert.Throws<SafeTensorException>(
            () => SafeTensors.Deserialize(new byte[] { 1, 0, 0, 0, 0, 0, 0, 0, 255 })
        );

        Assert.Equal(SafeTensorErrorCode.InvalidHeader, exception.Code);
    }

    [Fact]
    public void Deserialize_InvalidHeaderNotJson_Throws()
    {
        var exception = Assert.Throws<SafeTensorException>(
            () => SafeTensors.Deserialize(new byte[] { 1, 0, 0, 0, 0, 0, 0, 0, 123 })
        );

        Assert.Equal(SafeTensorErrorCode.InvalidHeaderDeserialization, exception.Code);
    }

    [Fact]
    public void Deserialize_WhitespacePaddedHeader_AllowsLeadingAndTrailingWhitespace()
    {
        var trailing = new byte[] { 6, 0, 0, 0, 0, 0, 0, 0, 123, 125, 13, 32, 9, 10 };
        var loaded = SafeTensors.Deserialize(trailing);
        Assert.Equal(0, loaded.Length);

        var leading = new byte[] { 6, 0, 0, 0, 0, 0, 0, 0, 9, 10, 123, 125, 13, 32 };
        loaded = SafeTensors.Deserialize(leading);
        Assert.Equal(0, loaded.Length);
    }

    [Fact]
    public void Deserialize_ZeroSizedTensor_AllowsEmptyPayload()
    {
        var serialized = BuildBuffer(
            "{\"test\":{\"dtype\":\"I32\",\"shape\":[2,0],\"data_offsets\":[0,0]}}",
            []
        );

        var loaded = SafeTensors.Deserialize(serialized);
        var tensor = loaded.Tensor("test");

        Assert.Equal(["test"], loaded.Names());
        Assert.Equal([2, 0], tensor.Shape);
        Assert.Equal(DataType.I32, tensor.DataType);
        Assert.Empty(tensor.Data.ToArray());
    }

    [Fact]
    public void Deserialize_InvalidInfo_Throws()
    {
        var serialized = BuildBuffer(
            "{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,4]}}",
            new byte[4]
        );

        var exception = Assert.Throws<SafeTensorException>(() => SafeTensors.Deserialize(serialized));
        Assert.Equal(SafeTensorErrorCode.TensorInvalidInfo, exception.Code);
    }

    [Fact]
    public void Deserialize_ValidationOverflow_Throws()
    {
        var serialized = BuildBuffer(
            "{\"test\":{\"dtype\":\"I32\",\"shape\":[2,18446744073709551614],\"data_offsets\":[0,16]}}",
            new byte[16]
        );

        var exception = Assert.Throws<SafeTensorException>(() => SafeTensors.Deserialize(serialized));
        Assert.Equal(SafeTensorErrorCode.ValidationOverflow, exception.Code);

        serialized = BuildBuffer(
            "{\"test\":{\"dtype\":\"I32\",\"shape\":[2,9223372036854775807],\"data_offsets\":[0,16]}}",
            new byte[16]
        );

        exception = Assert.Throws<SafeTensorException>(() => SafeTensors.Deserialize(serialized));
        Assert.Equal(SafeTensorErrorCode.ValidationOverflow, exception.Code);
    }

    [Fact]
    public void SliceHelpers_MatchUpstream()
    {
        var data = FloatsToBytes(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f);
        var tensor = new TensorView(DataType.F32, [1, 2, 3], data);

        var iterator = tensor.Slice(All());
        Assert.Equal(24UL, iterator.RemainingByteLength);
        Assert.Equal([1, 2, 3], iterator.NewShape);

        iterator = tensor.Slice(All(), Narrow(0, 1));
        Assert.Equal(12UL, iterator.RemainingByteLength);
        Assert.Equal([1, 1, 3], iterator.NewShape);
    }

    [Fact]
    public void SliceFp4Simple_MatchUpstream()
    {
        var tensor = new TensorView(DataType.F4, [1, 2, 2], new byte[] { 0, 1 });

        var iterator = tensor.Slice(All());
        Assert.Equal(2UL, iterator.RemainingByteLength);
        Assert.Equal([1, 2, 2], iterator.NewShape);

        iterator = tensor.Slice(All(), Narrow(0, 1));
        Assert.Equal(1UL, iterator.RemainingByteLength);
        Assert.Equal([1, 1, 2], iterator.NewShape);
    }

    [Fact]
    public void SliceFp4Misaligned_Throws()
    {
        var tensor = new TensorView(DataType.F4, new ulong[] { 1, 2 }, new byte[] { 0 });

        var iterator = tensor.Slice(All());
        Assert.Equal(1UL, iterator.RemainingByteLength);
        Assert.Equal([1, 2], iterator.NewShape);

        var exception = Assert.Throws<InvalidSliceException>(() => tensor.Slice(All(), Narrow(0, 1)));
        Assert.Equal(InvalidSliceErrorCode.MisalignedSlice, exception.Code);
    }

    [Fact]
    public void SliceDummyCases_MatchUpstream()
    {
        var data = FloatsToBytes(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f);
        var tensor = new TensorView(DataType.F32, [1, 2, 3], data);

        AssertSingleChunk(tensor.Slice(All()), data);
        AssertSingleChunk(tensor.Slice(All(), All()), data);
        AssertSingleChunk(tensor.Slice(All(), All()), data);
        AssertSingleChunk(tensor.Slice(All(), All(), All()), data);

        var exception = Assert.Throws<InvalidSliceException>(() => tensor.Slice(All(), All(), All(), All()));
        Assert.Equal(InvalidSliceErrorCode.TooManySlices, exception.Code);
    }

    [Fact]
    public void SliceVariety_MatchesUpstream()
    {
        var data = FloatsToBytes(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f);
        var tensor = new TensorView(DataType.F32, [1, 2, 3], data);

        AssertSingleChunk(tensor.Slice(Narrow(0, 1)), data);
        AssertSingleChunk(tensor.Slice(All(), Narrow(0, 1)), data[..12]);
        Assert.Equal(new[] { data[0..4], data[12..16] }, CollectChunks(tensor.Slice(All(), All(), Narrow(0, 1))));
        AssertSingleChunk(tensor.Slice(All(), Narrow(1, 2), Narrow(0, 1)), data[12..16]);
    }

    [Fact]
    public void SliceVariety2_MatchesUpstream()
    {
        var data = FloatsToBytes(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f);
        var tensor = new TensorView(DataType.F32, [2, 3], data);

        Assert.Equal(
            [data[4..12], data[16..24]],
            CollectChunks(tensor.Slice(All(), Narrow(1, 3)))
        );
    }

    [Fact]
    public void SliceSelect_MatchesUpstream()
    {
        var data = FloatsToBytes(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f);
        var tensor = new TensorView(DataType.F32, [2, 3], data);

        AssertSingleChunk(tensor.Slice(Select(1), Narrow(1, 3)), data[16..24]);
        AssertSingleChunk(tensor.Slice(Select(0), Narrow(1, 3)), data[4..12]);
        AssertSingleChunk(tensor.Slice(Narrow(1, 2), Select(0)), data[12..16]);
    }

    [Fact]
    public void Slice_InvalidRange_MatchesUpstream()
    {
        var data = FloatsToBytes(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f);
        var tensor = new TensorView(DataType.F32, [2, 3], data);

        var exception = Assert.Throws<InvalidSliceException>(() => tensor.Slice(Select(1), Narrow(1, 4)));
        Assert.Equal(InvalidSliceErrorCode.SliceOutOfRange, exception.Code);
        Assert.Equal(1, exception.DimensionIndex);
        Assert.Equal(3UL, exception.Asked);
        Assert.Equal(3UL, exception.DimensionSize);

        exception = Assert.Throws<InvalidSliceException>(() => tensor.Slice(Select(1), InvalidNarrow(3, 2)));
        Assert.Equal(InvalidSliceErrorCode.SliceOutOfRange, exception.Code);
        Assert.Equal(1, exception.DimensionIndex);
        Assert.Equal(3UL, exception.Asked);
        Assert.Equal(3UL, exception.DimensionSize);

        exception = Assert.Throws<InvalidSliceException>(() => tensor.Slice(Select(1), Select(1), Select(1)));
        Assert.Equal(InvalidSliceErrorCode.TooManySlices, exception.Code);
    }

    private static byte[] FloatsToBytes(params float[] values)
    {
        var bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static byte[] IntsToBytes(params int[] values)
    {
        var bytes = new byte[values.Length * sizeof(int)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static byte[] BuildBuffer(string headerJson, byte[] data)
    {
        var headerBytes = Encoding.UTF8.GetBytes(headerJson);
        return BuildBufferWithHeaderLength((ulong)headerBytes.Length, headerBytes, data);
    }

    private static byte[] BuildBufferWithHeaderLength(ulong headerLength, byte[] headerBytes, byte[] data)
    {
        var buffer = new byte[8 + headerBytes.Length + data.Length];
        BinaryPrimitives.WriteUInt64LittleEndian(buffer.AsSpan(0, 8), headerLength);
        headerBytes.CopyTo(buffer, 8);
        data.CopyTo(buffer, 8 + headerBytes.Length);
        return buffer;
    }

    private static NarrowTensorIndexer All()
        => new(TensorBounds.Unbounded(), TensorBounds.Unbounded());

    private static NarrowTensorIndexer Narrow(ulong start, ulong stopExclusive)
        => new(TensorBounds.Included(start), TensorBounds.Excluded(stopExclusive));

    private static NarrowTensorIndexer InvalidNarrow(ulong start, ulong stopExclusive)
        => new(TensorBounds.Included(start), TensorBounds.Excluded(stopExclusive));

    private static SelectTensorIndexer Select(ulong index)
        => new(index);

    private static byte[] FlattenBytes(IEnumerable<ReadOnlyMemory<byte>> chunks)
        => chunks.SelectMany(static chunk => chunk.ToArray()).ToArray();

    private static byte[][] CollectChunks(IEnumerable<ReadOnlyMemory<byte>> chunks)
        => chunks.Select(static chunk => chunk.ToArray()).ToArray();

    private static void AssertSingleChunk(SliceIterator iterator, ReadOnlySpan<byte> expected)
    {
        var chunks = CollectChunks(iterator);
        Assert.Single(chunks);
        Assert.Equal(expected.ToArray(), chunks[0]);
    }
}
