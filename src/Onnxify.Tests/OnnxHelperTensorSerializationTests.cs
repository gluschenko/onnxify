using Onnxify.Data.Numerics;
using Onnxify.Helpers;

namespace Onnxify.Tests;

public sealed class OnnxHelperTensorSerializationTests
{
    [Fact]
    public void SerializeTensor_AndDeserializeTensor_RoundTripNumericTensor()
    {
        var model = OnnxModel.Create();
        var tensor = model.Graph.AddTensor("weights", [2, 2], [1.0f, 2.0f, 3.5f, 4.5f]);

        var bytes = tensor.SerializeTensor();
        var restored = OnnxHelper.DeserializeTensor(bytes);

        var typed = Assert.IsType<OnnxTensor<float>>(restored);
        Assert.Equal("weights", typed.Name);
        Assert.Equal([2L, 2L], typed.Shape);
        Assert.Equal([1.0f, 2.0f, 3.5f, 4.5f], typed.Value.ToArray());
    }

    [Fact]
    public void SerializeTensor_AndDeserializeTensor_RoundTripStringTensor()
    {
        var model = OnnxModel.Create();
        var tensor = model.Graph.AddTensor("tokens", [2], ["hello", "world"]);

        var bytes = tensor.SerializeTensor();
        var restored = OnnxHelper.DeserializeTensor(bytes);

        var typed = Assert.IsType<OnnxTensor<string>>(restored);
        Assert.Equal("tokens", typed.Name);
        Assert.Equal([2L], typed.Shape);
        Assert.Equal(["hello", "world"], typed.Value.ToArray());
    }

    [Fact]
    public void DeserializeTensor_TrimsPackedTensorPadding()
    {
        var model = OnnxModel.Create();
        var tensor = model.Graph.AddTensor(
            "packed",
            [5],
            [new UInt2(1), new UInt2(2), new UInt2(3), new UInt2(0), new UInt2(1)]
        );

        var bytes = tensor.SerializeTensor();
        var restored = OnnxHelper.DeserializeTensor(bytes);

        var typed = Assert.IsType<OnnxTensor<UInt2>>(restored);
        Assert.Equal([5L], typed.Shape);
        Assert.Equal(5, typed.Value.Count());
        Assert.Equal([1, 2, 3, 0, 1], typed.Value.Select(static x => (int)x.Value).ToArray());
    }
}
