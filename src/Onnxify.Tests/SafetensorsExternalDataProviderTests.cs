using Onnxify.Safetensors;
using Onnxify.TorchSharp;
using SafeTensorsArchive = Onnxify.Safetensors.SafeTensors;

namespace Onnxify.Tests;

public sealed class SafetensorsExternalDataProviderTests
{
    [Fact]
    public void WriteTensor_AndReadTensorValue_RoundTripsSingleTensorArchive()
    {
        var model = OnnxModel.Create();
        var tensor = model.Graph.AddTensor("weights", [2, 2], [1.0f, 2.0f, 3.5f, 4.5f]);
        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.safetensors");

        try
        {
            SafetensorsExternalDataProvider.Instance.WriteTensor(
                tensor,
                path,
                metadata: new Dictionary<string, string> { ["format"] = "onnx" }
            );

            var raw = File.ReadAllBytes(path);
            var archive = SafeTensorsArchive.Deserialize(raw);

            Assert.Equal(["weights"], archive.Names());
            var view = archive.Tensor("weights");
            Assert.Equal(DataType.F32, view.DataType);
            Assert.Equal([2UL, 2UL], view.Shape);

            var values = SafetensorsExternalDataProvider.Instance.ReadTensorValue<float>(
                path,
                offset: 0,
                length: -1
            );

            Assert.Equal([1.0f, 2.0f, 3.5f, 4.5f], values);
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
    public void ReadTensorValue_MultiTensorArchive_Throws()
    {
        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.safetensors");
        var tensorBytes = BitConverter.GetBytes(1.0f);

        try
        {
            SafeTensorsArchive.SerializeToFile(
                data:
                [
                    new KeyValuePair<string, TensorView>("left", new TensorView(DataType.F32, [1], tensorBytes)),
                    new KeyValuePair<string, TensorView>("right", new TensorView(DataType.F32, [1], tensorBytes)),
                ],
                metadata: null,
                path: path
            );

            var exception = Assert.Throws<InvalidOperationException>(
                () => SafetensorsExternalDataProvider.Instance.ReadTensorValue<float>(
                    path,
                    offset: 0,
                    length: -1
                )
            );

            Assert.Contains("Expected exactly one tensor", exception.Message);
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
    public void WriteTensor_StringTensor_ThrowsNotSupportedException()
    {
        var model = OnnxModel.Create();
        var tensor = model.Graph.AddTensor("tokens", [2], ["a", "b"]);
        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.safetensors");

        try
        {
            var exception = Assert.Throws<NotSupportedException>(
                () => SafetensorsExternalDataProvider.Instance.WriteTensor(tensor, path)
            );

            Assert.Contains("string tensors", exception.Message);
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }
}
