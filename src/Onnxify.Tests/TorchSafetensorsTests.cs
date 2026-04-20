using Onnxify.TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Onnxify.Tests;

public sealed class TorchSafetensorsTests
{
    [Fact]
    public void SaveAndLoad_RoundTripsTensorDictionary()
    {
        if (!TorchSharpRuntimeAvailable())
        {
            return;
        }

        using var weight = tensor(
            new float[] { 1f, 2f, 3f, 4f },
            new long[] { 2, 2 });
        using var bias = tensor(
            new long[] { 5, 6, 7 },
            dtype: ScalarType.Int64);

        var state = new Dictionary<string, Tensor>(StringComparer.Ordinal)
        {
            ["weight"] = weight,
            ["bias"] = bias,
        };

        using var loaded = new TensorScope(TorchSafetensors.Load(
            TorchSafetensors.Save(state, new Dictionary<string, string> { ["framework"] = "pt" })));

        Assert.Equal(["bias", "weight"], loaded.State.Keys.OrderBy(static x => x, StringComparer.Ordinal).ToArray());
        Assert.Equal(new long[] { 5, 6, 7 }, loaded.State["bias"].data<long>().ToArray());
        Assert.Equal(new float[] { 1f, 2f, 3f, 4f }, loaded.State["weight"].data<float>().ToArray());
        Assert.Equal([2L, 2L], loaded.State["weight"].shape);
    }

    [Fact]
    public void Save_RejectsSharedTensorDictionary()
    {
        if (!TorchSharpRuntimeAvailable())
        {
            return;
        }

        using var shared = tensor(
            new float[] { 1f, 2f, 3f, 4f },
            new long[] { 2, 2 });

        var state = new Dictionary<string, Tensor>(StringComparer.Ordinal)
        {
            ["first"] = shared,
            ["second"] = shared,
        };

        var exception = Assert.Throws<InvalidOperationException>(() => TorchSafetensors.Save(state));
        Assert.Contains("share memory", exception.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("SaveModel", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Save_AcceptsModuleParameterTensor()
    {
        if (!TorchSharpRuntimeAvailable())
        {
            return;
        }

        using var linear = Linear(4, 3);
        var state = new Dictionary<string, Tensor>(StringComparer.Ordinal)
        {
            ["weight"] = linear.weight,
        };

        using var loaded = new TensorScope(TorchSafetensors.Load(TorchSafetensors.Save(state)));

        Assert.Equal(linear.weight.data<float>().ToArray(), loaded.State["weight"].data<float>().ToArray());
        Assert.Equal(linear.weight.shape, loaded.State["weight"].shape);
    }

    [Fact]
    public void SaveModelAndLoadModel_RoundTripLinearWeights()
    {
        if (!TorchSharpRuntimeAvailable())
        {
            return;
        }

        using var source = Linear(4, 3);
        using var target = Linear(4, 3);
        Assert.NotNull(source.bias);
        Assert.NotNull(target.bias);
        var sourceBias = source.bias!;
        var targetBias = target.bias!;

        source.weight.fill_(1.5);
        sourceBias.fill_(-0.25);
        target.weight.zero_();
        targetBias.zero_();

        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.safetensors");
        try
        {
            TorchSafetensors.SaveModel(source, path);
            var result = TorchSafetensors.LoadModel(target, path, strict: true);

            Assert.Empty(result.Missing);
            Assert.Empty(result.Unexpected);
            Assert.Equal(source.weight.data<float>().ToArray(), target.weight.data<float>().ToArray());
            Assert.Equal(sourceBias.data<float>().ToArray(), targetBias.data<float>().ToArray());
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
    public void LoadModel_StrictFalse_ReturnsMissingAndUnexpectedKeys()
    {
        if (!TorchSharpRuntimeAvailable())
        {
            return;
        }

        using var source = Linear(4, 3);
        using var target = Sequential(
            ("proj", Linear(4, 3)),
            ("head", Linear(3, 2)));

        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.safetensors");
        try
        {
            TorchSafetensors.SaveModel(source, path);
            var result = TorchSafetensors.LoadModel(target, path, strict: false);

            Assert.Equal(
                ["bias", "weight"],
                result.Unexpected.OrderBy(static x => x, StringComparer.Ordinal).ToArray());
            Assert.Equal(
                ["head.bias", "head.weight", "proj.bias", "proj.weight"],
                result.Missing.OrderBy(static x => x, StringComparer.Ordinal).ToArray());
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    private static bool TorchSharpRuntimeAvailable()
    {
        try
        {
            using var probe = tensor(new float[] { 0f });
            return true;
        }
        catch (Exception ex) when (
            ex is DllNotFoundException
            || ex is TypeInitializationException
            || ex is NotSupportedException)
        {
            return false;
        }
    }

    private sealed class TensorScope : IDisposable
    {
        public TensorScope(Dictionary<string, Tensor> state)
        {
            State = state;
        }

        public Dictionary<string, Tensor> State { get; }

        public void Dispose()
        {
            foreach (var tensor in State.Values)
            {
                tensor.Dispose();
            }
        }
    }
}
