using System.Text.Json;
using Onnxify.Helpers;
using Onnxify.TorchSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace Onnxify.Examples.Models;

/// <summary>
/// https://github.com/AU-DIS/LSTM_langid/blob/main/src/LSTMLID.py
/// https://github.com/dotnet/TorchSharp/blob/main/test/TorchSharpTest/TestSaveSD.cs
/// </summary>
public class LSTMLIDModel : torch.nn.Module<Tensor, Tensor>
{
    private readonly Dictionary<string, int> _charToIdx;
    private readonly Dictionary<string, int> _langToIdx;
    private readonly int _vocabSize;
    private readonly int _langSetSize;
    private readonly int _numClasses;

    private readonly global::TorchSharp.Modules.Embedding _charEmbeddings;
    private readonly global::TorchSharp.Modules.LSTM _lstm;
    private readonly global::TorchSharp.Modules.Linear _hidden2Lang;
    private readonly torch.Device _device;

    public LSTMLIDModel(
        Dictionary<string, int> charToIdx,
        Dictionary<string, int> langToIdx,
        int numClasses,
        int embeddingDim,
        int hiddenDim,
        int layers
    ) : base("LSTMLIDModel")
    {
        _charToIdx = charToIdx;
        _langToIdx = langToIdx;
        _vocabSize = charToIdx.Count;
        _langSetSize = langToIdx.Count;
        _numClasses = numClasses;

        _device = torch.CPU;

        _charEmbeddings = torch.nn.Embedding(_vocabSize, embeddingDim, padding_idx: charToIdx["PAD"]);
        _lstm = torch.nn.LSTM(embeddingDim, hiddenDim, numLayers: layers, bidirectional: true, batchFirst: true);
        _hidden2Lang = torch.nn.Linear(hiddenDim * 2, _langSetSize);

        RegisterComponents();

        if (_device.type != DeviceType.CPU)
        {
            this.to(_device);
        }
    }

    public override Tensor forward(Tensor input)
    {
        var x = _charEmbeddings.forward(input);
        var (lstm, _, _) = _lstm.forward(x);
        var linear = _hidden2Lang.forward(lstm);
        var logit = torch.sum(linear, 1);
        return logit;
    }

    public void SaveModel(string modelPath)
    {
        // var scriptedModel = torch.jit.load(model);
        // torch.jit.save(this, "mymodel_scripted.pt");
        this.save(modelPath);
    }

    public OnnxModel Export()
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions());
        var graph = model.Graph;

        var input = graph.AddInput(
            name: "input",
            type: OnnxTensorType.Create<long>(["batch_size", "seq_len"])
        );

        var x = _charEmbeddings.Export(graph, input);
        x = _lstm.Export(graph, x).Y ?? throw new Exception();
        x = _hidden2Lang.Export(graph, x);

        x = graph.ReduceSum(
            name: "sum_logits",
            options: new ReduceSumInputOptions
            {
                Data = x,
                Axes = graph.AddTensor<long>("sum_logits_axes", [1], [1]),
                Keepdims = 0,
            }
        );

        var outputEdge = graph.AddEdge("output");

        graph.Identity(
            name: "output_identity",
            options: new IdentityInputOutputOptions
            {
                Input = x,
                Output = outputEdge,
            }
        );

        graph.AddOutput(
            name: "output",
            type: OnnxTensorType.Create<float>(["batch_size", _langSetSize])
        );

        model.AddMetadataProps(
            key: "char_to_idx",
            value: JsonSerializer.Serialize(_charToIdx)
        );

        model.AddMetadataProps(
            key: "lang_to_idx",
            value: JsonSerializer.Serialize(_langToIdx)
        );

        model.AddMetadataProps(
            key: "idx_to_lang",
            value: JsonSerializer.Serialize(
                _langToIdx.OrderBy(x => x.Value).Select(x => x.Key).ToArray()
            )
        );

        return model;
    }
}
