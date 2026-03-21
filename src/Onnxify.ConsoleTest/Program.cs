using System.Globalization;
using System.Text;
using System.Text.Json;
using Google.Protobuf;
using Onnx;
using Onnxify;
using TorchSharp.Modules;

namespace Onnxify.ConsoleTest
{
    internal class Program
    {
        static void Main(string[] args)
        {
            CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;
            Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
            Console.Title = nameof(Onnxify);
            Console.InputEncoding = Encoding.Unicode;
            Console.OutputEncoding = Encoding.Unicode;

            Test0();
            Test1();
            Test2();

            Console.WriteLine("Press any key to pay respect...");
            Console.ReadKey();
        }

        static void Test0()
        {
            var model = OnnxModel.Create(new OnnxModelCreationOptions());

            var conv1_w = model.Graph.AddTensor<float>(
                name: "conv1_w",
                shape: [64, 3, 11, 11],
                value: new float[64 * 3 * 11 * 11]
            );

            var conv1_b = model.Graph.AddTensor<float>(
                name: "conv1_b",
                shape: [1, 3, 128, 128],
                value: new float[1 * 3 * 128 * 128]
            );

            var conv1_in = model.Graph.AddEdge("conv1_in");
            var conv1_out = model.Graph.AddEdge("conv1_out");

            var conv = new Operators.Conv(
                name: "conv1",
                x: conv1_in,
                w: conv1_w,
                b: conv1_b,
                y: conv1_out
            );

            return;
        }

        static void Test1()
        {
            var inputPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets", "bvlcalexnet-12-qdq.onnx");
            var outputPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets", "bvlcalexnet-12-qdq__test.onnx");
            var model = OnnxModel.FromFile(inputPath);

            var text = model.ToString();
            Console.WriteLine(text);

            model.Save(outputPath, true);
            return;
        }

        static void Test2()
        {
            var outputPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets", "test.onnx");
            var model = OnnxModel.Create(new OnnxModelCreationOptions());

            var input = model.Graph.AddInput(
                name: "input_0",
                type: OnnxTensorType.Create<float>([256, 128])
            );

            var output = model.Graph.AddOutput(
                name: "probs_0",
                type: OnnxTensorType.Create<float>([128, 64])
            );

            var conv1_w = model.Graph.AddTensor<float>(
                name: "conv1_w",
                shape: [64, 3, 11, 11],
                value: new float[64 * 3 * 11 * 11]
            );

            var conv1_b = model.Graph.AddTensor<float>(
                name: "conv1_b",
                shape: [1, 3, 128, 128],
                value: new float[1 * 3 * 128 * 128]
            );

            var fc_w = model.Graph.AddTensor<float>(
                name: "fc_w",
                shape: [1000, 9216],
                value: new float[1000 * 9216]
            );

            var fc_b = model.Graph.AddTensor<float>(
                name: "fc_b",
                shape: [1000, 9216],
                value: new float[1000]
            );

            // var conv1_out = model.Graph.AddEdge("conv1_out");

            /*
            model.Graph.AddNode(
                name: "conv1",
                opType: "Conv",
                domain: "",
                docString: "",
                inputs: [input, conv1_w, conv1_b],
                outputs: [conv1_out],
                attributes: []
            );
            */

            var conv1 = model.Graph.ConvTest(
                name: "conv1",
                options: new ConvTestInputOptions
                {
                    X = input,
                    W = conv1_w,
                    B = conv1_b,
                }
            );

            var relu1_out = model.Graph.AddEdge("relu1_out");

            model.Graph.AddNode(
                name: "relu1",
                opType: "Relu",
                domain: "",
                docString: "",
                inputs: [conv1],
                outputs: [relu1_out],
                attributes: []
            );

            var pool1_out = model.Graph.AddEdge("pool1_out");

            model.Graph.AddNode(
                name: "pool1",
                opType: "MaxPool",
                domain: "",
                docString: "",
                inputs: [relu1_out],
                outputs: [pool1_out],
                attributes: [
                    new OnnxAttribute<long[]>("kernel_shape", [2, 2]),
                    new OnnxAttribute<long[]>("strides", [2, 2]),
                ]
            );

            var flat_out = model.Graph.AddEdge("flat_out");

            model.Graph.AddNode(
                name: "flatten",
                opType: "Flatten",
                domain: "",
                docString: "",
                inputs: [pool1_out],
                outputs: [flat_out],
                attributes: []
            );

            model.Graph.AddNode(
                name: "fc",
                opType: "Gemm",
                domain: "",
                docString: "",
                inputs: [flat_out, fc_w, fc_b],
                outputs: [output],
                attributes: []
            );

            model.Save(outputPath, true);

            return;
        }

        static void B()
        {
            Console.WriteLine("B");

            var inputPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets", "bvlcalexnet-12-qdq.onnx");
            var outputPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets", "bvlcalexnet-12-qdq__.onnx");
            var outputPath2 = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets", "bvlcalexnet-12-qdq.txt");

            var data = File.ReadAllBytes(inputPath);
            var model = ModelProto.Parser.ParseFrom(data);

            var text = model.Graph.ToString();
            File.WriteAllTextAsync(outputPath2, text);

            using (var fs = File.Create(outputPath))
            {
                model.WriteTo(fs);
            }

        }
    }
}

public static class GraphExtensions
{
    public static IOnnxGraphEdge ConvTest(
        this OnnxGraph graph,
        string name,
        ConvTestInputOptions options
    )
    {
        var op = new ConvTest(
            name: name,
            options: new ConvTestInputOutputOptions
            {
                X = options.X,
                W = options.W,
                B = options.B,
                Y = graph.AddEdge(name + "_out"),
            }
        );

        graph.AddNode(op);
        return op.Y;
    }

    public static IOnnxGraphEdge ConvTest(
        this OnnxGraph graph,
        string name,
        ConvTestInputOutputOptions options
    )
    {
        var op = new ConvTest(
            name: name,
            options: options
        );

        graph.AddNode(op);
        return options.Y;
    }
}

public class ConvTestInputOptions
{
    public required IOnnxGraphEdge X {  get; set; }
    public required IOnnxGraphEdge W { get; set; }
    public IOnnxGraphEdge? B { get; set; }
}

public class ConvTestInputOutputOptions : ConvTestInputOptions
{
    public required IOnnxGraphEdge Y { get; set; }
}

public sealed class ConvTest : OnnxNode
{
    public ConvTest(
        string name,
        ConvTestInputOutputOptions options
    ) : base(
        name: name,
        opType: "Conv",
        domain: "",
        docString: "",
        inputs: OnnxHelper.NotNull([options.X, options.W, options.B]),
        outputs: OnnxHelper.NotNull([options.Y]),
        attributes: []
    )
    {
    }

    public IOnnxGraphEdge X
    {
        get => Inputs[0];
        set => SetInput(0, value);
    }

    public IOnnxGraphEdge W
    {
        get => Inputs[1];
        set => SetInput(1, value);
    }

    public IOnnxGraphEdge? B
    {
        get => Inputs.Count > 2 ? Inputs[2] : null;
        set => SetOptionalInput(2, value);
    }

    public IOnnxGraphEdge Y
    {
        get => Outputs[0];
        set => SetOutput(0, value);
    }

    public long[]? Strides
    {
        get => GetAttribute<long[]>("strides");
        set => SetAttribute("strides", value);
    }

    public long[]? Pads
    {
        get => GetAttribute<long[]>("pads");
        set => SetAttribute("pads", value);
    }

    public long[]? Dilations
    {
        get => GetAttribute<long[]>("dilations");
        set => SetAttribute("dilations", value);
    }

    public long Group
    {
        get => GetAttribute<long?>("group") ?? 1;
        set => SetAttribute("group", value);
    }

    public string AutoPad
    {
        get => GetAttribute<string>("auto_pad") ?? "NOTSET";
        set => SetAttribute("auto_pad", value);
    }

    internal static ConvTest FromProto(NodeProto node, OnnxGraph graph)
    {
        var inputs = node.Input
            .Select(x => graph.GetValue(x) ?? throw new InvalidOperationException($"Missing value '{x}'"))
            .ToArray();

        var outputs = node.Output
            .Select(x => graph.GetValue(x) ?? throw new InvalidOperationException($"Missing value '{x}'"))
            .ToArray();

        var conv = new ConvTest(
            name: node.Name,
            options: new ConvTestInputOutputOptions
            {
                X = inputs[0],
                W = inputs[1],
                B = inputs.Length > 2 ? inputs[2] : null,
                Y = outputs[0],
            }
        );

        conv.LoadAttributes(node);

        return conv;
    }
}

