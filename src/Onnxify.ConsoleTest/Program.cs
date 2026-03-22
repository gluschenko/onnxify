using System.Globalization;
using System.Text;
using Google.Protobuf;
using Onnx;
using Onnxify;

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
            Test3();
            Test4();

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

            var conv = model.Graph.Conv(
                name: "conv1",
                options: new ConvInputOptions
                {
                    X = conv1_in,
                    W = conv1_w,
                    B = conv1_b,
                }
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

            var conv1 = model.Graph.Conv(
                name: "conv1",
                options: new ConvInputOptions
                {
                    X = input,
                    W = conv1_w,
                    B = conv1_b,
                }
            );

            var relu1 = model.Graph.Relu(
                name: "relu1",
                options: new ReluInputOptions
                {
                    X = conv1
                }
            );

            var pool1 = model.Graph.MaxPool(
                name: "pool1",
                options: new MaxPoolInputOptions
                {
                    X = relu1,
                    KernelShape = [2, 2],
                    Strides = [2, 2]
                }
            );

            var flatten = model.Graph.Flatten(
                name: "flatten",
                options: new FlattenInputOptions
                {
                    Input = pool1.Y,
                }
            );

            model.Graph.Gemm(
                name: "fc",
                options: new GemmInputOutputOptions
                {
                    A = flatten,
                    B = fc_b,
                    C = fc_w,
                    Y = output,
                }
            );

            model.Save(outputPath, true);

            return;
        }

        static void Test3()
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

        static void Test4()
        {
            var model = new AlexNet("alexnet", 10);
            var onnxModel = model.Onnxify();

            var outputPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets", "alexnet__test.onnx");
            onnxModel.Save(outputPath, true);
            return;
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
        get => (IOnnxGraphEdge)Inputs[0];
        set => SetInput(0, value);
    }
    public IOnnxGraphEdge W
    {
        get => (IOnnxGraphEdge)Inputs[1];
        set => SetInput(1, value);
    }
    public IOnnxGraphEdge? B
    {
        get => Inputs.Count > 2 ? (IOnnxGraphEdge)Inputs[2] : null;
        set => SetOptionalInput(2, value);
    }
    public IOnnxGraphEdge Y
    {
        get => (IOnnxGraphEdge)Outputs[0];
        set => SetOutput(0, value);
    }
    public string? AutoPad
    {
        get => HasAttribute("auto_pad") ? GetAttribute<string>("auto_pad") : null;
        set
        {
            if (value is not null)
            {
                SetAttribute<string>("auto_pad", (string)value);
            }
            else
            {
                RemoveAttribute("auto_pad");
            }
        }
    }
    public long[]? Dilations
    {
        get => HasAttribute("dilations") ? GetAttribute<long[]>("dilations") : null;
        set
        {
            if (value is not null)
            {
                SetAttribute<long[]>("dilations", (long[])value);
            }
            else
            {
                RemoveAttribute("dilations");
            }
        }
    }
    public long? Group
    {
        get => HasAttribute("group") ? GetAttribute<long>("group") : null;
        set
        {
            if (value is not null)
            {
                SetAttribute<long>("group", (long)value);
            }
            else
            {
                RemoveAttribute("group");
            }
        }
    }
    public long[]? KernelShape
    {
        get => HasAttribute("kernel_shape") ? GetAttribute<long[]>("kernel_shape") : null;
        set
        {
            if (value is not null)
            {
                SetAttribute<long[]>("kernel_shape", (long[])value);
            }
            else
            {
                RemoveAttribute("kernel_shape");
            }
        }
    }
    public long[]? Pads
    {
        get => HasAttribute("pads") ? GetAttribute<long[]>("pads") : null;
        set
        {
            if (value is not null)
            {
                SetAttribute<long[]>("pads", (long[])value);
            }
            else
            {
                RemoveAttribute("pads");
            }
        }
    }
    public long[]? Strides
    {
        get => HasAttribute("strides") ? GetAttribute<long[]>("strides") : null;
        set
        {
            if (value is not null)
            {
                SetAttribute<long[]>("strides", (long[])value);
            }
            else
            {
                RemoveAttribute("strides");
            }
        }
    }

    internal static Conv FromProto(NodeProto node, OnnxGraph graph)
    {
        var inputs = node.Input
            .Select(x => graph.GetValue(x) ?? throw new InvalidOperationException($"Missing value '{x}'"))
            .ToArray();

        var outputs = node.Output
            .Select(x => graph.GetValue(x) ?? throw new InvalidOperationException($"Missing value '{x}'"))
            .ToArray();

        var attributes = node.Attribute.ToDictionary(x => x.Name, x => x.GetValue());

        var op = new Conv(
            name: node.Name,
            options: new ConvInputOutputOptions
            {
                X = inputs[0],
                W = inputs[1],
                B = inputs.Length > 2 ? inputs[2] : null,
                Y = outputs[0],
                AutoPad = (string?)attributes.GetValueOrDefault("auto_pad"),
            }
        );

        return op;
    }
}

