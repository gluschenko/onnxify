using System.Globalization;
using System.Text;
using System.Text.Json;
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

            Test1();
            Test2();
            /*
            AA();
            A();
            B();
            */

            Console.WriteLine("Press any key to pay respect...");
            Console.ReadKey();
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

            var conv1_out = model.Graph.AddEdge("conv1_out");

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

            model.Graph.AddNode(new Conv(
                name: "conv1",
                x: input,
                w: conv1_w,
                b: conv1_b,
                y: conv1_out
            ));

            var relu1_out = model.Graph.AddEdge("relu1_out");

            model.Graph.AddNode(
                name: "relu1",
                opType: "Relu",
                domain: "",
                docString: "",
                inputs: [conv1_out],
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

        static void AA()
        {
            Console.WriteLine("AA");

            static string Pascal(string s)
            {
                if (string.IsNullOrEmpty(s))
                {
                    return s;
                }

                return char.ToUpperInvariant(s[0]) + s.Substring(1);
            }

            static string InputName(string name)
            {
                var p = Pascal(name);

                if (p.Equals("Input", StringComparison.OrdinalIgnoreCase))
                {
                    return "Input";
                }

                return "Input" + p;
            }

            static string OutputName(string name)
            {
                var p = Pascal(name);

                if (p.Equals("Output", StringComparison.OrdinalIgnoreCase))
                {
                    return "Output";
                }

                return "Output" + p;
            }

            static string AttributeName(string name)
            {
                var p = Pascal(name);

                if (p.Equals("Attribute", StringComparison.OrdinalIgnoreCase))
                {
                    return "Attribute";
                }

                return "Attribute" + p;
            }

            static string MapType(string type)
            {
                return type switch
                {
                    "T" => nameof(TensorProto),
                    "Tind" => nameof(TensorProto),
                    "T1" => nameof(TensorProto),
                    "tensor(int64)" => nameof(TensorProto),
                    "tensor(int32)" => nameof(TensorProto),
                    "tensor(float)" => nameof(TensorProto),
                    "tensor(double)" => nameof(TensorProto),
                    "tensor(bool)" => nameof(TensorProto),

                    _ => throw new NotSupportedException($"Unsupported ONNX type: {type}")
                };
            }

            var inputPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets", "onnx_operators.json");
            var outputPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "Onnxify", "Operators", "OnnxOperators.cs");

            var json = File.ReadAllText(inputPath);
            var root = JsonSerializer.Deserialize<OperatorSchemaRoot>(json) ?? throw new Exception();

            var sourceBuilder = new StringBuilder();

            sourceBuilder.AppendLine($"""
            using Onnx;
            using Onnxify.Abstractions;

            namespace Onnxify.Operators;

            """);

            foreach (var op in root.Operators)
            {
                var propBuilder = new StringBuilder();

                foreach (var x in op.Inputs)
                {
                    var required = x.Option == FormalParameterOption.Single ? " required " : " ";
                    var nullable = x.Option == FormalParameterOption.Optional ? "?" : "";

                    propBuilder.AppendLine($$"""
                        public{{required}}FormalParameter<{{MapType(x.Type)}}>{{nullable}} {{InputName(x.Name)}} { get; set; }
                    """);
                }

                if (op.Inputs.Count != 0)
                {
                    propBuilder.AppendLine();
                }

                foreach (var x in op.Outputs)
                {
                    var required = x.Option == FormalParameterOption.Single ? " required " : " ";
                    var nullable = x.Option == FormalParameterOption.Optional ? "?" : "";

                    propBuilder.AppendLine($$"""
                        public{{required}}FormalParameter<{{MapType(x.Type)}}>{{nullable}} {{OutputName(x.Name)}} { get; set; }
                    """);
                }

                if (op.Outputs.Count != 0)
                {
                    propBuilder.AppendLine();
                }

                foreach (var x in op.Attributes)
                {
                    var required = x.Required ? " required " : " ";
                    var nullable = x.Required ? "" : "?";

                    var typeEnum = (AttributeProto.Types.AttributeType)x.Type;

                    propBuilder.AppendLine($$"""
                        public{{required}}OperatorAttribute<{{OperatorHelpers.MapAttributeType(typeEnum)}}>{{nullable}} {{AttributeName(x.Name)}} { get; set; }
                    """);
                }

                if (op.Attributes.Count != 0)
                {
                    propBuilder.AppendLine();
                }

                sourceBuilder.AppendLine($$"""
                /// <summary>
                /// {{op.Name}} operator:
                /// <para>
                /// {{(op.Doc ?? "").Trim().Replace("\n", $"{Environment.NewLine}/// ")}}
                /// </para>
                /// </summary>
                public sealed class {{op.Name}} : Operator
                {
                    public override string Name => "{{op.Name}}";
                    public override string Domain => "{{op.Domain}}";
                    public override int SinceVersion => {{op.SinceVersion}};

                    {{propBuilder.ToString().TrimStart()}}
                }

                """);
            }

            var sourceCode = sourceBuilder.ToString();

            File.WriteAllText(outputPath, sourceCode);
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

public sealed class Conv : OnnxNode
{
    public Conv(
        string name,
        IOnnxGraphEdge x,
        IOnnxGraphEdge w,
        IOnnxGraphEdge? b,
        IOnnxGraphEdge y
    ) : base(
        name: name,
        opType: "Conv",
        domain: "",
        docString: "",
        inputs: b is null ? [x, w] : [x, w, b],
        outputs: [y],
        attributes: []
    )
    {
    }

    // --- Inputs ---

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

    // --- Output ---

    public IOnnxGraphEdge Y
    {
        get => Outputs[0];
        set => SetOutput(0, value);
    }

    // --- Attributes (typed) ---

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

    internal static Conv FromProto(NodeProto node, OnnxGraph graph)
    {
        var inputs = node.Input
            .Select(x => graph.GetValue(x) ?? throw new InvalidOperationException($"Missing value '{x}'"))
            .ToArray();

        var outputs = node.Output
            .Select(x => graph.GetValue(x) ?? throw new InvalidOperationException($"Missing value '{x}'"))
            .ToArray();

        var conv = new Conv(
            name: node.Name,
            x: inputs[0],
            w: inputs[1],
            b: inputs.Length > 2 ? inputs[2] : null,
            y: outputs[0]
        );

        conv.LoadAttributes(node);

        return conv;
    }
}

