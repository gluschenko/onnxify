using Onnx;

namespace Onnxify.ConsoleTest
{
    public static class AlexNetExporter
    {
        public static ModelProto Export(AlexNet model)
        {
            var g = new OnnxGraphBuilder();

            g.AddInput("input", TensorProto.Types.DataType.Float, new long[] { -1, 3, 224, 224 });

            string x = "input";

            x = ConvBlock(g, x, "features.c1", 3, 64, 3, 2, 1);
            x = MaxPool(g, x, "mp1");

            x = ConvBlock(g, x, "features.c2", 64, 192, 3, 1, 1);
            x = MaxPool(g, x, "mp2");

            x = ConvBlock(g, x, "features.c3", 192, 384, 3, 1, 1);
            x = ConvBlock(g, x, "features.c4", 384, 256, 3, 1, 1);
            x = ConvBlock(g, x, "features.c5", 256, 256, 3, 1, 1);

            x = MaxPool(g, x, "mp3");

            x = g.AddNode("GlobalAveragePool", [x], ["avg_out"]);

            x = g.AddNode("Flatten", new[] { x }, new[] { "flat" });

            x = Linear(g, x, "classifier.l1");
            x = Relu(g, x);

            x = Linear(g, x, "classifier.l2");
            x = Relu(g, x);

            x = Linear(g, x, "classifier.l3");

            g.AddOutput(x, TensorProto.Types.DataType.Float, new long[] { -1, 10 });

            return g.Build();
        }

        static string ConvBlock(OnnxGraphBuilder g, string input, string name,
            int inC, int outC, int k, int s, int p)
        {
            var conv = g.AddNode(
                "Conv",
                new[] { input, $"{name}.weight", $"{name}.bias" },
                new[] { $"{name}_out" },
                new Dictionary<string, object>
                {
                {"kernel_shape", new long[]{k,k}},
                {"strides", new long[]{s,s}},
                {"pads", new long[]{p,p,p,p}}
                });

            return Relu(g, conv);
        }

        static string Relu(OnnxGraphBuilder g, string input)
        {
            return g.AddNode("Relu", new[] { input }, new[] { $"{input}_relu" });
        }

        static string MaxPool(OnnxGraphBuilder g, string input, string name)
        {
            return g.AddNode(
                "MaxPool",
                new[] { input },
                new[] { $"{name}_out" },
                new Dictionary<string, object>
                {
                {"kernel_shape", new long[]{2,2}}
                });
        }

        static string Linear(OnnxGraphBuilder g, string input, string name)
        {
            return g.AddNode(
                "Gemm",
                new[] { input, $"{name}.weight", $"{name}.bias" },
                new[] { $"{name}_out" });
        }
    }
}
