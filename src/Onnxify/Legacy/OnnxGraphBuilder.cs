using Onnx;
using static Onnxify.Legacy.OnnxUtils;
using static TorchSharp.torch;

namespace Onnxify.Legacy
{
    public class OnnxGraphBuilder
    {
        private readonly List<NodeProto> nodes = new();
        private readonly List<TensorProto> weights = new();
        private readonly List<ModelArgs> inputs = new();
        private readonly List<ModelArgs> outputs = new();

        public void AddInput(string name, TensorProto.Types.DataType type, long[] dims)
        {
            inputs.Add(new OnnxUtils.ModelArgs(name, type, dims.ToList(), null));
        }

        public void AddOutput(string name, TensorProto.Types.DataType type, long[] dims)
        {
            outputs.Add(new OnnxUtils.ModelArgs(name, type, dims.ToList(), null));
        }

        public void AddWeight(string name, Tensor tensor)
        {
            var proto = new TensorProto
            {
                Name = name,
                DataType = (int)TensorProto.Types.DataType.Float
            };

            foreach (var d in tensor.shape)
            {
                proto.Dims.Add(d);
            }

            proto.FloatData.AddRange(tensor.data<float>().ToArray());

            weights.Add(proto);
        }

        public string AddNode(
            string op,
            string[] input,
            string[] output,
            Dictionary<string, object>? attrs = null)
        {
            var node = OnnxUtils.MakeNode(op, input, output, Guid.NewGuid().ToString());

            if (attrs != null)
            {
                foreach (var kv in attrs)
                {
                    if (kv.Value is long l)
                    {
                        OnnxUtils.NodeAddAttributes(node, kv.Key, l);
                    }

                    if (kv.Value is long[] la)
                    {
                        OnnxUtils.NodeAddAttributes(node, kv.Key, la);
                    }

                    if (kv.Value is float f)
                    {
                        OnnxUtils.NodeAddAttributes(node, kv.Key, (double)f);
                    }
                }
            }

            nodes.Add(node);
            return output[0];
        }


        public ModelProto Build()
        {
            return OnnxUtils.MakeModel(
                nodes,
                "torchsharp-exporter",
                "model",
                "google.com",
                "1.0",
                1,
                17,
                inputs,
                outputs,
                new(),
                weights
            );
        }

        private TensorProto FromTorchTensor(string name, Tensor tensor)
        {
            var proto = new TensorProto
            {
                Name = name,
                DataType = (int)TensorProto.Types.DataType.Float
            };

            foreach (var x in tensor.shape)
            {
                proto.Dims.Add(x);
            }

            proto.FloatData.AddRange(tensor.data<float>().ToArray());

            return proto;
        }
    }
}

