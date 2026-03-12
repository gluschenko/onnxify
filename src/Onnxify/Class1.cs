using Onnx;
using static TorchSharp.torch;

namespace Onnxify
{
    public class Class1
    {
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
