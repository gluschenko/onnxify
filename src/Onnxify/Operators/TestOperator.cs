using Onnx;
using Onnxify.Abstractions;

namespace Onnxify.Operators
{
    public class TestOperator : Operator
    {
        public override string Name => "Conv";
        public override string Domain => "";
        public override int SinceVersion => 13;

        public required FormalParameter<TensorProto> InputX { get; set; }
        public required FormalParameter<TensorProto> InputW { get; set; }
        public required FormalParameter<TensorProto>? InputB { get; set; }

        public required FormalParameter<TensorProto>? OutputY { get; set; }
    }
}
