using Onnx;
using Onnxify.Abstractions;

namespace Onnxify.Operators
{
    internal sealed class ConvTest : Operator
    {
        public override string Name => "Conv";
        public override string Domain => "";
        public override int SinceVersion => 11;

        public required FormalParameter<TensorProto> InputX { get; set; }
        public required FormalParameter<TensorProto> InputW { get; set; }
        public FormalParameter<TensorProto>? InputB { get; set; }

        public required FormalParameter<TensorProto> OutputY { get; set; }

        public OperatorAttribute<String>? AttributeAuto_pad { get; set; }
        public OperatorAttribute<Int64[]>? AttributeDilations { get; set; }
        public OperatorAttribute<Int64>? AttributeGroup { get; set; }
        public OperatorAttribute<Int64[]>? AttributeKernel_shape { get; set; }
        public OperatorAttribute<Int64[]>? AttributePads { get; set; }
        public OperatorAttribute<Int64[]>? AttributeStrides { get; set; }


    }

    public static class OperatorExtensions
    {

    }
}
