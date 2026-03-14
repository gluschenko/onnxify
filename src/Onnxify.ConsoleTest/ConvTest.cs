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

        public static ConvTest FromNodeProto(NodeProto nodeProto)
        {
            if (nodeProto.OpType != "Conv")
            {
                throw new InvalidOperationException($"Expected Conv node, got {nodeProto.OpType}");
            }

            var inputX = nodeProto.Input.Count > 0 ? nodeProto.Input[0] : null;
            var inputW = nodeProto.Input.Count > 1 ? nodeProto.Input[1] : null;
            var inputB = nodeProto.Input.Count > 2 ? nodeProto.Input[2] : null;
            var outputY = nodeProto.Output.Count > 0 ? nodeProto.Output[0] : null;

            var op = new ConvTest
            {
                InputX = new FormalParameter<TensorProto>("X", inputX),
                InputW = new FormalParameter<TensorProto>("W", inputW),
                InputB = inputB != null
                    ? new FormalParameter<TensorProto>("B", inputB)
                    : null,

                OutputY = new FormalParameter<TensorProto>("Y", outputY),
            };

            foreach (var attr in nodeProto.Attribute)
            {
                switch (attr.Name)
                {
                    case "auto_pad":
                        op.AttributeAuto_pad = new OperatorAttribute<string>(attr.Name, attr.S.ToStringUtf8());
                        break;

                    case "dilations":
                        op.AttributeDilations = new OperatorAttribute<long[]>(attr.Name, attr.Ints.ToArray());
                        break;

                    case "group":
                        op.AttributeGroup = new OperatorAttribute<long>(attr.Name, attr.I);
                        break;

                    case "kernel_shape":
                        op.AttributeKernel_shape = new OperatorAttribute<long[]>(attr.Name, attr.Ints.ToArray());
                        break;

                    case "pads":
                        op.AttributePads = new OperatorAttribute<long[]>(attr.Name, attr.Ints.ToArray());
                        break;

                    case "strides":
                        op.AttributeStrides = new OperatorAttribute<long[]>(attr.Name, attr.Ints.ToArray());
                        break;
                }
            }

            return op;
        }

        public NodeProto ToNodeProto()
        {
            var proj = GetProjection();
            var result = OperatorHelpers.ToNodeProto(proj);
            return result;
        }

        private OperatorProjection GetProjection()
        {
            var inputs = new List<IFormalParameter>();
            var outputs = new List<IFormalParameter>();
            var attributes = new List<IOperatorAttribute>();

            if (InputX is not null)
            {
                inputs.Add(InputX);
            }

            if (InputW is not null)
            {
                inputs.Add(InputW);
            }

            if (InputB is not null)
            {
                inputs.Add(InputB);
            }

            if (OutputY is not null)
            {
                outputs.Add(OutputY);
            }

            if (AttributeAuto_pad is not null)
            {
                attributes.Add(AttributeAuto_pad);
            }

            if (AttributeDilations is not null)
            {
                attributes.Add(AttributeDilations);
            }

            if (AttributeGroup is not null)
            {
                attributes.Add(AttributeGroup);
            }

            if (AttributeKernel_shape is not null)
            {
                attributes.Add(AttributeKernel_shape);
            }

            if (AttributePads is not null)
            {
                attributes.Add(AttributePads);
            }

            if (AttributeStrides is not null)
            {
                attributes.Add(AttributeStrides);
            }

            return new OperatorProjection
            {
                Name = Name,
                Domain = Domain,
                SinceVersion = SinceVersion,
                Inputs = inputs,
                Outputs = outputs,
                Attributes = attributes,
            };
        }
    }
}
