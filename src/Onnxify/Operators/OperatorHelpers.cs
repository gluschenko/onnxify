using Onnx;

namespace Onnxify.Operators
{
    public static class OperatorHelpers
    {
        public static NodeProto ToNodeProto(OperatorProjection op)
        {
            var node = new NodeProto
            {
                OpType = op.Name,
                Domain = op.Domain
            };

            foreach (var input in op.Inputs)
            {
                var value = input.ValueName;
                if (!string.IsNullOrEmpty(value))
                {
                    node.Input.Add(value);
                }
            }

            foreach (var output in op.Outputs)
            {
                var value = output.ValueName;
                if (!string.IsNullOrEmpty(value))
                {
                    node.Output.Add(value);
                }
            }

            foreach (var attr in op.Attributes)
            {
                var attribute = new AttributeProto
                {
                    Name = attr.Name
                };

                var type = attr.Type;

                if (type == typeof(long))
                {
                    attribute.Type = AttributeProto.Types.AttributeType.Int;
                }
                else if (type == typeof(long[]))
                {
                    attribute.Type = AttributeProto.Types.AttributeType.Ints;
                }
                else if (type == typeof(string))
                {
                    attribute.Type = AttributeProto.Types.AttributeType.String;
                }
                else if (type == typeof(float))
                {
                    attribute.Type = AttributeProto.Types.AttributeType.Float;
                }
                else if (type == typeof(float[]))
                {
                    attribute.Type = AttributeProto.Types.AttributeType.Floats;
                }
                else
                {
                    throw new NotImplementedException($"Not implemented for '{type.FullName}'");
                }

                node.Attribute.Add(attribute);
            }

            return node;
        }
    }
}
