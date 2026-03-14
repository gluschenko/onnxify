using Onnx;
using Onnxify.Abstractions;

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

        public static string MapAttributeType(AttributeProto.Types.AttributeType type)
        {
            return type switch
            {
                AttributeProto.Types.AttributeType.Undefined => typeof(object).Name,

                AttributeProto.Types.AttributeType.Float => typeof(float).Name,
                AttributeProto.Types.AttributeType.Int => typeof(long).Name,
                AttributeProto.Types.AttributeType.String => typeof(string).Name,

                AttributeProto.Types.AttributeType.Tensor => nameof(TensorProto),
                AttributeProto.Types.AttributeType.Graph => nameof(GraphProto),
                AttributeProto.Types.AttributeType.SparseTensor => nameof(SparseTensorProto),

                AttributeProto.Types.AttributeType.Floats => $"{typeof(float).Name}[]",
                AttributeProto.Types.AttributeType.Ints => $"{typeof(long).Name}[]",
                AttributeProto.Types.AttributeType.Strings => $"{typeof(string).Name}[]",

                AttributeProto.Types.AttributeType.Tensors => $"{nameof(TensorProto)}[]",
                AttributeProto.Types.AttributeType.Graphs => $"{nameof(GraphProto)}[]",
                AttributeProto.Types.AttributeType.SparseTensors => $"{nameof(SparseTensorProto)}[]",

                _ => throw new NotSupportedException($"Unsupported AttributeType: {type}")
            };
        }

        public static IOperatorAttribute MapAttributeType(AttributeProto attr)
        {
            return attr.Type switch
            {
                AttributeProto.Types.AttributeType.Undefined
                    => new OperatorAttribute<object?>(attr.Name, null),
                AttributeProto.Types.AttributeType.Float
                    => new OperatorAttribute<float>(attr.Name, attr.F),
                AttributeProto.Types.AttributeType.Int
                    => new OperatorAttribute<long>(attr.Name, attr.I),
                AttributeProto.Types.AttributeType.String
                    => new OperatorAttribute<string>(attr.Name, attr.S.ToStringUtf8()),
                AttributeProto.Types.AttributeType.Tensor
                    => new OperatorAttribute<TensorProto>(attr.Name, attr.T),
                AttributeProto.Types.AttributeType.Graph
                    => new OperatorAttribute<GraphProto>(attr.Name, attr.G),
                AttributeProto.Types.AttributeType.SparseTensor
                    => new OperatorAttribute<SparseTensorProto>(attr.Name, attr.SparseTensor),
                AttributeProto.Types.AttributeType.Floats
                    => new OperatorAttribute<float[]>(attr.Name, attr.Floats.ToArray()),
                AttributeProto.Types.AttributeType.Ints
                    => new OperatorAttribute<long[]>(attr.Name, attr.Ints.ToArray()),
                AttributeProto.Types.AttributeType.Strings
                    => new OperatorAttribute<string[]>(attr.Name, attr.Strings.Select(x => x.ToStringUtf8()).ToArray()),
                AttributeProto.Types.AttributeType.Tensors
                    => new OperatorAttribute<TensorProto[]>(attr.Name, attr.Tensors.ToArray()),
                AttributeProto.Types.AttributeType.Graphs
                    => new OperatorAttribute<GraphProto[]>(attr.Name, attr.Graphs.ToArray()),
                AttributeProto.Types.AttributeType.SparseTensors
                    => new OperatorAttribute<SparseTensorProto[]>(attr.Name, attr.SparseTensors.ToArray()),
                _ => throw new NotSupportedException($"Unsupported AttributeType: {attr.Type}")
            };
        }
    }
}
