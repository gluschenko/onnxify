namespace Onnxify.TorchSharp.Operators;

public static class OperatorExtensions
{
    public static IOnnxGraphEdge Conv(
        this OnnxGraph graph,
        string name,
        IOnnxGraphEdge input,
        global::TorchSharp.Modules.Conv2d op
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(op);

        var weight = graph.AddTensor(
            name: $"{name}_w",
            shape: GetShape(op.weight),
            value: GetFloatData(op.weight)
        );

        IOnnxGraphEdge? bias = null;
        if (op.bias is not null)
        {
            bias = graph.AddTensor(
                name: $"{name}_b",
                shape: GetShape(op.bias),
                value: GetFloatData(op.bias)
            );
        }

        var padding = ToLongArray(op.padding);

        var strides = ToLongArray(op.stride);
        var dilations = ToLongArray(op.dilation);

        return graph.Conv(
            name: name,
            options: new ConvInputOptions
            {
                X = input,
                W = weight,
                B = bias,
                KernelShape = ToLongArray(op.kernel_size),
                Strides = strides.Length == 0 ? null : strides,
                Pads = padding.Length == 0 ? null : ExpandPadding(padding),
                Dilations = dilations.Length == 0 ? null : dilations,
                Group = op.groups,
            }
        );
    }

    private static long[] GetShape(global::TorchSharp.torch.Tensor tensor)
    {
        return ((IEnumerable<long>)tensor.shape).ToArray();
    }

    private static float[] GetFloatData(global::TorchSharp.torch.Tensor tensor)
    {
        return tensor
            .detach()
            .cpu()
            .data<float>()
            .ToArray();
    }

    private static long[] ToLongArray(IEnumerable<long>? value)
    {
        return value?.ToArray() ?? [];
    }

    private static long[] ExpandPadding(long[] padding)
    {
        return padding.Length switch
        {
            1 => [padding[0], padding[0], padding[0], padding[0]],
            2 => [padding[0], padding[1], padding[0], padding[1]],
            4 => padding,
            _ => throw new NotSupportedException($"Unsupported Conv2d padding rank: {padding.Length}."),
        };
    }
}
