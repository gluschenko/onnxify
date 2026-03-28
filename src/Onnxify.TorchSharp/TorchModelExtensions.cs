using Onnxify.TorchSharp.Operators;

namespace Onnxify.TorchSharp;

using TorchModule = global::TorchSharp.torch.nn.Module<global::TorchSharp.torch.Tensor, global::TorchSharp.torch.Tensor>;

public static class TorchModelExtensions
{
    public static IOnnxGraphEdge ToOnnxGraph(
        this TorchModule module,
        OnnxGraph graph,
        IOnnxGraphEdge input)
    {
        ArgumentNullException.ThrowIfNull(module);
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportModule(module, graph, input, new TorchModuleExportState());
    }

    private static IOnnxGraphEdge ExportModule(
        TorchModule module,
        OnnxGraph graph,
        IOnnxGraphEdge input,
        TorchModuleExportState state)
    {
        if (module is global::TorchSharp.Modules.Conv2d conv2d)
        {
            return graph.Conv(state.Next("conv"), input, conv2d);
        }

        if (module is global::TorchSharp.Modules.ReLU)
        {
            return graph.Relu(
                name: state.Next("relu"),
                options: new ReluInputOptions
                {
                    X = input,
                }
            );
        }

        if (module is global::TorchSharp.Modules.MaxPool2d maxPool2d)
        {
            var padding = ToLongArray(maxPool2d.padding);
            var strides = ToLongArray(maxPool2d.stride);
            var result = graph.MaxPool(
                name: state.Next("maxpool"),
                options: new MaxPoolInputOptions
                {
                    X = input,
                    KernelShape = ToLongArray(maxPool2d.kernel_size),
                    Strides = strides.Length == 0 ? null : strides,
                    Pads = padding.Length == 0 ? null : ExpandPadding(padding),
                });

            return result.Y ?? throw new InvalidOperationException("MaxPool export did not produce an output edge.");
        }

        if (module is global::TorchSharp.Modules.Dropout)
        {
            return input;
        }

        if (module is global::TorchSharp.Modules.Linear linear)
        {
            var name = state.Next("fc");
            var weight = graph.AddTensor(
                name: $"{name}_w",
                shape: GetShape(linear.weight),
                value: GetFloatData(linear.weight)
            );

            IOnnxGraphEdge? bias = null;
            if (linear.bias is not null)
            {
                bias = graph.AddTensor(
                    name: $"{name}_b",
                    shape: GetShape(linear.bias),
                    value: GetFloatData(linear.bias)
                );
            }

            return graph.Gemm(
                name: name,
                options: new GemmInputOptions
                {
                    A = input,
                    B = weight,
                    C = bias,
                    TransB = 1,
                }
            );
        }

        if (module is global::TorchSharp.Modules.AdaptiveAvgPool2d adaptiveAvgPool2d)
        {
            var outputSize = ToLongArray(adaptiveAvgPool2d.output_size);
            if (outputSize.SequenceEqual([1L, 1L]))
            {
                return graph.GlobalAveragePool(
                    name: state.Next("gap"),
                    options: new GlobalAveragePoolInputOptions
                    {
                        X = input,
                    }
                );
            }

            throw new NotSupportedException(
                $"AdaptiveAvgPool2d with output_size [{string.Join(", ", outputSize)}] requires shape-aware lowering and cannot be exported by the recursive module walker.");
        }

        var children = module.children().OfType<TorchModule>().ToArray();
        if (children.Length == 0)
        {
            throw new NotSupportedException(
                $"Unsupported TorchSharp module leaf: {module.GetType().FullName}.");
        }

        // This walker assumes child modules form a simple feed-forward chain in registration order.
        var current = input;
        foreach (var child in children)
        {
            current = ExportModule(child, graph, current, state);
        }

        return current;
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
            _ => throw new NotSupportedException($"Unsupported padding rank: {padding.Length}."),
        };
    }
}

internal sealed class TorchModuleExportState
{
    private readonly Dictionary<string, int> _counters = new(StringComparer.Ordinal);

    public string Next(string prefix)
    {
        if (!_counters.TryGetValue(prefix, out var index))
        {
            _counters[prefix] = 1;
            return $"{prefix}0";
        }

        _counters[prefix] = index + 1;
        return $"{prefix}{index}";
    }
}
