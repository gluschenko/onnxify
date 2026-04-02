namespace Onnxify.TorchSharp;

public interface ITorchModuleExporter
{
    public bool IsMatch(TorchModule module);

    public abstract IOnnxGraphEdge Export(
        OnnxGraph graph,
        TorchModule module,
        IOnnxGraphEdge input,
        TorchModuleExportState state
    );
}

public abstract class TorchModuleExporter<TSource, TDestination> : ITorchModuleExporter
    where TSource : TorchModule
    where TDestination : OnnxNode
{
    public virtual bool IsMatch(TorchModule module)
    {
        return module is TSource;
    }

    public IOnnxGraphEdge Export(
        OnnxGraph graph,
        TorchModule module,
        IOnnxGraphEdge input,
        TorchModuleExportState state
    )
    {
        if (module is TSource sourceModule)
        {
            return Export(graph, sourceModule, input, state);
        }
        else
        {
            throw new NotSupportedException($"'{this.GetType().Name}' is not designed for '{module.GetType().FullName}'");
        }
    }

    public abstract IOnnxGraphEdge Export(
        OnnxGraph graph,
        TSource module,
        IOnnxGraphEdge input,
        TorchModuleExportState state
    );
}

[AttributeUsage(AttributeTargets.Class)]
public class TorchOpAttribute : Attribute
{
    public string Name { get; init; }

    public TorchOpAttribute(string name)
    {
        Name = name;
    }
}

[TorchOp("aten::conv2d")]
public sealed class ConvExporter : TorchModuleExporter<TorchModules.Conv2d, Conv>
{
    public override IOnnxGraphEdge Export(
        OnnxGraph graph,
        TorchModules.Conv2d module,
        IOnnxGraphEdge input,
        TorchModuleExportState state
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(module);

        var name = state.Next("conv");

        var weight = graph.AddTensor(
            name: $"{name}_w",
            shape: TorchHelper.GetShape(module.weight),
            value: TorchHelper.GetFloatData(module.weight)
        );

        IOnnxGraphEdge? bias = null;
        if (module.bias is not null)
        {
            bias = graph.AddTensor(
                name: $"{name}_b",
                shape: TorchHelper.GetShape(module.bias),
                value: TorchHelper.GetFloatData(module.bias)
            );
        }

        var padding = TorchHelper.ToLongArray(module.padding);
        var strides = TorchHelper.ToLongArray(module.stride);
        var dilations = TorchHelper.ToLongArray(module.dilation);

        return graph.Conv(
            name: name,
            options: new ConvInputOptions
            {
                X = input,
                W = weight,
                B = bias,
                KernelShape = TorchHelper.ToLongArray(module.kernel_size),
                Strides = strides.Length == 0 ? null : strides,
                Pads = padding.Length == 0 ? null : TorchHelper.ExpandPadding(padding),
                Dilations = dilations.Length == 0 ? null : dilations,
                Group = module.groups,
            }
        );
    }
}

[TorchOp("aten::relu")]
public sealed class ReluExporter : TorchModuleExporter<TorchModules.ReLU, Relu>
{
    public override IOnnxGraphEdge Export(
        OnnxGraph graph,
        TorchModules.ReLU module,
        IOnnxGraphEdge input,
        TorchModuleExportState state
    )
    {
        return graph.Relu(
            name: state.Next("relu"),
            options: new ReluInputOptions
            {
                X = input,
            }
        );
    }
}

[TorchOp("aten::max_pool2d")]
public sealed class MaxPool2dExporter : TorchModuleExporter<TorchModules.MaxPool2d, MaxPool>
{
    public override IOnnxGraphEdge Export(
        OnnxGraph graph,
        TorchModules.MaxPool2d module,
        IOnnxGraphEdge input,
        TorchModuleExportState state
    )
    {
        var padding = TorchHelper.ToLongArray(module.padding);
        var strides = TorchHelper.ToLongArray(module.stride);
        var result = graph.MaxPool(
            name: state.Next("maxpool"),
            options: new MaxPoolInputOptions
            {
                X = input,
                KernelShape = TorchHelper.ToLongArray(module.kernel_size),
                Strides = strides.Length == 0 ? null : strides,
                Pads = padding.Length == 0 ? null : TorchHelper.ExpandPadding(padding),
            }
        );

        return result.Y ?? throw new InvalidOperationException("MaxPool export did not produce an output edge.");
    }
}

[TorchOp("aten::dropout")]
public sealed class DropoutExporter : TorchModuleExporter<TorchModules.Dropout, Dropout>
{
    public override IOnnxGraphEdge Export(
        OnnxGraph graph,
        TorchModules.Dropout module,
        IOnnxGraphEdge input,
        TorchModuleExportState state
    )
    {
        var output = graph.Dropout(
            name: state.Next("dropout"),
            options: new DropoutInputOptions
            {
                Data = input,
            }
        );

        return output.Output;
    }
}

[TorchOp("aten::linear")]
public sealed class LinearExporter : TorchModuleExporter<TorchModules.Linear, Gemm>
{
    public override IOnnxGraphEdge Export(
        OnnxGraph graph,
        TorchModules.Linear module,
        IOnnxGraphEdge input,
        TorchModuleExportState state
    )
    {
        var name = state.Next("linear");
        var weight = graph.AddTensor(
            name: $"{name}_w",
            shape: TorchHelper.GetShape(module.weight),
            value: TorchHelper.GetFloatData(module.weight)
        );

        IOnnxGraphEdge? bias = null;
        if (module.bias is not null)
        {
            bias = graph.AddTensor(
                name: $"{name}_b",
                shape: TorchHelper.GetShape(module.bias),
                value: TorchHelper.GetFloatData(module.bias)
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
}

[TorchOp("aten::avg_pool2d")] // ???
public sealed class AdaptiveAvgPool2dExporter : TorchModuleExporter<TorchModules.AdaptiveAvgPool2d, GlobalAveragePool>
{
    public override IOnnxGraphEdge Export(
        OnnxGraph graph,
        TorchModules.AdaptiveAvgPool2d module,
        IOnnxGraphEdge input,
        TorchModuleExportState state
    )
    {
        var outputSize = TorchHelper.ToLongArray(module.output_size);
        if (outputSize.SequenceEqual([1L, 1L]))
        {
            return graph.GlobalAveragePool(
                name: state.Next("global_average_pool"),
                options: new GlobalAveragePoolInputOptions
                {
                    X = input,
                }
            );
        }

        throw new NotSupportedException(
            $"AdaptiveAvgPool2d with output_size [{string.Join(", ", outputSize)}] requires shape-aware " +
            $"lowering and cannot be exported by the recursive module walker."
        );
    }
}

public sealed class SequentialExporter : TorchModuleExporter<TorchModules.Sequential, GlobalAveragePool>
{
    public override IOnnxGraphEdge Export(
        OnnxGraph graph,
        TorchModules.Sequential module,
        IOnnxGraphEdge input,
        TorchModuleExportState state
    )
    {
        var children = module.children().OfType<TorchModule>().ToArray();
        if (children.Length == 0)
        {
            throw new NotSupportedException($"Unsupported TorchSharp module leaf: {module.GetType().FullName}.");
        }

        // This walker assumes child modules form a simple feed-forward chain in registration order.
        var current = input;
        foreach (var child in children)
        {
            current = child.Export(graph, current, state);
        }

        return current;
    }
}


