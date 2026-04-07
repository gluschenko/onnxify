using System.Reflection;
using static TorchSharp.torch;

namespace Onnxify.TorchSharp;

public static class TorchModuleExtensions
{
    public static IOnnxGraphEdge Export(
        this TorchModule module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(module);
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return module switch
        {
            TorchModules.Sequential m => m.Export(graph, input),
            TorchModules.Conv1d m => m.Export(graph, input),
            TorchModules.Conv2d m => m.Export(graph, input),
            TorchModules.Conv3d m => m.Export(graph, input),
            TorchModules.ReLU m => m.Export(graph, input),
            TorchModules.ReLU6 m => m.Export(graph, input),
            TorchModules.LeakyReLU m => m.Export(graph, input),
            TorchModules.ELU m => m.Export(graph, input),
            TorchModules.CELU m => m.Export(graph, input),
            TorchModules.Sigmoid m => m.Export(graph, input),
            TorchModules.Tanh m => m.Export(graph, input),
            TorchModules.Softmax m => m.Export(graph, input),
            TorchModules.LogSoftmax m => m.Export(graph, input),
            TorchModules.Hardtanh m => m.Export(graph, input),
            TorchModules.Hardsigmoid m => m.Export(graph, input),
            TorchModules.Hardswish m => m.Export(graph, input),
            TorchModules.SiLU m => m.Export(graph, input),
            TorchModules.LogSigmoid m => m.Export(graph, input),
            TorchModules.MaxPool2d m => m.Export(graph, input),
            TorchModules.MaxPool1d m => m.Export(graph, input),
            TorchModules.MaxPool3d m => m.Export(graph, input),
            TorchModules.Dropout m => m.Export(graph, input),
            TorchModules.Linear m => m.Export(graph, input),
            TorchModules.AvgPool1d m => m.Export(graph, input),
            TorchModules.AvgPool3d m => m.Export(graph, input),
            TorchModules.Flatten m => m.Export(graph, input),
            TorchModules.AdaptiveAvgPool2d m => m.Export(graph, input),
            TorchModules.Embedding m => m.Export(graph, input),
            _ => throw new NotImplementedException($"Not implemented for '{module.GetType().FullName}'"),
        };
    }

    public static IOnnxGraphEdge Export(
        this TorchModules.Sequential module,
        OnnxGraph graph,
        IOnnxGraphEdge input
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
            current = child.Export(graph, current);
        }

        return current;
    }

    [TorchOp("aten::conv1d")]
    public static IOnnxGraphEdge Export(
        this TorchModules.Conv1d module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(module);

        var name = graph.NextName("conv");

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
                Pads = padding.Length == 0 ? null : TorchHelper.ExpandPadding(padding, spatialRank: 1),
                Dilations = dilations.Length == 0 ? null : dilations,
                Group = module.groups,
            }
        );
    }

    [TorchOp("aten::conv2d")]
    public static IOnnxGraphEdge Export(
        this TorchModules.Conv2d module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(module);

        var name = graph.NextName("conv");

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
                Pads = padding.Length == 0 ? null : TorchHelper.ExpandPadding(padding, spatialRank: 2),
                Dilations = dilations.Length == 0 ? null : dilations,
                Group = module.groups,
            }
        );
    }

    [TorchOp("aten::conv3d")]
    public static IOnnxGraphEdge Export(
        this TorchModules.Conv3d module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(module);

        var name = graph.NextName("conv");

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
                Pads = padding.Length == 0 ? null : TorchHelper.ExpandPadding(padding, spatialRank: 3),
                Dilations = dilations.Length == 0 ? null : dilations,
                Group = module.groups,
            }
        );
    }

    [TorchOp("aten::relu")]
    public static IOnnxGraphEdge Export(
        this TorchModules.ReLU module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        return graph.Relu(
            name: graph.NextName("relu"),
            options: new ReluInputOptions
            {
                X = input,
            }
        );
    }

    [TorchOp("aten::relu6")]
    public static IOnnxGraphEdge Export(
        this TorchModules.ReLU6 module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        var name = graph.NextName("relu6");
        var min = graph.AddTensor($"{name}_min", [], [0f]);
        var max = graph.AddTensor($"{name}_max", [], [6f]);

        return graph.Clip(
            name: name,
            options: new ClipInputOptions
            {
                Input = input,
                Min = min,
                Max = max,
            }
        );
    }

    [TorchOp("aten::leaky_relu")]
    public static IOnnxGraphEdge Export(
        this TorchModules.LeakyReLU module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        return graph.LeakyRelu(
            name: graph.NextName("leaky_relu"),
            options: new LeakyReluInputOptions
            {
                X = input,
                Alpha = (float)module.negative_slope,
            }
        );
    }

    [TorchOp("aten::elu")]
    public static IOnnxGraphEdge Export(
        this TorchModules.ELU module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        return graph.Elu(
            name: graph.NextName("elu"),
            options: new EluInputOptions
            {
                X = input,
                Alpha = (float)module.alpha,
            }
        );
    }

    [TorchOp("aten::celu")]
    public static IOnnxGraphEdge Export(
        this TorchModules.CELU module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        var alpha = (float)GetOptionalDoubleMember(module, defaultValue: 1.0, "alpha", "_alpha");

        return graph.Celu(
            name: graph.NextName("celu"),
            options: new CeluInputOptions
            {
                X = input,
                Alpha = alpha,
            }
        );
    }

    [TorchOp("aten::sigmoid")]
    public static IOnnxGraphEdge Export(
        this TorchModules.Sigmoid module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        return graph.Sigmoid(
            name: graph.NextName("sigmoid"),
            options: new SigmoidInputOptions
            {
                X = input,
            }
        );
    }

    [TorchOp("aten::tanh")]
    public static IOnnxGraphEdge Export(
        this TorchModules.Tanh module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        return graph.Tanh(
            name: graph.NextName("tanh"),
            options: new TanhInputOptions
            {
                Input = input,
            }
        );
    }

    [TorchOp("aten::hardtanh")]
    public static IOnnxGraphEdge Export(
        this TorchModules.Hardtanh module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        var name = graph.NextName("hardtanh");
        var minValue = (float)GetOptionalDoubleMember(module, defaultValue: -1.0, "min_val", "_min_val");
        var maxValue = (float)GetOptionalDoubleMember(module, defaultValue: 1.0, "max_val", "_max_val");

        var min = graph.AddTensor($"{name}_min", [], [minValue]);
        var max = graph.AddTensor($"{name}_max", [], [maxValue]);

        return graph.Clip(
            name: name,
            options: new ClipInputOptions
            {
                Input = input,
                Min = min,
                Max = max,
            }
        );
    }

    [TorchOp("aten::hardsigmoid")]
    public static IOnnxGraphEdge Export(
        this TorchModules.Hardsigmoid module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        return graph.HardSigmoid(
            name: graph.NextName("hardsigmoid"),
            options: new HardSigmoidInputOptions
            {
                X = input,
                Alpha = 1f / 6f,
                Beta = 0.5f,
            }
        );
    }

    [TorchOp("aten::hardswish")]
    public static IOnnxGraphEdge Export(
        this TorchModules.Hardswish module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        return graph.HardSwish(
            name: graph.NextName("hardswish"),
            options: new HardSwishInputOptions
            {
                X = input,
            }
        );
    }

    [TorchOp("aten::silu")]
    public static IOnnxGraphEdge Export(
        this TorchModules.SiLU module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        var sigmoid = graph.Sigmoid(
            name: graph.NextName("silu_sigmoid"),
            options: new SigmoidInputOptions
            {
                X = input,
            }
        );

        return graph.Mul(
            name: graph.NextName("silu"),
            options: new MulInputOptions
            {
                A = input,
                B = sigmoid,
            }
        );
    }

    [TorchOp("aten::softmax.int")]
    public static IOnnxGraphEdge Export(
        this TorchModules.Softmax module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        return graph.Softmax(
            name: graph.NextName("softmax"),
            options: new SoftmaxInputOptions
            {
                Input = input,
                Axis = module.dim,
            }
        );
    }

    [TorchOp("aten::log_softmax.int")]
    public static IOnnxGraphEdge Export(
        this TorchModules.LogSoftmax module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        return graph.LogSoftmax(
            name: graph.NextName("log_softmax"),
            options: new LogSoftmaxInputOptions
            {
                Input = input,
                Axis = module.dim,
            }
        );
    }

    [TorchOp("aten::log_sigmoid")]
    public static IOnnxGraphEdge Export(
        this TorchModules.LogSigmoid module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        var sigmoid = graph.Sigmoid(
            name: graph.NextName("log_sigmoid_sigmoid"),
            options: new SigmoidInputOptions
            {
                X = input,
            }
        );

        return graph.Log(
            name: graph.NextName("log_sigmoid"),
            options: new LogInputOptions
            {
                Input = sigmoid,
            }
        );
    }

    [TorchOp("aten::max_pool2d")]
    public static IOnnxGraphEdge Export(
        this TorchModules.MaxPool2d module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        var padding = TorchHelper.ToLongArray(module.padding);
        var strides = TorchHelper.ToLongArray(module.stride);
        var result = graph.MaxPool(
            name: graph.NextName("maxpool"),
            options: new MaxPoolInputOptions
            {
                X = input,
                KernelShape = TorchHelper.ToLongArray(module.kernel_size),
                Strides = strides.Length == 0 ? null : strides,
                Pads = padding.Length == 0 ? null : TorchHelper.ExpandPadding(padding, spatialRank: 2),
            }
        );

        return result.Y ?? throw new InvalidOperationException("MaxPool export did not produce an output edge.");
    }

    [TorchOp("aten::max_pool1d")]
    public static IOnnxGraphEdge Export(
        this TorchModules.MaxPool1d module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        var strides = TorchHelper.ToLongArray(module.stride);
        var padding = TorchHelper.ToLongArray(module.padding);
        var dilations = TorchHelper.ToLongArray(module.dilation);
        var result = graph.MaxPool(
            name: graph.NextName("maxpool"),
            options: new MaxPoolInputOptions
            {
                X = input,
                KernelShape = TorchHelper.ToLongArray(module.kernel_size),
                Strides = strides.Length == 0 ? null : strides,
                Pads = padding.Length == 0 ? null : TorchHelper.ExpandPadding(padding, spatialRank: 1),
                Dilations = dilations.Length == 0 ? null : dilations,
                CeilMode = module.ceil_mode ? 1 : 0,
            }
        );

        return result.Y ?? throw new InvalidOperationException("MaxPool export did not produce an output edge.");
    }

    [TorchOp("aten::max_pool3d")]
    public static IOnnxGraphEdge Export(
        this TorchModules.MaxPool3d module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        var strides = TorchHelper.ToLongArray(module.stride);
        var padding = TorchHelper.ToLongArray(module.padding);
        var dilations = TorchHelper.ToLongArray(module.dilation);
        var result = graph.MaxPool(
            name: graph.NextName("maxpool"),
            options: new MaxPoolInputOptions
            {
                X = input,
                KernelShape = TorchHelper.ToLongArray(module.kernel_size),
                Strides = strides.Length == 0 ? null : strides,
                Pads = padding.Length == 0 ? null : TorchHelper.ExpandPadding(padding, spatialRank: 3),
                Dilations = dilations.Length == 0 ? null : dilations,
                CeilMode = module.ceil_mode ? 1 : 0,
            }
        );

        return result.Y ?? throw new InvalidOperationException("MaxPool export did not produce an output edge.");
    }

    [TorchOp("aten::dropout")]
    public static IOnnxGraphEdge Export(
        this TorchModules.Dropout module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        var output = graph.Dropout(
            name: graph.NextName("dropout"),
            options: new DropoutInputOptions
            {
                Data = input,
            }
        );

        return output.Output;
    }

    [TorchOp("aten::linear")]
    public static IOnnxGraphEdge Export(
        this TorchModules.Linear module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        var name = graph.NextName("linear");
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

    [TorchOp("aten::avg_pool1d")]
    public static IOnnxGraphEdge Export(
        this TorchModules.AvgPool1d module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        var strides = TorchHelper.ToLongArray(module.stride);
        var padding = TorchHelper.ToLongArray(module.padding);
        return graph.AveragePool(
            name: graph.NextName("avgpool"),
            options: new AveragePoolInputOptions
            {
                X = input,
                KernelShape = TorchHelper.ToLongArray(module.kernel_size),
                Strides = strides.Length == 0 ? null : strides,
                Pads = padding.Length == 0 ? null : TorchHelper.ExpandPadding(padding, spatialRank: 1),
                CeilMode = module.ceil_mode ? 1 : 0,
                CountIncludePad = module.count_include_pad ? 1 : 0,
            }
        );
    }

    [TorchOp("aten::avg_pool3d")]
    public static IOnnxGraphEdge Export(
        this TorchModules.AvgPool3d module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        var strides = TorchHelper.ToLongArray(module.stride);
        var padding = TorchHelper.ToLongArray(module.padding);
        return graph.AveragePool(
            name: graph.NextName("avgpool"),
            options: new AveragePoolInputOptions
            {
                X = input,
                KernelShape = TorchHelper.ToLongArray(module.kernel_size),
                Strides = strides.Length == 0 ? null : strides,
                Pads = padding.Length == 0 ? null : TorchHelper.ExpandPadding(padding, spatialRank: 3),
                CeilMode = module.ceil_mode ? 1 : 0,
                CountIncludePad = module.count_include_pad ? 1 : 0,
            }
        );
    }

    [TorchOp("aten::flatten.using_ints")]
    public static IOnnxGraphEdge Export(
        this TorchModules.Flatten module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        if (module.start_dim == 1 && module.end_dim == -1)
        {
            return graph.Flatten(
                name: graph.NextName("flatten"),
                options: new FlattenInputOptions
                {
                    Input = input,
                    Axis = module.start_dim,
                }
            );
        }

        throw new NotSupportedException(
            $"Flatten with start_dim={module.start_dim} and end_dim={module.end_dim} is not supported by the recursive module walker."
        );
    }

    [TorchOp("aten::avg_pool2d")] // ???
    public static IOnnxGraphEdge Export(
        this TorchModules.AdaptiveAvgPool2d module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        var outputSize = TorchHelper.ToLongArray(module.output_size);
        if (outputSize.SequenceEqual([1L, 1L]))
        {
            return graph.GlobalAveragePool(
                name: graph.NextName("global_average_pool"),
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

    [TorchOp("aten::embedding")]
    public static IOnnxGraphEdge Export(
        this TorchModules.Embedding module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(module.weight);

        var name = graph.NextName("embedding");

        var weight = graph.AddTensor(
            name: $"{name}_w",
            shape: TorchHelper.GetShape(module.weight),
            value: TorchHelper.GetFloatData(module.weight)
        );

        return graph.Gather(
            name: name,
            options: new GatherInputOptions
            {
                Data = weight,
                Indices = input,
            }
        );
    }

    [TorchOp("aten::lstm.input")]
    public static LSTMOutput Export(
        this TorchModules.LSTM module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(module);
        ArgumentNullException.ThrowIfNull(input);

        var numLayers = checked((int)GetRequiredInt64Member(module, "_num_layers"));
        var hiddenSize = checked((int)GetRequiredInt64Member(module, "_hidden_size"));
        var bidirectional = GetRequiredBoolMember(module, "_bidirectional");
        var batchFirst = GetRequiredBoolMember(module, "_batch_first");

        var numDirections = bidirectional ? 2 : 1;
        var direction = bidirectional ? "bidirectional" : "forward";

        var current = input;

        // PyTorch batch_first=true => [batch, seq, feat]
        // ONNX LSTM expects           [seq, batch, feat]
        if (batchFirst)
        {
            current = graph.Transpose(
                name: graph.NextName("transpose"),
                options: new TransposeInputOptions
                {
                    Data = current,
                    Perm = new long[] { 1, 0, 2 },
                }
            );
        }

        var outputH = new List<IOnnxGraphEdge>();
        var outputC = new List<IOnnxGraphEdge>();

        for (var layer = 0; layer < numLayers; layer++)
        {
            var name = graph.NextName("lstm");

            var flatW = new List<float>();
            var flatR = new List<float>();
            List<float>? flatB = null;

            long inputSize = -1;
            long recurrentInputSize = -1;

            var hasBiases =
                TryGetTensorParameter(module, GetBiasIhName(layer, false), out _) ||
                TryGetTensorParameter(module, GetBiasHhName(layer, false), out _);

            if (hasBiases)
            {
                flatB = new List<float>();
            }

            for (var dir = 0; dir < numDirections; dir++)
            {
                var reverse = dir == 1;

                var weightIh = GetRequiredTensorParameter(module, GetWeightIhName(layer, reverse));
                var weightHh = GetRequiredTensorParameter(module, GetWeightHhName(layer, reverse));

                var weightIhShape = TorchHelper.GetShape(weightIh);
                var weightHhShape = TorchHelper.GetShape(weightHh);

                if (weightIhShape.Length != 2 || weightHhShape.Length != 2)
                {
                    throw new NotSupportedException(
                        $"LSTM weights must be rank-2. Got weight_ih rank={weightIhShape.Length}, weight_hh rank={weightHhShape.Length}."
                    );
                }

                if (weightIhShape[0] != 4L * hiddenSize)
                {
                    throw new NotSupportedException(
                        $"Unexpected weight_ih rows for layer {layer}, dir {dir}. Expected {4L * hiddenSize}, got {weightIhShape[0]}."
                    );
                }

                if (weightHhShape[0] != 4L * hiddenSize)
                {
                    throw new NotSupportedException(
                        $"Unexpected weight_hh rows for layer {layer}, dir {dir}. Expected {4L * hiddenSize}, got {weightHhShape[0]}."
                    );
                }

                if (inputSize < 0)
                {
                    inputSize = weightIhShape[1];
                }
                else if (inputSize != weightIhShape[1])
                {
                    throw new NotSupportedException(
                        $"All directions of the same LSTM layer must have identical input_size. Layer {layer}: got {inputSize} and {weightIhShape[1]}."
                    );
                }

                if (recurrentInputSize < 0)
                {
                    recurrentInputSize = weightHhShape[1];
                }
                else if (recurrentInputSize != weightHhShape[1])
                {
                    throw new NotSupportedException(
                        $"All directions of the same LSTM layer must have identical hidden/recurrent size. Layer {layer}: got {recurrentInputSize} and {weightHhShape[1]}."
                    );
                }

                var reorderedW = ReorderLstmGateMatrix(
                    TorchHelper.GetFloatData(weightIh),
                    hiddenSize,
                    checked((int)weightIhShape[1])
                );

                var reorderedR = ReorderLstmGateMatrix(
                    TorchHelper.GetFloatData(weightHh),
                    hiddenSize,
                    checked((int)weightHhShape[1])
                );

                flatW.AddRange(reorderedW);
                flatR.AddRange(reorderedR);

                if (hasBiases)
                {
                    var biasIh = GetRequiredTensorParameter(module, GetBiasIhName(layer, reverse));
                    var biasHh = GetRequiredTensorParameter(module, GetBiasHhName(layer, reverse));

                    var reorderedBiasIh = ReorderLstmGateVector(
                        TorchHelper.GetFloatData(biasIh),
                        hiddenSize
                    );

                    var reorderedBiasHh = ReorderLstmGateVector(
                        TorchHelper.GetFloatData(biasHh),
                        hiddenSize
                    );

                    flatB!.AddRange(reorderedBiasIh);
                    flatB.AddRange(reorderedBiasHh);
                }
            }

            var w = graph.AddTensor(
                name: $"{name}_W",
                shape: [numDirections, 4L * hiddenSize, inputSize],
                value: flatW.ToArray()
            );

            var r = graph.AddTensor(
                name: $"{name}_R",
                shape: [numDirections, 4L * hiddenSize, recurrentInputSize],
                value: flatR.ToArray()
            );

            IOnnxGraphEdge? b = null;
            if (hasBiases)
            {
                var shape = new long[] { numDirections, 8L * hiddenSize };
                b = graph.AddTensor(
                    name: $"{name}_B",
                    shape: shape,
                    value: flatB!.ToArray()
                );
            }

            var lstm = graph.LSTM(
                name: name,
                options: new LSTMInputOptions
                {
                    X = current,
                    W = w,
                    R = r,
                    B = b,
                    Direction = direction,
                    HiddenSize = hiddenSize,
                }
            );

            var y = lstm.Y
                ?? throw new InvalidOperationException("LSTM export did not produce the Y output.");

            var yh = lstm.YH
                ?? throw new InvalidOperationException("LSTM export did not produce the YH output.");

            var yc = lstm.YC
                ?? throw new InvalidOperationException("LSTM export did not produce the YC output.");

            outputH.Add(yh);
            outputC.Add(yc);

            // ONNX Y: [seq, num_directions, batch, hidden]
            // Torch output: [seq, batch, num_directions * hidden]
            var yTransposed = graph.Transpose(
                name: $"{name}_transpose",
                options: new TransposeInputOptions
                {
                    Data = y,
                    Perm = [0, 2, 1, 3],
                }
            );

            // Reshape with zeros keeps seq and batch dimensions from input tensor.
            var reshapeShape = graph.AddTensor(
                name: $"{name}_shape",
                shape: [3],
                value: [0, 0, numDirections * (long)hiddenSize]
            );

            current = graph.Reshape(
                name: $"{name}_reshape",
                options: new ReshapeInputOptions
                {
                    Data = yTransposed,
                    Shape = reshapeShape,
                }
            );
        }

        if (batchFirst)
        {
            current = graph.Transpose(
                name: graph.NextName("transpose"),
                options: new TransposeInputOptions
                {
                    Data = current,
                    Perm = [1, 0, 2],
                }
            );
        }

        var finalH = outputH.Count == 1
            ? outputH[0]
            : graph.Concat(
                name: graph.NextName("concat"),
                options: new ConcatInputOptions
                {
                    In = outputH.ToArray(),
                    Axis = 0,
                }
            );

        var finalC = outputC.Count == 1
            ? outputC[0]
            : graph.Concat(
                name: graph.NextName("concat"),
                options: new ConcatInputOptions
                {
                    In = outputC.ToArray(),
                    Axis = 0,
                }
            );

        return new LSTMOutput
        {
            Y = current,
            YH = finalH,
            YC = finalC,
        };
    }

    private static string GetWeightIhName(int layer, bool reverse)
        => reverse ? $"weight_ih_l{layer}_reverse" : $"weight_ih_l{layer}";

    private static string GetWeightHhName(int layer, bool reverse)
        => reverse ? $"weight_hh_l{layer}_reverse" : $"weight_hh_l{layer}";

    private static string GetBiasIhName(int layer, bool reverse)
        => reverse ? $"bias_ih_l{layer}_reverse" : $"bias_ih_l{layer}";

    private static string GetBiasHhName(int layer, bool reverse)
        => reverse ? $"bias_hh_l{layer}_reverse" : $"bias_hh_l{layer}";

    private static Tensor GetRequiredTensorParameter(TorchModules.LSTM module, string name)
    {
        if (TryGetTensorParameter(module, name, out var tensor))
        {
            return tensor;
        }

        throw new NotSupportedException($"LSTM parameter '{name}' was not found.");
    }

    private static bool TryGetTensorParameter(TorchModules.LSTM module, string name, out Tensor tensor)
    {
        const BindingFlags Flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

        // 1. Direct property/field lookup for names like weight_ih_l0
        var type = module.GetType();

        var prop = type.GetProperty(name, Flags);
        if (prop?.GetValue(module) is Tensor propTensor)
        {
            tensor = propTensor;
            return true;
        }

        var field = type.GetField(name, Flags);
        if (field?.GetValue(module) is Tensor fieldTensor)
        {
            tensor = fieldTensor;
            return true;
        }

        // 2. Fallback to named_parameters()
        foreach (var entry in module.named_parameters())
        {
            if (TryReadNamedTensor(entry, out var entryName, out var entryTensor)
                && string.Equals(entryName, name, StringComparison.Ordinal))
            {
                tensor = entryTensor;
                return true;
            }
        }

        tensor = null!;
        return false;
    }

    private static bool TryReadNamedTensor(object entry, out string name, out Tensor tensor)
    {
        const BindingFlags Flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

        name = string.Empty;
        tensor = null!;

        if (entry is null)
        {
            return false;
        }

        var type = entry.GetType();

        object? nameValue =
            type.GetProperty("name", Flags)?.GetValue(entry)
            ?? type.GetProperty("Name", Flags)?.GetValue(entry)
            ?? type.GetField("Item1", Flags)?.GetValue(entry)
            ?? type.GetProperty("Key", Flags)?.GetValue(entry);

        object? tensorValue =
            type.GetProperty("parameter", Flags)?.GetValue(entry)
            ?? type.GetProperty("Parameter", Flags)?.GetValue(entry)
            ?? type.GetProperty("tensor", Flags)?.GetValue(entry)
            ?? type.GetProperty("Tensor", Flags)?.GetValue(entry)
            ?? type.GetProperty("Value", Flags)?.GetValue(entry)
            ?? type.GetField("Item2", Flags)?.GetValue(entry);

        if (nameValue is string s && tensorValue is Tensor t)
        {
            name = s;
            tensor = t;
            return true;
        }

        return false;
    }

    private static long GetRequiredInt64Member(object instance, string name)
    {
        if (TryGetMemberValue(instance, name, out var value))
        {
            return Convert.ToInt64(value);
        }

        throw new NotSupportedException($"Required member '{name}' was not found on '{instance.GetType().FullName}'.");
    }

    private static long GetOptionalInt64Member(object instance, string name, long defaultValue = 0)
    {
        if (TryGetMemberValue(instance, name, out var value))
        {
            return Convert.ToInt64(value);
        }

        return defaultValue;
    }

    private static bool GetRequiredBoolMember(object instance, string name)
    {
        if (TryGetMemberValue(instance, name, out var value))
        {
            return Convert.ToBoolean(value);
        }

        throw new NotSupportedException($"Required member '{name}' was not found on '{instance.GetType().FullName}'.");
    }

    private static double GetOptionalDoubleMember(object instance, double defaultValue, params string[] names)
    {
        foreach (var name in names)
        {
            if (TryGetMemberValue(instance, name, out var value))
            {
                return Convert.ToDouble(value);
            }
        }

        return defaultValue;
    }

    private static bool TryGetMemberValue(object instance, string name, out object value)
    {
        const BindingFlags Flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

        var type = instance.GetType();

        var prop = type.GetProperty(name, Flags);
        if (prop is not null)
        {
            value = prop.GetValue(instance)!;
            return true;
        }

        var field = type.GetField(name, Flags);
        if (field is not null)
        {
            value = field.GetValue(instance)!;
            return true;
        }

        value = null!;
        return false;
    }

    // PyTorch LSTM gate order: [i, f, g, o]
    // ONNX LSTM gate order:    [i, o, f, g]
    private static float[] ReorderLstmGateMatrix(float[] source, int hiddenSize, int width)
    {
        ArgumentNullException.ThrowIfNull(source);

        var expectedLength = 4 * hiddenSize * width;
        if (source.Length != expectedLength)
        {
            throw new ArgumentException(
                $"Unexpected matrix length. Expected {expectedLength}, got {source.Length}.",
                nameof(source)
            );
        }

        var result = new float[source.Length];
        var gateBlockLength = hiddenSize * width;

        CopyGateBlock(source, result, sourceGate: 0, targetGate: 0, gateBlockLength); // i -> i
        CopyGateBlock(source, result, sourceGate: 3, targetGate: 1, gateBlockLength); // o -> o
        CopyGateBlock(source, result, sourceGate: 1, targetGate: 2, gateBlockLength); // f -> f
        CopyGateBlock(source, result, sourceGate: 2, targetGate: 3, gateBlockLength); // g -> g

        return result;
    }

    private static float[] ReorderLstmGateVector(float[] source, int hiddenSize)
    {
        ArgumentNullException.ThrowIfNull(source);

        var expectedLength = 4 * hiddenSize;
        if (source.Length != expectedLength)
        {
            throw new ArgumentException(
                $"Unexpected vector length. Expected {expectedLength}, got {source.Length}.",
                nameof(source)
            );
        }

        var result = new float[source.Length];

        CopyGateBlock(source, result, sourceGate: 0, targetGate: 0, hiddenSize); // i -> i
        CopyGateBlock(source, result, sourceGate: 3, targetGate: 1, hiddenSize); // o -> o
        CopyGateBlock(source, result, sourceGate: 1, targetGate: 2, hiddenSize); // f -> f
        CopyGateBlock(source, result, sourceGate: 2, targetGate: 3, hiddenSize); // g -> g

        return result;
    }

    private static void CopyGateBlock(
        float[] source,
        float[] destination,
        int sourceGate,
        int targetGate,
        int gateBlockLength
    )
    {
        Array.Copy(
            sourceArray: source,
            sourceIndex: sourceGate * gateBlockLength,
            destinationArray: destination,
            destinationIndex: targetGate * gateBlockLength,
            length: gateBlockLength
        );
    }
}


