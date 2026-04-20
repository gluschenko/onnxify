using System.Reflection;
using System.Runtime.CompilerServices;
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
            TorchModules.BatchNorm1d m => m.Export(graph, input),
            TorchModules.BatchNorm2d m => m.Export(graph, input),
            TorchModules.BatchNorm3d m => m.Export(graph, input),
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
            TorchModules.GELU m => m.Export(graph, input),
            TorchModules.Mish m => m.Export(graph, input),
            TorchModules.SELU m => m.Export(graph, input),
            TorchModules.Softplus m => m.Export(graph, input),
            TorchModules.PReLU m => m.Export(graph, input),
            TorchModules.PixelShuffle m => m.Export(graph, input),
            TorchModules.PixelUnshuffle m => m.Export(graph, input),
            TorchModules.ReflectionPad1d m => m.Export(graph, input),
            TorchModules.ReflectionPad2d m => m.Export(graph, input),
            TorchModules.MaxPool2d m => m.Export(graph, input),
            TorchModules.MaxPool1d m => m.Export(graph, input),
            TorchModules.MaxPool3d m => m.Export(graph, input),
            TorchModules.Dropout m => m.Export(graph, input),
            TorchModules.Linear m => m.Export(graph, input),
            TorchModules.AvgPool1d m => m.Export(graph, input),
            TorchModules.AvgPool2d m => m.Export(graph, input),
            TorchModules.AvgPool3d m => m.Export(graph, input),
            TorchModules.Flatten m => m.Export(graph, input),
            TorchModules.AdaptiveAvgPool2d m => m.Export(graph, input),
            TorchModules.Embedding m => m.Export(graph, input),
            TorchModules.GLU m => m.Export(graph, input),
            TorchModules.GroupNorm m => m.Export(graph, input),
            TorchModules.LayerNorm m => m.Export(graph, input),
            TorchModules.Upsample m => m.Export(graph, input),
            TorchModules.ReflectionPad3d m => m.Export(graph, input),
            TorchModules.ReplicationPad1d m => m.Export(graph, input),
            TorchModules.ReplicationPad2d m => m.Export(graph, input),
            TorchModules.ReplicationPad3d m => m.Export(graph, input),
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

    private static IOnnxGraphEdge ExportBatchNorm(
        OnnxGraph graph,
        IOnnxGraphEdge input,
        Tensor? weight,
        Tensor? bias,
        Tensor? runningMean,
        Tensor? runningVar,
        double epsilon,
        bool training
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        if (training)
        {
            throw new NotSupportedException("BatchNorm export only supports inference mode.");
        }

        if (runningMean is null || runningMean.IsInvalid || runningVar is null || runningVar.IsInvalid)
        {
            throw new NotSupportedException("BatchNorm export requires valid running statistics.");
        }

        var name = graph.NextName("batch_norm");
        var channelCount = checked((int)runningMean.shape[0]);

        var scale = graph.AddTensor(
            name: $"{name}_scale",
            shape: [channelCount],
            value: weight is not null && !weight.IsInvalid
                ? TorchHelper.GetFloatData(weight)
                : Enumerable.Repeat(1f, channelCount).ToArray()
        );

        var shift = graph.AddTensor(
            name: $"{name}_bias",
            shape: [channelCount],
            value: bias is not null && !bias.IsInvalid
                ? TorchHelper.GetFloatData(bias)
                : new float[channelCount]
        );

        var mean = graph.AddTensor(
            name: $"{name}_mean",
            shape: TorchHelper.GetShape(runningMean),
            value: TorchHelper.GetFloatData(runningMean)
        );

        var variance = graph.AddTensor(
            name: $"{name}_var",
            shape: TorchHelper.GetShape(runningVar),
            value: TorchHelper.GetFloatData(runningVar)
        );

        var output = graph.AddEdge($"{name}_output");
        var op = new BatchNormalization(
            name: name,
            options: new BatchNormalizationInputOutputOptions
            {
                X = input,
                Scale = scale,
                B = shift,
                InputMean = mean,
                InputVar = variance,
                Epsilon = (float)epsilon,
                Momentum = 0.9f,
                TrainingMode = 0,
                Y = output,
                RunningMean = null,
                RunningVar = null,
            }
        );

        graph.AddNode(op);
        return op.Y;
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

    [TorchOp("aten::_native_batch_norm_legit")]
    [TorchOp("aten::_native_batch_norm_legit.no_stats")]
    [TorchOp("aten::_native_batch_norm_legit_functional")]
    [TorchOp("aten::_native_batch_norm_legit_no_training")]
    [TorchOp("aten::native_batch_norm")]
    public static IOnnxGraphEdge Export(
        this TorchModules.BatchNorm1d module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        return ExportBatchNorm(
            graph: graph,
            input: input,
            weight: module.weight,
            bias: module.bias,
            runningMean: module.running_mean,
            runningVar: module.running_var,
            epsilon: module.eps,
            training: module.training
        );
    }

    [TorchOp("aten::_native_batch_norm_legit")]
    [TorchOp("aten::_native_batch_norm_legit.no_stats")]
    [TorchOp("aten::_native_batch_norm_legit_functional")]
    [TorchOp("aten::_native_batch_norm_legit_no_training")]
    [TorchOp("aten::native_batch_norm")]
    public static IOnnxGraphEdge Export(
        this TorchModules.BatchNorm2d module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        return ExportBatchNorm(
            graph: graph,
            input: input,
            weight: module.weight,
            bias: module.bias,
            runningMean: module.running_mean,
            runningVar: module.running_var,
            epsilon: module.eps,
            training: module.training
        );
    }

    [TorchOp("aten::_native_batch_norm_legit")]
    [TorchOp("aten::_native_batch_norm_legit.no_stats")]
    [TorchOp("aten::_native_batch_norm_legit_functional")]
    [TorchOp("aten::_native_batch_norm_legit_no_training")]
    [TorchOp("aten::native_batch_norm")]
    public static IOnnxGraphEdge Export(
        this TorchModules.BatchNorm3d module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        return ExportBatchNorm(
            graph: graph,
            input: input,
            weight: module.weight,
            bias: module.bias,
            runningMean: module.running_mean,
            runningVar: module.running_var,
            epsilon: module.eps,
            training: module.training
        );
    }

    [TorchOp("aten::upsample_nearest1d")]
    [TorchOp("aten::upsample_nearest1d.vec")]
    [TorchOp("aten::upsample_nearest2d")]
    [TorchOp("aten::upsample_nearest2d.vec")]
    [TorchOp("aten::upsample_nearest3d")]
    [TorchOp("aten::upsample_nearest3d.vec")]
    public static IOnnxGraphEdge Export(
        this TorchModules.Upsample module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(module);

        if (module.training)
        {
            throw new NotSupportedException("Upsample export only supports inference mode.");
        }

        if (module.mode != global::TorchSharp.torch.UpsampleMode.Nearest)
        {
            throw new NotSupportedException(
                $"Upsample export currently supports only 'Nearest', got '{module.mode}'."
            );
        }

        var name = graph.NextName("resize");
        var spatialSizes = module.size.ToArray();
        var spatialScales = module.scale_factor.ToArray();

        if (spatialSizes.Length == 0 && spatialScales.Length == 0)
        {
            throw new NotSupportedException("Upsample export requires either 'size' or 'scale_factor'.");
        }

        var spatialRank = spatialSizes.Length != 0 ? spatialSizes.Length : spatialScales.Length;
        var axes = Enumerable.Range(2, spatialRank).Select(x => (long)x).ToArray();

        IOnnxGraphEdge? sizes = null;
        if (spatialSizes.Length != 0)
        {
            sizes = graph.AddTensor(
                name: $"{name}_sizes",
                shape: [spatialSizes.Length],
                value: spatialSizes
            );
        }

        IOnnxGraphEdge? scales = null;
        if (sizes is null)
        {
            scales = graph.AddTensor(
                name: $"{name}_scales",
                shape: [spatialScales.Length],
                value: Array.ConvertAll(spatialScales, x => (float)x)
            );
        }

        return graph.Resize(
            name: name,
            options: new ResizeInputOptions
            {
                X = input,
                Sizes = sizes,
                Scales = scales,
                Axes = axes,
                CoordinateTransformationMode = "asymmetric",
                Mode = "nearest",
                NearestMode = "floor",
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
    [TorchOp("aten::_softmax")]
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
    [TorchOp("aten::_log_softmax")]
    [TorchOp("aten::special_log_softmax")]
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

    [TorchOp("aten::gelu")]
    public static IOnnxGraphEdge Export(
        this TorchModules.GELU module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        var approximate = GetOptionalStringMember(module, defaultValue: "none", "approximate", "_approximate");

        return graph.Gelu(
            name: graph.NextName("gelu"),
            options: new GeluInputOptions
            {
                X = input,
                Approximate = approximate,
            }
        );
    }

    [TorchOp("aten::mish")]
    public static IOnnxGraphEdge Export(
        this TorchModules.Mish module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        return graph.Mish(
            name: graph.NextName("mish"),
            options: new MishInputOptions
            {
                X = input,
            }
        );
    }

    [TorchOp("aten::selu")]
    public static IOnnxGraphEdge Export(
        this TorchModules.SELU module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        return graph.Selu(
            name: graph.NextName("selu"),
            options: new SeluInputOptions
            {
                X = input,
                Alpha = 1.6732632f,
                Gamma = 1.050701f,
            }
        );
    }

    [TorchOp("aten::softplus")]
    public static IOnnxGraphEdge Export(
        this TorchModules.Softplus module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        var beta = GetOptionalDoubleMember(module, defaultValue: 1.0, "beta", "_beta");
        var threshold = GetOptionalDoubleMember(module, defaultValue: 20.0, "threshold", "_threshold");

        if (Math.Abs(beta - 1.0) > 1e-6 || Math.Abs(threshold - 20.0) > 1e-6)
        {
            throw new NotSupportedException(
                $"Softplus export currently supports only beta=1 and threshold=20, got beta={beta}, threshold={threshold}."
            );
        }

        return graph.Softplus(
            name: graph.NextName("softplus"),
            options: new SoftplusInputOptions
            {
                X = input,
            }
        );
    }

    [TorchOp("aten::prelu")]
    public static IOnnxGraphEdge Export(
        this TorchModules.PReLU module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        var name = graph.NextName("prelu");
        var weightTensor = GetRequiredTensorMember(module, "weight", "_weight");
        var weightShape = TorchHelper.GetShape(weightTensor);

        IOnnxGraphEdge slope = graph.AddTensor(
            name: $"{name}_slope",
            shape: weightShape,
            value: TorchHelper.GetFloatData(weightTensor)
        );

        var inputRank = TryGetTensorRank(input);
        if (inputRank is >= 2 && weightShape.Length == 1)
        {
            var reshapeShapeValues = new long[inputRank.Value];
            reshapeShapeValues[0] = 1;
            reshapeShapeValues[1] = -1;
            for (var i = 2; i < reshapeShapeValues.Length; i++)
            {
                reshapeShapeValues[i] = 1;
            }

            var reshapeShape = graph.AddTensor(
                name: $"{name}_slope_shape",
                shape: [reshapeShapeValues.Length],
                value: reshapeShapeValues
            );

            slope = graph.Reshape(
                name: $"{name}_slope_reshape",
                options: new ReshapeInputOptions
                {
                    Data = slope,
                    Shape = reshapeShape,
                }
            );
        }

        return graph.PRelu(
            name: name,
            options: new PReluInputOptions
            {
                X = input,
                Slope = slope,
            }
        );
    }

    [TorchOp("aten::pixel_shuffle")]
    public static IOnnxGraphEdge Export(
        this TorchModules.PixelShuffle module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        var upscaleFactor = GetRequiredInt64Member(module, "upscale_factor", "_upscale_factor");

        return graph.DepthToSpace(
            name: graph.NextName("pixel_shuffle"),
            options: new DepthToSpaceInputOptions
            {
                Input = input,
                Blocksize = upscaleFactor,
                Mode = "CRD",
            }
        );
    }

    [TorchOp("aten::pixel_unshuffle")]
    public static IOnnxGraphEdge Export(
        this TorchModules.PixelUnshuffle module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        var downscaleFactor = GetRequiredInt64Member(module, "downscale_factor", "_downscale_factor");

        return graph.SpaceToDepth(
            name: graph.NextName("pixel_unshuffle"),
            options: new SpaceToDepthInputOptions
            {
                Input = input,
                Blocksize = downscaleFactor,
            }
        );
    }

    [TorchOp("aten::reflection_pad1d")]
    public static IOnnxGraphEdge Export(
        this TorchModules.ReflectionPad1d module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        return ExportPad(
            module: module,
            graph: graph,
            input: input,
            prefix: "reflection_pad",
            mode: "reflect",
            spatialRank: 1
        );
    }

    [TorchOp("aten::reflection_pad2d")]
    public static IOnnxGraphEdge Export(
        this TorchModules.ReflectionPad2d module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        return ExportPad(
            module: module,
            graph: graph,
            input: input,
            prefix: "reflection_pad",
            mode: "reflect",
            spatialRank: 2
        );
    }

    [TorchOp("aten::reflection_pad3d")]
    public static IOnnxGraphEdge Export(
        this TorchModules.ReflectionPad3d module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        return ExportPad(
            module: module,
            graph: graph,
            input: input,
            prefix: "reflection_pad",
            mode: "reflect",
            spatialRank: 3
        );
    }

    [TorchOp("aten::replication_pad1d")]
    public static IOnnxGraphEdge Export(
        this TorchModules.ReplicationPad1d module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        return ExportPad(
            module: module,
            graph: graph,
            input: input,
            prefix: "replication_pad",
            mode: "edge",
            spatialRank: 1
        );
    }

    [TorchOp("aten::replication_pad2d")]
    public static IOnnxGraphEdge Export(
        this TorchModules.ReplicationPad2d module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        return ExportPad(
            module: module,
            graph: graph,
            input: input,
            prefix: "replication_pad",
            mode: "edge",
            spatialRank: 2
        );
    }

    [TorchOp("aten::replication_pad3d")]
    public static IOnnxGraphEdge Export(
        this TorchModules.ReplicationPad3d module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        return ExportPad(
            module: module,
            graph: graph,
            input: input,
            prefix: "replication_pad",
            mode: "edge",
            spatialRank: 3
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
        var kernelShape = TorchHelper.ToLongArray(module.kernel_size);
        var strides = ResolvePoolStrides(TorchHelper.ToLongArray(module.stride), kernelShape);
        var result = graph.MaxPool(
            name: graph.NextName("maxpool"),
            options: new MaxPoolInputOptions
            {
                X = input,
                KernelShape = kernelShape,
                Strides = strides,
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
        var kernelShape = TorchHelper.ToLongArray(module.kernel_size);
        var strides = ResolvePoolStrides(TorchHelper.ToLongArray(module.stride), kernelShape);
        var padding = TorchHelper.ToLongArray(module.padding);
        var dilations = TorchHelper.ToLongArray(module.dilation);
        var result = graph.MaxPool(
            name: graph.NextName("maxpool"),
            options: new MaxPoolInputOptions
            {
                X = input,
                KernelShape = kernelShape,
                Strides = strides,
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
        var kernelShape = TorchHelper.ToLongArray(module.kernel_size);
        var strides = ResolvePoolStrides(TorchHelper.ToLongArray(module.stride), kernelShape);
        var padding = TorchHelper.ToLongArray(module.padding);
        var dilations = TorchHelper.ToLongArray(module.dilation);
        var result = graph.MaxPool(
            name: graph.NextName("maxpool"),
            options: new MaxPoolInputOptions
            {
                X = input,
                KernelShape = kernelShape,
                Strides = strides,
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
        var kernelShape = TorchHelper.ToLongArray(module.kernel_size);
        var strides = ResolvePoolStrides(TorchHelper.ToLongArray(module.stride), kernelShape);
        var padding = TorchHelper.ToLongArray(module.padding);
        return graph.AveragePool(
            name: graph.NextName("avgpool"),
            options: new AveragePoolInputOptions
            {
                X = input,
                KernelShape = kernelShape,
                Strides = strides,
                Pads = padding.Length == 0 ? null : TorchHelper.ExpandPadding(padding, spatialRank: 1),
                CeilMode = module.ceil_mode ? 1 : 0,
                CountIncludePad = module.count_include_pad ? 1 : 0,
            }
        );
    }

    [TorchOp("aten::avg_pool2d")]
    public static IOnnxGraphEdge Export(
        this TorchModules.AvgPool2d module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        var kernelShape = TorchHelper.ToLongArray(module.kernel_size);
        var strides = ResolvePoolStrides(TorchHelper.ToLongArray(module.stride), kernelShape);
        var padding = TorchHelper.ToLongArray(module.padding);
        return graph.AveragePool(
            name: graph.NextName("avgpool"),
            options: new AveragePoolInputOptions
            {
                X = input,
                KernelShape = kernelShape,
                Strides = strides,
                Pads = padding.Length == 0 ? null : TorchHelper.ExpandPadding(padding, spatialRank: 2),
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
        var kernelShape = TorchHelper.ToLongArray(module.kernel_size);
        var strides = ResolvePoolStrides(TorchHelper.ToLongArray(module.stride), kernelShape);
        var padding = TorchHelper.ToLongArray(module.padding);
        return graph.AveragePool(
            name: graph.NextName("avgpool"),
            options: new AveragePoolInputOptions
            {
                X = input,
                KernelShape = kernelShape,
                Strides = strides,
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

    [TorchOp("aten::adaptive_avg_pool2d")]
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

    [TorchOp("aten::layer_norm")]
    public static IOnnxGraphEdge Export(
        this TorchModules.LayerNorm module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        var normalizedShape = module.normalized_shape;
        if (normalizedShape.Length == 0)
        {
            throw new NotSupportedException("LayerNorm requires a non-empty normalized_shape.");
        }

        var name = graph.NextName("layer_norm");

        var scale = module.weight is not null
            ? graph.AddTensor(
                name: $"{name}_scale",
                shape: TorchHelper.GetShape(module.weight),
                value: TorchHelper.GetFloatData(module.weight)
            )
            : graph.AddTensor(
                name: $"{name}_scale",
                shape: normalizedShape,
                value: CreateFilledFloatArray(normalizedShape, 1f)
            );

        IOnnxGraphEdge? bias = module.bias is not null
            ? graph.AddTensor(
                name: $"{name}_bias",
                shape: TorchHelper.GetShape(module.bias),
                value: TorchHelper.GetFloatData(module.bias)
            )
            : null;

        return graph.LayerNormalization(
            name: name,
            options: new LayerNormalizationInputOptions
            {
                X = input,
                Scale = scale,
                B = bias,
                Axis = -normalizedShape.Length,
                Epsilon = (float)module.eps,
            }
        ).Y;
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

    [TorchOp("aten::glu")]
    public static IOnnxGraphEdge Export(
        this TorchModules.GLU module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        var dim = GetRequiredInt64Member(module, "dim", "_dim");
        var first = graph.AddEdge(graph.NextName("glu_first"));
        var second = graph.AddEdge(graph.NextName("glu_second"));

        _ = graph.Split(
            name: graph.NextName("glu_split"),
            options: new SplitInputOutputOptions
            {
                Input = input,
                Out = [first, second],
                Axis = dim,
                NumOutputs = 2,
            }
        );

        var gate = graph.Sigmoid(
            name: graph.NextName("glu_sigmoid"),
            options: new SigmoidInputOptions
            {
                X = second,
            }
        );

        return graph.Mul(
            name: graph.NextName("glu_mul"),
            options: new MulInputOptions
            {
                A = first,
                B = gate,
            }
        );
    }

    [TorchOp("aten::group_norm")]
    public static IOnnxGraphEdge Export(
        this TorchModules.GroupNorm module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        var numGroups = GetRequiredInt64Member(module, "num_groups", "_num_groups");
        var epsilon = (float)GetOptionalDoubleMember(module, 1e-5, "eps", "_eps");
        var name = graph.NextName("group_norm");
        TryGetTensorMember(module, out var weight, "weight", "_weight");
        TryGetTensorMember(module, out var biasTensor, "bias", "_bias");

        var channelCount = weight is not null
            ? TorchHelper.GetShape(weight)[0]
            : biasTensor is not null
                ? TorchHelper.GetShape(biasTensor)[0]
                : GetRequiredChannelCount(input);

        var scale = weight is not null
            ? graph.AddTensor(
                name: $"{name}_scale",
                shape: TorchHelper.GetShape(weight),
                value: TorchHelper.GetFloatData(weight)
            )
            : graph.AddTensor(
                name: $"{name}_scale",
                shape: [channelCount],
                value: CreateFilledFloatArray([channelCount], 1f)
            );

        var bias = biasTensor is not null
            ? graph.AddTensor(
                name: $"{name}_bias",
                shape: TorchHelper.GetShape(biasTensor),
                value: TorchHelper.GetFloatData(biasTensor)
            )
            : graph.AddTensor(
                name: $"{name}_bias",
                shape: [channelCount],
                value: CreateFilledFloatArray([channelCount], 0f)
            );

        return graph.GroupNormalization(
            name: name,
            options: new GroupNormalizationInputOptions
            {
                X = input,
                Scale = scale,
                Bias = bias,
                NumGroups = numGroups,
                Epsilon = epsilon,
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
        // TorchSharp state_dict() already provides the named parameter tensors we need here.
        foreach (var (entryName, entryTensor) in module.state_dict())
        {
            if (string.Equals(entryName, name, StringComparison.Ordinal)
                && entryTensor is not null
                && !entryTensor.IsInvalid)
            {
                tensor = entryTensor;
                return true;
            }
        }

        tensor = null!;
        return false;
    }

    private static long GetRequiredInt64Member(object instance, string name)
    {
        return GetRequiredInt64Member(instance, [name]);
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

    private static long GetRequiredInt64Member(object instance, params string[] names)
    {
        foreach (var name in names)
        {
            if (TryGetMemberValue(instance, name, out var value))
            {
                return Convert.ToInt64(value);
            }
        }

        throw new NotSupportedException(
            $"Required member '{string.Join("' or '", names)}' was not found on '{instance.GetType().FullName}'."
        );
    }

    private static string GetOptionalStringMember(object instance, string defaultValue, params string[] names)
    {
        foreach (var name in names)
        {
            if (TryGetMemberValue(instance, name, out var value))
            {
                return Convert.ToString(value) ?? defaultValue;
            }
        }

        return defaultValue;
    }

    private static Tensor GetRequiredTensorMember(object instance, params string[] names)
    {
        foreach (var name in names)
        {
            if (TryGetMemberValue(instance, name, out var value) && value is Tensor tensor)
            {
                return tensor;
            }
        }

        throw new NotSupportedException(
            $"Required tensor member '{string.Join("' or '", names)}' was not found on '{instance.GetType().FullName}'."
        );
    }

    private static bool TryGetTensorMember(object instance, out Tensor? tensor, params string[] names)
    {
        foreach (var name in names)
        {
            if (TryGetMemberValue(instance, name, out var value) && value is Tensor memberTensor)
            {
                tensor = memberTensor;
                return true;
            }
        }

        tensor = null;
        return false;
    }

    private static long[] GetRequiredInt64ArrayMember(object instance, params string[] names)
    {
        foreach (var name in names)
        {
            if (TryGetMemberValue(instance, name, out var value))
            {
                return ConvertToLongArray(value);
            }
        }

        throw new NotSupportedException(
            $"Required array member '{string.Join("' or '", names)}' was not found on '{instance.GetType().FullName}'."
        );
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

    private static IOnnxGraphEdge ExportPad(
        object module,
        OnnxGraph graph,
        IOnnxGraphEdge input,
        string prefix,
        string mode,
        int spatialRank
    )
    {
        var name = graph.NextName(prefix);
        var padding = GetRequiredInt64ArrayMember(module, "padding", "_padding");
        var padsTensor = graph.AddTensor(
            name: $"{name}_pads",
            shape: [2L * (spatialRank + 2)],
            value: CreatePadVector(padding, spatialRank)
        );

        return graph.Pad(
            name: name,
            options: new PadInputOptions
            {
                Data = input,
                Pads = padsTensor,
                Mode = mode,
            }
        );
    }

    private static long[] CreatePadVector(long[] padding, int spatialRank)
    {
        return spatialRank switch
        {
            1 => CreatePadVector1d(padding),
            2 => CreatePadVector2d(padding),
            3 => CreatePadVector3d(padding),
            _ => throw new NotSupportedException($"Unsupported pad spatial rank: {spatialRank}."),
        };
    }

    private static long[] CreatePadVector1d(long[] padding)
    {
        var normalized = padding.Length switch
        {
            1 => new[] { padding[0], padding[0] },
            2 => padding,
            _ => throw new NotSupportedException(
                $"Unsupported 1D padding shape: [{string.Join(", ", padding)}]."
            ),
        };

        return [0, 0, normalized[0], 0, 0, normalized[1]];
    }

    private static long[] CreatePadVector2d(long[] padding)
    {
        var normalized = padding.Length switch
        {
            1 => new[] { padding[0], padding[0], padding[0], padding[0] },
            2 => new[] { padding[0], padding[0], padding[1], padding[1] },
            4 => padding,
            _ => throw new NotSupportedException(
                $"Unsupported 2D padding shape: [{string.Join(", ", padding)}]."
            ),
        };

        var left = normalized[0];
        var right = normalized[1];
        var top = normalized[2];
        var bottom = normalized[3];

        return [0, 0, top, left, 0, 0, bottom, right];
    }

    private static long[] CreatePadVector3d(long[] padding)
    {
        var normalized = padding.Length switch
        {
            1 => [padding[0], padding[0], padding[0], padding[0], padding[0], padding[0]],
            3 => [padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]],
            6 => padding,
            _ => throw new NotSupportedException(
                $"Unsupported 3D padding shape: [{string.Join(", ", padding)}]."
            ),
        };

        var left = normalized[0];
        var right = normalized[1];
        var top = normalized[2];
        var bottom = normalized[3];
        var front = normalized[4];
        var back = normalized[5];

        return [0, 0, front, top, left, 0, 0, back, bottom, right];
    }

    private static long[] ConvertToLongArray(object value)
    {
        ArgumentNullException.ThrowIfNull(value);

        if (value is long[] longArray)
        {
            return longArray;
        }

        if (value is int[] intArray)
        {
            return [.. intArray.Select(x => (long)x)];
        }

        if (value is IEnumerable<long> longEnumerable)
        {
            return [.. longEnumerable];
        }

        if (value is System.Collections.IEnumerable enumerable and not string)
        {
            var values = new List<long>();
            foreach (var item in enumerable)
            {
                values.Add(Convert.ToInt64(item));
            }

            return [.. values];
        }

        if (value is ITuple tuple)
        {
            var values = new long[tuple.Length];
            for (var i = 0; i < tuple.Length; i++)
            {
                values[i] = Convert.ToInt64(tuple[i]);
            }

            return values;
        }

        return [Convert.ToInt64(value)];
    }

    private static float[] CreateFilledFloatArray(long[] shape, float value)
    {
        var elementCount = checked((int)shape.Aggregate(1L, (current, dimension) => current * dimension));
        var data = new float[elementCount];
        Array.Fill(data, value);
        return data;
    }

    private static long[] ResolvePoolStrides(long[] strides, long[] kernelShape)
    {
        return strides.Length == 0 ? kernelShape : strides;
    }

    private static int? TryGetTensorRank(IOnnxGraphEdge edge)
    {
        if (edge is OnnxValue<OnnxTensorType> value && value.Type.Shape is not null)
        {
            return value.Type.Shape.Dimensions.Length;
        }

        return null;
    }

    private static long GetRequiredChannelCount(IOnnxGraphEdge edge)
    {
        if (edge is OnnxValue<OnnxTensorType> value && value.Type.Shape is not null && value.Type.Shape.Dimensions.Length >= 2)
        {
            if (value.Type.Shape.Dimensions[1] is OnnxDimension<long> channelDimension)
            {
                return channelDimension.Value;
            }
        }

        throw new NotSupportedException(
            "GroupNorm export requires either affine weights/biases or a statically known channel dimension on the input."
        );
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


