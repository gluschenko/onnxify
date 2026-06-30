using System.Reflection;
using System.Runtime.CompilerServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Onnxify.TorchSharp;

public static class TorchExportEvaluationExtensions
{
    public static EvaluationResult<TTorchModule, OnnxModel> Evaluate<TTorchModule, TInput>(
        this TTorchModule torchModule,
        OnnxModel onnxModel,
        IReadOnlyList<TInput> inputs,
        TorchExportEvaluationOptions? options = null
    )
        where TTorchModule : global::TorchSharp.torch.nn.Module
    {
        ArgumentNullException.ThrowIfNull(torchModule);
        ArgumentNullException.ThrowIfNull(onnxModel);
        ArgumentNullException.ThrowIfNull(inputs);

        options ??= new TorchExportEvaluationOptions();

        var diagnostics = new List<EvaluationDiagnostic>();
        var sampleResults = new List<EvaluationSampleResult>();
        var modelPath = CreateTemporaryModelPath(options);

        try
        {
            onnxModel.Save(modelPath, overwrite: true);

            using var session = options.SessionOptions is null
                ? new InferenceSession(modelPath)
                : new InferenceSession(modelPath, options.SessionOptions);

            for (var index = 0; index < inputs.Count; index++)
            {
                try
                {
                    sampleResults.Add(EvaluateSample(
                        torchModule: torchModule,
                        onnxModel: onnxModel,
                        session: session,
                        input: inputs[index],
                        sampleIndex: index,
                        options: options,
                        diagnostics: diagnostics
                    ));
                }
                catch (Exception exception) when (
                    exception is OnnxRuntimeException
                        or InvalidOperationException
                        or NotSupportedException
                        or TargetInvocationException
                )
                {
                    var diagnostic = new EvaluationDiagnostic(
                        severity: EvaluationDiagnosticSeverity.Error,
                        code: "SampleEvaluationFailed",
                        message: exception.InnerException?.Message ?? exception.Message,
                        sampleIndex: index,
                        outputName: null
                    );
                    diagnostics.Add(diagnostic);
                    ThrowIfRequested(options, diagnostic);
                    sampleResults.Add(new EvaluationSampleResult(
                        sampleIndex: index,
                        loss: double.NaN,
                        outputs: [],
                        diagnostics: [diagnostic],
                        passed: false
                    ));
                }
            }
        }
        catch (Exception exception) when (
            exception is OnnxRuntimeException
                or InvalidOperationException
                or NotSupportedException
                or TargetInvocationException
        )
        {
            var diagnostic = new EvaluationDiagnostic(
                severity: EvaluationDiagnosticSeverity.Error,
                code: "RuntimeExecutionFailed",
                message: exception.InnerException?.Message ?? exception.Message,
                sampleIndex: null,
                outputName: null
            );
            diagnostics.Add(diagnostic);
            ThrowIfRequested(options, diagnostic);
        }
        finally
        {
            DeleteIfExists(modelPath);
        }

        var loss = sampleResults.Count == 0
            ? double.NaN
            : sampleResults.Average(static sample => sample.Loss);
        var passed = diagnostics.All(static diagnostic => diagnostic.Severity != EvaluationDiagnosticSeverity.Error)
            && sampleResults.All(static sample => sample.Passed);

        return new EvaluationResult<TTorchModule, OnnxModel>(
            torchModule: torchModule,
            onnxModel: onnxModel,
            loss: loss,
            samples: sampleResults,
            diagnostics: diagnostics,
            passed: passed
        );
    }

    private static EvaluationSampleResult EvaluateSample<TTorchModule, TInput>(
        TTorchModule torchModule,
        OnnxModel onnxModel,
        InferenceSession session,
        TInput input,
        int sampleIndex,
        TorchExportEvaluationOptions options,
        List<EvaluationDiagnostic> diagnostics
    )
        where TTorchModule : global::TorchSharp.torch.nn.Module
    {
        var sampleDiagnostics = new List<EvaluationDiagnostic>();
        using var noGrad = global::TorchSharp.torch.no_grad();
        var torchOutput = InvokeForward(torchModule, onnxModel, input);
        var expectedOutputs = GetTorchOutputs(torchOutput, onnxModel, options);
        var runtimeInputs = CreateRuntimeInputs(input, onnxModel, options);

        using var results = session.Run(runtimeInputs);
        var actualOutputs = results.ToDictionary(
            static output => output.Name,
            static output => output.Value,
            StringComparer.Ordinal
        );

        var outputResults = new List<EvaluationOutputResult>();
        foreach (var expected in expectedOutputs)
        {
            if (!actualOutputs.TryGetValue(expected.Name, out var actualValue))
            {
                var diagnostic = AddDiagnostic(
                    diagnostics,
                    sampleDiagnostics,
                    options,
                    EvaluationDiagnosticSeverity.Error,
                    "MissingOnnxOutput",
                    $"ONNX Runtime output '{expected.Name}' was not returned.",
                    sampleIndex,
                    expected.Name
                );
                outputResults.Add(EvaluationOutputResult.Failed(expected.Name, diagnostic.Message));
                continue;
            }

            var actual = CreateComparableTensor(expected.Name, actualValue);
            var context = new TorchExportEvaluationComparisonContext(
                sampleIndex: sampleIndex,
                outputName: expected.Name,
                expected: expected,
                actual: actual,
                options: options
            );

            var outputResult = options.Comparer is null
                ? CompareDefault(context)
                : options.Comparer(context);

            outputResults.Add(outputResult);
            if (!outputResult.Passed)
            {
                AddDiagnostic(
                    diagnostics,
                    sampleDiagnostics,
                    options,
                    EvaluationDiagnosticSeverity.Error,
                    "OutputComparisonFailed",
                    outputResult.Message ?? $"Output '{expected.Name}' failed comparison.",
                    sampleIndex,
                    expected.Name
                );
            }
        }

        var loss = outputResults.Count == 0
            ? double.NaN
            : outputResults.Average(static output => output.Loss);
        var passed = sampleDiagnostics.All(static diagnostic => diagnostic.Severity != EvaluationDiagnosticSeverity.Error)
            && outputResults.All(static output => output.Passed);

        return new EvaluationSampleResult(
            sampleIndex: sampleIndex,
            loss: loss,
            outputs: outputResults,
            diagnostics: sampleDiagnostics,
            passed: passed
        );
    }

    private static object? InvokeForward<TInput>(
        global::TorchSharp.torch.nn.Module torchModule,
        OnnxModel onnxModel,
        TInput input
    )
    {
        var arguments = GetForwardArguments(input, onnxModel);
        var forward = torchModule.GetType()
            .GetMethods(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
            .Where(static method => string.Equals(method.Name, "forward", StringComparison.Ordinal))
            .FirstOrDefault(method => method.GetParameters().Length == arguments.Length)
            ?? throw new NotSupportedException(
                $"Module type '{torchModule.GetType().FullName}' does not expose a supported forward method with {arguments.Length} tensor argument(s)."
            );

        return forward.Invoke(torchModule, arguments);
    }

    private static object?[] GetForwardArguments<TInput>(
        TInput input,
        OnnxModel onnxModel
    )
    {
        if (input is global::TorchSharp.torch.Tensor tensor)
        {
            return [tensor];
        }

        if (input is IReadOnlyList<global::TorchSharp.torch.Tensor> tensorList)
        {
            return tensorList.Cast<object?>().ToArray();
        }

        if (input is IReadOnlyDictionary<string, global::TorchSharp.torch.Tensor> tensorDictionary)
        {
            if (tensorDictionary.Count == 1 && onnxModel.Graph.Inputs.Count == 1)
            {
                return [tensorDictionary.Values.Single()];
            }

            return onnxModel.Graph.Inputs
                .Select(graphInput => tensorDictionary.TryGetValue(graphInput.Name, out var tensorInput)
                    ? tensorInput
                    : throw new NotSupportedException(
                        $"Input sample dictionary does not contain tensor '{graphInput.Name}'."
                    ))
                .Cast<object?>()
                .ToArray();
        }

        throw new NotSupportedException(
            $"Input sample type '{typeof(TInput).FullName}' is not supported. Use Tensor, IReadOnlyList<Tensor>, or IReadOnlyDictionary<string, Tensor>."
        );
    }

    private static IReadOnlyList<NamedOnnxValue> CreateRuntimeInputs<TInput>(
        TInput input,
        OnnxModel onnxModel,
        TorchExportEvaluationOptions options
    )
    {
        if (input is global::TorchSharp.torch.Tensor tensor)
        {
            if (onnxModel.Graph.Inputs.Count != 1)
            {
                throw new NotSupportedException(
                    $"Single tensor samples require exactly one ONNX graph input, but the model has {onnxModel.Graph.Inputs.Count}."
                );
            }

            return [CreateNamedOnnxValue(onnxModel.Graph.Inputs[0].Name, tensor)];
        }

        if (input is IReadOnlyList<global::TorchSharp.torch.Tensor> tensorList)
        {
            if (onnxModel.Graph.Inputs.Count != tensorList.Count)
            {
                throw new NotSupportedException(
                    $"Input sample contains {tensorList.Count} tensor(s), but the model has {onnxModel.Graph.Inputs.Count} graph input(s)."
                );
            }

            return tensorList
                .Select((tensorInput, index) => CreateNamedOnnxValue(onnxModel.Graph.Inputs[index].Name, tensorInput))
                .ToArray();
        }

        if (input is IReadOnlyDictionary<string, global::TorchSharp.torch.Tensor> tensorDictionary)
        {
            return tensorDictionary
                .Select(entry =>
                {
                    var name = options.InputNameMapping is not null
                        && options.InputNameMapping.TryGetValue(entry.Key, out var mappedName)
                            ? mappedName
                            : entry.Key;
                    return CreateNamedOnnxValue(name, entry.Value);
                })
                .ToArray();
        }

        throw new NotSupportedException(
            $"Input sample type '{typeof(TInput).FullName}' is not supported. Use Tensor, IReadOnlyList<Tensor>, or IReadOnlyDictionary<string, Tensor>."
        );
    }

    private static NamedOnnxValue CreateNamedOnnxValue(
        string name,
        global::TorchSharp.torch.Tensor tensor
    )
    {
        return tensor.dtype switch
        {
            global::TorchSharp.torch.ScalarType.Float32 => NamedOnnxValue.CreateFromTensor(
                name,
                new DenseTensor<float>(GetTorchData<float>(tensor), GetShape(tensor))
            ),
            global::TorchSharp.torch.ScalarType.Float64 => NamedOnnxValue.CreateFromTensor(
                name,
                new DenseTensor<double>(GetTorchData<double>(tensor), GetShape(tensor))
            ),
            global::TorchSharp.torch.ScalarType.Int64 => NamedOnnxValue.CreateFromTensor(
                name,
                new DenseTensor<long>(GetTorchData<long>(tensor), GetShape(tensor))
            ),
            global::TorchSharp.torch.ScalarType.Int32 => NamedOnnxValue.CreateFromTensor(
                name,
                new DenseTensor<int>(GetTorchData<int>(tensor), GetShape(tensor))
            ),
            global::TorchSharp.torch.ScalarType.Int16 => NamedOnnxValue.CreateFromTensor(
                name,
                new DenseTensor<short>(GetTorchData<short>(tensor), GetShape(tensor))
            ),
            global::TorchSharp.torch.ScalarType.Int8 => NamedOnnxValue.CreateFromTensor(
                name,
                new DenseTensor<sbyte>(GetTorchData<sbyte>(tensor), GetShape(tensor))
            ),
            global::TorchSharp.torch.ScalarType.Byte => NamedOnnxValue.CreateFromTensor(
                name,
                new DenseTensor<byte>(GetTorchData<byte>(tensor), GetShape(tensor))
            ),
            global::TorchSharp.torch.ScalarType.Bool => NamedOnnxValue.CreateFromTensor(
                name,
                new DenseTensor<bool>(GetTorchData<bool>(tensor), GetShape(tensor))
            ),
            _ => throw new NotSupportedException($"Torch input dtype '{tensor.dtype}' is not supported by export evaluation."),
        };
    }

    private static IReadOnlyList<EvaluationTensorData> GetTorchOutputs(
        object? torchOutput,
        OnnxModel onnxModel,
        TorchExportEvaluationOptions options
    )
    {
        if (torchOutput is global::TorchSharp.torch.Tensor tensor)
        {
            if (onnxModel.Graph.Outputs.Count != 1)
            {
                throw new NotSupportedException(
                    $"Single tensor outputs require exactly one ONNX graph output, but the model has {onnxModel.Graph.Outputs.Count}."
                );
            }

            return [CreateComparableTensor(onnxModel.Graph.Outputs[0].Name, tensor)];
        }

        if (torchOutput is IReadOnlyDictionary<string, global::TorchSharp.torch.Tensor> dictionary)
        {
            return dictionary
                .Select(entry =>
                {
                    var name = options.OutputNameMapping is not null
                        && options.OutputNameMapping.TryGetValue(entry.Key, out var mappedName)
                            ? mappedName
                            : entry.Key;
                    return CreateComparableTensor(name, entry.Value);
                })
                .ToArray();
        }

        if (torchOutput is ITuple tuple)
        {
            return Enumerable.Range(0, tuple.Length)
                .Select(index => tuple[index] is global::TorchSharp.torch.Tensor tensorOutput
                    ? CreateComparableTensor(onnxModel.Graph.Outputs[index].Name, tensorOutput)
                    : throw new NotSupportedException(
                        $"Tuple output item {index} is '{tuple[index]?.GetType().FullName ?? "<null>"}', not a Torch tensor."
                    ))
                .ToArray();
        }

        if (torchOutput is object?[] array)
        {
            return array
                .Select((item, index) => item is global::TorchSharp.torch.Tensor tensorOutput
                    ? CreateComparableTensor(onnxModel.Graph.Outputs[index].Name, tensorOutput)
                    : throw new NotSupportedException(
                        $"Array output item {index} is '{item?.GetType().FullName ?? "<null>"}', not a Torch tensor."
                    ))
                .ToArray();
        }

        throw new NotSupportedException(
            $"Torch forward output type '{torchOutput?.GetType().FullName ?? "<null>"}' is not supported by export evaluation."
        );
    }

    private static EvaluationOutputResult CompareDefault(TorchExportEvaluationComparisonContext context)
    {
        var expected = context.Expected;
        var actual = context.Actual;

        if (expected.ElementType != actual.ElementType)
        {
            return EvaluationOutputResult.Failed(
                context.OutputName,
                $"Expected dtype '{expected.ElementType.Name}' but ONNX Runtime returned '{actual.ElementType.Name}'."
            );
        }

        if (!expected.Shape.SequenceEqual(actual.Shape))
        {
            return EvaluationOutputResult.Failed(
                context.OutputName,
                $"Expected shape [{string.Join(", ", expected.Shape)}] but ONNX Runtime returned [{string.Join(", ", actual.Shape)}]."
            );
        }

        if (expected.Values.Count != actual.Values.Count)
        {
            return EvaluationOutputResult.Failed(
                context.OutputName,
                $"Expected {expected.Values.Count} value(s) but ONNX Runtime returned {actual.Values.Count}."
            );
        }

        var squaredError = 0d;
        var absoluteError = 0d;
        var maxAbsoluteError = 0d;
        for (var index = 0; index < expected.Values.Count; index++)
        {
            var expectedValue = expected.Values[index];
            var actualValue = actual.Values[index];
            var difference = Math.Abs(expectedValue - actualValue);
            var tolerance = context.Options.AbsoluteTolerance
                + (context.Options.RelativeTolerance * Math.Abs(expectedValue));

            if (double.IsNaN(expectedValue) || double.IsNaN(actualValue))
            {
                if (!context.Options.AllowNaN || double.IsNaN(expectedValue) != double.IsNaN(actualValue))
                {
                    return EvaluationOutputResult.Failed(
                        context.OutputName,
                        $"NaN mismatch at flat index {index}."
                    );
                }
            }
            else if (double.IsInfinity(expectedValue) || double.IsInfinity(actualValue))
            {
                if (!context.Options.AllowInfinity || !expectedValue.Equals(actualValue))
                {
                    return EvaluationOutputResult.Failed(
                        context.OutputName,
                        $"Infinity mismatch at flat index {index}."
                    );
                }
            }
            else if (difference > tolerance)
            {
                return EvaluationOutputResult.Failed(
                    context.OutputName,
                    $"Absolute difference {difference} at flat index {index} exceeded tolerance {tolerance}."
                );
            }

            squaredError += difference * difference;
            absoluteError += difference;
            maxAbsoluteError = Math.Max(maxAbsoluteError, difference);
        }

        var elementCount = expected.Values.Count;
        return new EvaluationOutputResult(
            outputName: context.OutputName,
            loss: elementCount == 0 ? 0d : squaredError / elementCount,
            meanAbsoluteError: elementCount == 0 ? 0d : absoluteError / elementCount,
            maxAbsoluteError: maxAbsoluteError,
            elementCount: elementCount,
            expectedShape: expected.Shape,
            actualShape: actual.Shape,
            passed: true,
            message: null
        );
    }

    private static EvaluationTensorData CreateComparableTensor(
        string name,
        global::TorchSharp.torch.Tensor tensor
    )
    {
        return tensor.dtype switch
        {
            global::TorchSharp.torch.ScalarType.Float32 => new EvaluationTensorData(name, typeof(float), GetLongShape(tensor), GetTorchData<float>(tensor).Select(static value => (double)value).ToArray()),
            global::TorchSharp.torch.ScalarType.Float64 => new EvaluationTensorData(name, typeof(double), GetLongShape(tensor), GetTorchData<double>(tensor)),
            global::TorchSharp.torch.ScalarType.Int64 => new EvaluationTensorData(name, typeof(long), GetLongShape(tensor), GetTorchData<long>(tensor).Select(static value => (double)value).ToArray()),
            global::TorchSharp.torch.ScalarType.Int32 => new EvaluationTensorData(name, typeof(int), GetLongShape(tensor), GetTorchData<int>(tensor).Select(static value => (double)value).ToArray()),
            global::TorchSharp.torch.ScalarType.Int16 => new EvaluationTensorData(name, typeof(short), GetLongShape(tensor), GetTorchData<short>(tensor).Select(static value => (double)value).ToArray()),
            global::TorchSharp.torch.ScalarType.Int8 => new EvaluationTensorData(name, typeof(sbyte), GetLongShape(tensor), GetTorchData<sbyte>(tensor).Select(static value => (double)value).ToArray()),
            global::TorchSharp.torch.ScalarType.Byte => new EvaluationTensorData(name, typeof(byte), GetLongShape(tensor), GetTorchData<byte>(tensor).Select(static value => (double)value).ToArray()),
            _ => throw new NotSupportedException($"Torch output dtype '{tensor.dtype}' is not supported by default export evaluation."),
        };
    }

    private static EvaluationTensorData CreateComparableTensor(
        string name,
        object value
    )
    {
        return value switch
        {
            Tensor<float> tensor => new EvaluationTensorData(name, typeof(float), GetLongShape(tensor), tensor.ToArray().Select(static x => (double)x).ToArray()),
            Tensor<double> tensor => new EvaluationTensorData(name, typeof(double), GetLongShape(tensor), tensor.ToArray()),
            Tensor<long> tensor => new EvaluationTensorData(name, typeof(long), GetLongShape(tensor), tensor.ToArray().Select(static x => (double)x).ToArray()),
            Tensor<int> tensor => new EvaluationTensorData(name, typeof(int), GetLongShape(tensor), tensor.ToArray().Select(static x => (double)x).ToArray()),
            Tensor<short> tensor => new EvaluationTensorData(name, typeof(short), GetLongShape(tensor), tensor.ToArray().Select(static x => (double)x).ToArray()),
            Tensor<sbyte> tensor => new EvaluationTensorData(name, typeof(sbyte), GetLongShape(tensor), tensor.ToArray().Select(static x => (double)x).ToArray()),
            Tensor<byte> tensor => new EvaluationTensorData(name, typeof(byte), GetLongShape(tensor), tensor.ToArray().Select(static x => (double)x).ToArray()),
            _ => throw new NotSupportedException($"ONNX Runtime output type '{value.GetType().FullName}' is not supported by default export evaluation."),
        };
    }

    private static T[] GetTorchData<T>(global::TorchSharp.torch.Tensor tensor)
        where T : unmanaged
    {
        using var detached = tensor.detach();
        using var cpu = detached.cpu();
        using var contiguous = cpu.contiguous();
        return contiguous.data<T>().ToArray();
    }

    private static int[] GetShape(global::TorchSharp.torch.Tensor tensor)
    {
        return tensor.shape.Select(static dimension => checked((int)dimension)).ToArray();
    }

    private static long[] GetLongShape(global::TorchSharp.torch.Tensor tensor)
    {
        return tensor.shape.ToArray();
    }

    private static long[] GetLongShape<T>(Tensor<T> tensor)
        where T : unmanaged
    {
        var dimensions = tensor.Dimensions.ToArray();
        var result = new long[dimensions.Length];
        for (var index = 0; index < dimensions.Length; index++)
        {
            result[index] = dimensions[index];
        }

        return result;
    }

    private static EvaluationDiagnostic AddDiagnostic(
        List<EvaluationDiagnostic> diagnostics,
        List<EvaluationDiagnostic> sampleDiagnostics,
        TorchExportEvaluationOptions options,
        EvaluationDiagnosticSeverity severity,
        string code,
        string message,
        int? sampleIndex,
        string? outputName
    )
    {
        var diagnostic = new EvaluationDiagnostic(
            severity: severity,
            code: code,
            message: message,
            sampleIndex: sampleIndex,
            outputName: outputName
        );
        diagnostics.Add(diagnostic);
        sampleDiagnostics.Add(diagnostic);
        ThrowIfRequested(options, diagnostic);
        return diagnostic;
    }

    private static void ThrowIfRequested(
        TorchExportEvaluationOptions options,
        EvaluationDiagnostic diagnostic
    )
    {
        if (options.ThrowOnFirstFailure && diagnostic.Severity == EvaluationDiagnosticSeverity.Error)
        {
            throw new InvalidOperationException(diagnostic.Message);
        }
    }

    private static string CreateTemporaryModelPath(TorchExportEvaluationOptions options)
    {
        return options.TemporaryModelPathFactory?.Invoke()
            ?? Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.onnx");
    }

    private static void DeleteIfExists(string path)
    {
        if (File.Exists(path))
        {
            File.Delete(path);
        }
    }
}

public sealed class TorchExportEvaluationOptions
{
    public SessionOptions? SessionOptions { get; set; }

    public IReadOnlyDictionary<string, string>? InputNameMapping { get; set; }

    public IReadOnlyDictionary<string, string>? OutputNameMapping { get; set; }

    public double AbsoluteTolerance { get; set; } = 1e-5d;

    public double RelativeTolerance { get; set; } = 1e-5d;

    public bool AllowNaN { get; set; }

    public bool AllowInfinity { get; set; }

    public bool ThrowOnFirstFailure { get; set; }

    public Func<TorchExportEvaluationComparisonContext, EvaluationOutputResult>? Comparer { get; set; }

    public Func<string>? TemporaryModelPathFactory { get; set; }
}

public sealed class EvaluationResult<TTorchModule, TOnnxModel>
{
    public EvaluationResult(
        TTorchModule torchModule,
        TOnnxModel onnxModel,
        double loss,
        IReadOnlyList<EvaluationSampleResult> samples,
        IReadOnlyList<EvaluationDiagnostic> diagnostics,
        bool passed
    )
    {
        TorchModule = torchModule;
        OnnxModel = onnxModel;
        Loss = loss;
        Samples = samples;
        Diagnostics = diagnostics;
        Passed = passed;
    }

    public TTorchModule TorchModule { get; }

    public TOnnxModel OnnxModel { get; }

    public double Loss { get; }

    public IReadOnlyList<EvaluationSampleResult> Samples { get; }

    public IReadOnlyList<EvaluationDiagnostic> Diagnostics { get; }

    public bool Passed { get; }
}

public sealed class EvaluationSampleResult
{
    public EvaluationSampleResult(
        int sampleIndex,
        double loss,
        IReadOnlyList<EvaluationOutputResult> outputs,
        IReadOnlyList<EvaluationDiagnostic> diagnostics,
        bool passed
    )
    {
        SampleIndex = sampleIndex;
        Loss = loss;
        Outputs = outputs;
        Diagnostics = diagnostics;
        Passed = passed;
    }

    public int SampleIndex { get; }

    public double Loss { get; }

    public IReadOnlyList<EvaluationOutputResult> Outputs { get; }

    public IReadOnlyList<EvaluationDiagnostic> Diagnostics { get; }

    public bool Passed { get; }
}

public sealed class EvaluationOutputResult
{
    public EvaluationOutputResult(
        string outputName,
        double loss,
        double meanAbsoluteError,
        double maxAbsoluteError,
        long elementCount,
        IReadOnlyList<long> expectedShape,
        IReadOnlyList<long> actualShape,
        bool passed,
        string? message
    )
    {
        OutputName = outputName;
        Loss = loss;
        MeanAbsoluteError = meanAbsoluteError;
        MaxAbsoluteError = maxAbsoluteError;
        ElementCount = elementCount;
        ExpectedShape = expectedShape;
        ActualShape = actualShape;
        Passed = passed;
        Message = message;
    }

    public string OutputName { get; }

    public double Loss { get; }

    public double MeanAbsoluteError { get; }

    public double MaxAbsoluteError { get; }

    public long ElementCount { get; }

    public IReadOnlyList<long> ExpectedShape { get; }

    public IReadOnlyList<long> ActualShape { get; }

    public bool Passed { get; }

    public string? Message { get; }

    public static EvaluationOutputResult Failed(
        string outputName,
        string message
    )
    {
        return new EvaluationOutputResult(
            outputName: outputName,
            loss: double.NaN,
            meanAbsoluteError: double.NaN,
            maxAbsoluteError: double.NaN,
            elementCount: 0,
            expectedShape: [],
            actualShape: [],
            passed: false,
            message: message
        );
    }
}

public sealed class EvaluationDiagnostic
{
    public EvaluationDiagnostic(
        EvaluationDiagnosticSeverity severity,
        string code,
        string message,
        int? sampleIndex,
        string? outputName
    )
    {
        Severity = severity;
        Code = code;
        Message = message;
        SampleIndex = sampleIndex;
        OutputName = outputName;
    }

    public EvaluationDiagnosticSeverity Severity { get; }

    public string Code { get; }

    public string Message { get; }

    public int? SampleIndex { get; }

    public string? OutputName { get; }
}

public enum EvaluationDiagnosticSeverity
{
    Info,
    Warning,
    Error,
}

public sealed class TorchExportEvaluationComparisonContext
{
    public TorchExportEvaluationComparisonContext(
        int sampleIndex,
        string outputName,
        EvaluationTensorData expected,
        EvaluationTensorData actual,
        TorchExportEvaluationOptions options
    )
    {
        SampleIndex = sampleIndex;
        OutputName = outputName;
        Expected = expected;
        Actual = actual;
        Options = options;
    }

    public int SampleIndex { get; }

    public string OutputName { get; }

    public EvaluationTensorData Expected { get; }

    public EvaluationTensorData Actual { get; }

    public TorchExportEvaluationOptions Options { get; }
}

public sealed class EvaluationTensorData
{
    public EvaluationTensorData(
        string name,
        Type elementType,
        IReadOnlyList<long> shape,
        IReadOnlyList<double> values
    )
    {
        Name = name;
        ElementType = elementType;
        Shape = shape;
        Values = values;
    }

    public string Name { get; }

    public Type ElementType { get; }

    public IReadOnlyList<long> Shape { get; }

    public IReadOnlyList<double> Values { get; }
}
