using System.Globalization;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using Onnxify.Data;
using Onnxify.Data.Numerics;

namespace Onnxify.ProjectGenerator;

public sealed class OnnxProjectGenerator
{
    public ProjectGenerationResult Generate(ProjectGeneratorOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);

        if (string.IsNullOrWhiteSpace(options.InputModelPath))
        {
            throw new ArgumentException("Input model path is required.", nameof(options));
        }

        if (string.IsNullOrWhiteSpace(options.OutputDirectoryPath))
        {
            throw new ArgumentException("Output directory path is required.", nameof(options));
        }

        var outputDirectoryPath = Path.GetFullPath(options.OutputDirectoryPath);
        Directory.CreateDirectory(outputDirectoryPath);
        Directory.CreateDirectory(Path.Combine(outputDirectoryPath, options.TensorDirectoryName));

        var model = OnnxModel.FromFile(options.InputModelPath);
        var context = new GenerationContext(options, outputDirectoryPath);

        var programSource = RenderProgram(model, context);
        var programFilePath = Path.Combine(outputDirectoryPath, options.ProgramFileName);
        WriteFile(programFilePath, programSource, options.Overwrite);

        string? projectFilePath = null;
        if (options.GenerateProjectFile)
        {
            projectFilePath = Path.Combine(outputDirectoryPath, options.GetProjectFileName());
            WriteFile(projectFilePath, RenderProjectFile(options), options.Overwrite);
        }

        return new ProjectGenerationResult
        {
            OutputDirectoryPath = outputDirectoryPath,
            ProgramFilePath = programFilePath,
            ProjectFilePath = projectFilePath,
            TensorFilePaths = context.TensorFilePaths.ToArray(),
            Warnings = context.Warnings.ToArray(),
        };
    }

    private static void WriteFile(string path, string content, bool overwrite)
    {
        if (File.Exists(path) && !overwrite)
        {
            throw new IOException($"File already exists at '{path}'.");
        }

        File.WriteAllText(path, content, Encoding.UTF8);
    }

    private static string RenderProjectFile(ProjectGeneratorOptions options)
    {
        return $$"""
        <Project Sdk="Microsoft.NET.Sdk">
          <PropertyGroup>
            <OutputType>Exe</OutputType>
            <TargetFramework>net10.0</TargetFramework>
            <ImplicitUsings>enable</ImplicitUsings>
            <Nullable>enable</Nullable>
            <AssemblyName>{{options.GetProjectName()}}</AssemblyName>
            <RootNamespace>{{options.GetNamespace()}}</RootNamespace>
          </PropertyGroup>

          <ItemGroup>
            <PackageReference Include="Onnxify" Version="{{options.GetOnnxifyPackageVersion()}}" />
          </ItemGroup>

          <ItemGroup>
            <None Include="{{options.TensorDirectoryName}}\**\*">
              <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
            </None>
          </ItemGroup>
        </Project>
        """;
    }

    private static string RenderProgram(OnnxModel model, GenerationContext context)
    {
        if (model.TrainingInfo.Count > 0)
        {
            throw new NotSupportedException("TrainingInfo generation is not supported yet.");
        }

        var options = context.Options;
        var namespaceName = options.GetNamespace();
        var className = options.ProgramClassName.SanitizeIdentifier("Program");
        var factoryMethodName = options.FactoryMethodName.SanitizeIdentifier("CreateModel");
        var outputFileName = $"{Path.GetFileNameWithoutExtension(options.InputModelPath)}.generated.onnx";
        var modelCreationOptions = RenderModelCreationOptions(model, context);
        var flow = new StringBuilder();
        var graphVarNames = new Dictionary<IOnnxGraphEdge, string>(ReferenceEqualityComparer<IOnnxGraphEdge>.Instance);

        string GetEdgeVariable(IOnnxGraphEdge edge)
        {
            if (graphVarNames.TryGetValue(edge, out var existing))
            {
                return existing;
            }

            var baseName = string.IsNullOrWhiteSpace(edge.Name)
                ? $"{edge.GetType().Name}_{graphVarNames.Count}"
                : edge.Name;

            var candidate = baseName.SanitizeLocalIdentifier($"{edge.GetType().Name}_{graphVarNames.Count}");
            while (graphVarNames.Values.Contains(candidate, StringComparer.Ordinal))
            {
                candidate += "_";
            }

            graphVarNames.Add(edge, candidate);
            return candidate;
        }

        AppendModelMetadata(model, flow, context);

        foreach (var input in model.Graph.Inputs)
        {
            flow.AppendLine(
                $$"""
                var {{GetEdgeVariable(input)}} = model.Graph.AddInput(
                    name: {{AsStringLiteral(input.Name)}},
                    type: {{RenderValueType(input.Type)}}
                );

                """);
        }

        foreach (var output in model.Graph.Outputs)
        {
            flow.AppendLine(
                $$"""
                var {{GetEdgeVariable(output)}} = model.Graph.AddOutput(
                    name: {{AsStringLiteral(output.Name)}},
                    type: {{RenderValueType(output.Type)}}
                );

                """);
        }

        foreach (var value in model.Graph.Placeholders)
        {
            flow.AppendLine(
                $$"""
                var {{GetEdgeVariable(value)}} = model.Graph.AddValue(
                    name: {{AsStringLiteral(value.Name)}},
                    type: {{RenderValueType(value.Type)}}
                );

                """
            );
        }

        foreach (var tensor in model.Graph.Initializers)
        {
            var assetRelativePath = ExportTensorAsset(tensor, context);

            flow.AppendLine(
                $$"""
                var {{GetEdgeVariable(tensor)}} = model.Graph.AddTensor(
                    name: {{AsStringLiteral(tensor.Name)}},
                    shape: {{RenderLongArray(tensor.Shape)}},
                    value: {{RenderTensorLoadExpression(tensor, assetRelativePath)}}
                );

                """
            );
        }

        var intermediateEdges = model.Graph.Nodes
            .SelectMany(node => node.Inputs.Concat(node.Outputs))
            .OfType<OnnxEdge>()
            .Distinct(ReferenceEqualityComparer<OnnxEdge>.Instance)
            .ToArray();

        foreach (var edge in intermediateEdges)
        {
            flow.AppendLine(
                $$"""
                var {{GetEdgeVariable(edge)}} = model.Graph.AddEdge({{AsStringLiteral(edge.Name)}});

                """
            );
        }

        foreach (var node in model.Graph.Nodes)
        {
            if (TryRenderTypedNodeInvocation(node, GetEdgeVariable, context, out var typedInvocation))
            {
                flow.AppendLine(typedInvocation);
                continue;
            }

            flow.AppendLine(
                $$"""
                model.Graph.AddNode(
                    name: {{AsStringLiteral(node.Name)}},
                    opType: {{AsStringLiteral(node.OpType)}},
                    domain: {{AsStringLiteral(node.Domain)}},
                    docString: {{AsStringLiteral(node.DocString)}},
                    inputs: {{RenderEdgeList(node.Inputs.Select(GetEdgeVariable))}},
                    outputs: {{RenderEdgeList(node.Outputs.Select(GetEdgeVariable))}},
                    attributes: {{RenderAttributes(node.Attributes, context)}}
                );

                """
            );
        }

        if (!string.IsNullOrWhiteSpace(model.Graph.Name))
        {
            context.Warnings.Add($"Graph.Name '{model.Graph.Name}' is not emitted because Onnxify does not expose a public setter.");
        }

        return $$"""
        using System;
        using System.IO;
        using Onnxify;
        using Onnxify.Data;
        using Onnxify.Data.Numerics;

        namespace {{namespaceName}};

        public static class {{className}}
        {
            public static void Main(string[] args)
            {
                var model = {{factoryMethodName}}();
                var outputPath = args.Length > 0
                    ? args[0]
                    : Path.Combine(AppContext.BaseDirectory, {{AsStringLiteral(outputFileName)}});

                model.Save(outputPath, overwrite: true);
                Console.WriteLine($"Model saved to: {outputPath}");
            }

            public static OnnxModel {{factoryMethodName}}()
            {
                var model = OnnxModel.Create({{modelCreationOptions.Indent(2)}});

                {{flow.ToString().Indent(2)}}

                return model;
            }

            private static OnnxTensor CreateDetachedTensor<T>(string name, long[] shape, T[] value)
            {
                var tempModel = OnnxModel.Create();
                return tempModel.Graph.AddTensor(name, shape, value);
            }

            private static string ResolveAssetPath(string relativePath)
            {
                return Path.Combine(AppContext.BaseDirectory, relativePath);
            }
        }
        """;
    }

    private static void AppendModelMetadata(OnnxModel model, StringBuilder flow, GenerationContext context)
    {
        flow.AppendLine(
            $$"""
            model.ProducerVersion = {{AsStringLiteral(model.ProducerVersion)}};
            model.ModelVersion = {{model.ModelVersion.ToString(CultureInfo.InvariantCulture)}}L;
            model.Domain = {{AsStringLiteral(model.Domain)}};
            model.Document = {{AsStringLiteral(model.Document)}};

            """);

        foreach (var metadataProp in model.MetadataProps)
        {
            flow.AppendLine(
                $$"""
                model.AddMetadataProps({{AsStringLiteral(metadataProp.Key)}}, {{AsStringLiteral(metadataProp.Value)}});

                """);
        }
    }

    private static string RenderModelCreationOptions(OnnxModel model, GenerationContext context)
    {
        var defaultOpset = new OnnxModelCreationOptions().Opset;
        var defaultOpsetImport = model.OpsetImport.FirstOrDefault(static x => string.IsNullOrEmpty(x.Domain));

        if (defaultOpsetImport is not null)
        {
            defaultOpset = checked((int)defaultOpsetImport.Version);
        }
        else if (model.OpsetImport.Count > 0)
        {
            context.Warnings.Add("Model uses only non-default opset imports. Generated code falls back to Onnxify's default ai.onnx opset.");
        }

        foreach (var opsetImport in model.OpsetImport.Where(static x => !string.IsNullOrEmpty(x.Domain)))
        {
            context.Warnings.Add($"Opset import '{opsetImport.Domain}' v{opsetImport.Version} is not emitted because Onnxify does not expose a public mutator for custom opset imports.");
        }

        return $$"""
        new OnnxModelCreationOptions
        {
            ProducerName = {{AsStringLiteral(model.ProducerName)}},
            IrVersion = {{model.IrVersion.ToString(CultureInfo.InvariantCulture)}}L,
            Opset = {{defaultOpset.ToString(CultureInfo.InvariantCulture)}},
        }
        """;
    }

    private static string ExportTensorAsset(OnnxTensor tensor, GenerationContext context)
    {
        var safeName = (string.IsNullOrWhiteSpace(tensor.Name) ? "tensor" : tensor.Name).SanitizeFileName();
        return tensor.DataType == typeof(string)
            ? context.WriteTextAsset(safeName, "json", OnnxExternalDataProvider.Instance.EncodeStringTensorJson(tensor))
            : context.WriteBinaryAsset(safeName, "bin", OnnxExternalDataProvider.Instance.EncodeTensorRawData(tensor));
    }

    private static string RenderTensorLoadExpression(OnnxTensor tensor, string relativePath)
    {
        var literalPath = AsStringLiteral(relativePath.Replace('\\', '/'));
        var resolvedPath = $"ResolveAssetPath({literalPath})";

        return tensor switch
        {
            OnnxTensor<float> => RenderTensorReadExpression<float>(resolvedPath),
            OnnxTensor<double> => RenderTensorReadExpression<double>(resolvedPath),
            OnnxTensor<sbyte> => RenderTensorReadExpression<sbyte>(resolvedPath),
            OnnxTensor<byte> => RenderTensorReadExpression<byte>(resolvedPath),
            OnnxTensor<short> => RenderTensorReadExpression<short>(resolvedPath),
            OnnxTensor<ushort> => RenderTensorReadExpression<ushort>(resolvedPath),
            OnnxTensor<int> => RenderTensorReadExpression<int>(resolvedPath),
            OnnxTensor<uint> => RenderTensorReadExpression<uint>(resolvedPath),
            OnnxTensor<long> => RenderTensorReadExpression<long>(resolvedPath),
            OnnxTensor<ulong> => RenderTensorReadExpression<ulong>(resolvedPath),
            OnnxTensor<bool> => RenderTensorReadExpression<bool>(resolvedPath),
            OnnxTensor<Half> => RenderTensorReadExpression<Half>(resolvedPath),
            OnnxTensor<BFloat16> => RenderTensorReadExpression<BFloat16>(resolvedPath),
            OnnxTensor<Float8E4M3FN> => RenderTensorReadExpression<Float8E4M3FN>(resolvedPath),
            OnnxTensor<Float8E4M3FNUZ> => RenderTensorReadExpression<Float8E4M3FNUZ>(resolvedPath),
            OnnxTensor<Float8E5M2> => RenderTensorReadExpression<Float8E5M2>(resolvedPath),
            OnnxTensor<Float8E5M2FNUZ> => RenderTensorReadExpression<Float8E5M2FNUZ>(resolvedPath),
            OnnxTensor<Float4E2M1> => RenderTensorReadExpression<Float4E2M1>(resolvedPath, GetTensorElementCount(tensor)),
            OnnxTensor<Float8E8M0> => RenderTensorReadExpression<Float8E8M0>(resolvedPath),
            OnnxTensor<UInt4> => RenderTensorReadExpression<UInt4>(resolvedPath, GetTensorElementCount(tensor)),
            OnnxTensor<Int4> => RenderTensorReadExpression<Int4>(resolvedPath, GetTensorElementCount(tensor)),
            OnnxTensor<UInt2> => RenderTensorReadExpression<UInt2>(resolvedPath, GetTensorElementCount(tensor)),
            OnnxTensor<Int2> => RenderTensorReadExpression<Int2>(resolvedPath, GetTensorElementCount(tensor)),
            OnnxTensor<Complex64> => RenderTensorReadExpression<Complex64>(resolvedPath),
            OnnxTensor<Complex128> => RenderTensorReadExpression<Complex128>(resolvedPath),
            OnnxTensor<string> => $"global::System.Text.Json.JsonSerializer.Deserialize<string[]>(File.ReadAllText({resolvedPath})) ?? throw new InvalidOperationException(\"Could not deserialize tensor data.\")",
            _ => throw CreateUnsupportedTensorException(tensor),
        };
    }

    private static string RenderTensorReadExpression<T>(string resolvedPath, long? elementCount = null)
    {
        var typeName = GetClrTypeName(typeof(T));
        var readExpression = $"OnnxExternalDataProvider.Instance.ReadTensorValue<{typeName}>({resolvedPath}, offset: 0, length: -1)";

        if (elementCount is null)
        {
            return readExpression;
        }

        return $"({readExpression})[..checked((int){elementCount.Value.ToString(CultureInfo.InvariantCulture)}L)]";
    }

    private static long GetTensorElementCount(OnnxTensor tensor)
    {
        return tensor.Shape.Aggregate(1L, static (count, dimension) => checked(count * dimension));
    }

    private static bool TryRenderTypedNodeInvocation(
        OnnxNode node,
        Func<IOnnxGraphEdge, string> getEdgeVariable,
        GenerationContext context,
        out string invocation)
    {
        invocation = string.Empty;

        var nodeType = node.GetType();
        if (nodeType == typeof(OnnxNode) || nodeType.Namespace is null)
        {
            return false;
        }

        var optionsType = nodeType.Assembly.GetType($"{nodeType.Namespace}.{nodeType.Name}InputOutputOptions", throwOnError: false);
        if (optionsType is null)
        {
            return false;
        }

        var assignments = new List<string>();

        var props = optionsType
            .GetProperties(BindingFlags.Instance | BindingFlags.Public)
            .OrderBy(static x => x.MetadataToken)
            .ToArray();

        foreach (var optionsProperty in props)
        {
            var nodeProperty = nodeType.GetProperty(optionsProperty.Name, BindingFlags.Instance | BindingFlags.Public);
            if (nodeProperty is null)
            {
                return false;
            }

            var value = nodeProperty.GetValue(node);
            if (value is null)
            {
                continue;
            }

            if (!TryRenderTypedOptionValue(value, optionsProperty.PropertyType, getEdgeVariable, context, out var renderedValue))
            {
                return false;
            }

            assignments.Add($"{optionsProperty.Name} = {renderedValue}");
        }

        if (!string.IsNullOrWhiteSpace(node.DocString))
        {
            context.Warnings.Add($"Node '{node.Name}' docString is not emitted because typed Onnxify wrappers do not expose it.");
        }

        var domainAccessor = GetDomainAccessor(node.Domain);
        var optionsTypeName = optionsType.FullName?.Replace("+", ".", StringComparison.Ordinal)
            ?? throw new InvalidOperationException($"Could not resolve options type name for '{nodeType.FullName}'.");

        invocation = $$"""
            {{domainAccessor}}.{{node.OpType}}(
                name: {{AsStringLiteral(node.Name)}},
                options: new global::{{optionsTypeName}}
                {
                    {{string.Join("," + Environment.NewLine, assignments).Indent(2)}}
                }
            );

            """;

        return true;
    }

    private static bool TryRenderTypedOptionValue(
        object value,
        Type declaredType,
        Func<IOnnxGraphEdge, string> getEdgeVariable,
        GenerationContext context,
        out string renderedValue)
    {
        if (declaredType == typeof(OnnxTensor) && value is OnnxTensor attributeTensor)
        {
            renderedValue = RenderAttributeTensor(attributeTensor, context);
            return true;
        }

        switch (value)
        {
            case IOnnxGraphEdge edge:
                renderedValue = getEdgeVariable(edge);
                return true;
            case IReadOnlyList<IOnnxGraphEdge> edges:
                renderedValue = RenderEdgeList(edges.Select(getEdgeVariable));
                return true;
            case string text:
                renderedValue = AsStringLiteral(text);
                return true;
            case long longValue:
                renderedValue = $"{longValue.ToString(CultureInfo.InvariantCulture)}L";
                return true;
            case float floatValue:
                renderedValue = RenderFloat(floatValue);
                return true;
            case double doubleValue:
                renderedValue = RenderDouble(doubleValue);
                return true;
            case bool boolValue:
                renderedValue = boolValue ? "true" : "false";
                return true;
            case long[] longValues:
                renderedValue = RenderLongArray(longValues);
                return true;
            case float[] floatValues:
                renderedValue = RenderFloatArray(floatValues);
                return true;
            case string[] stringValues:
                renderedValue = RenderStringArray(stringValues);
                return true;
            case OnnxValueType valueType:
                renderedValue = RenderValueType(valueType);
                return true;
            case OnnxValueType[] valueTypes:
                renderedValue = $"[{string.Join(", ", valueTypes.Select(RenderValueType))}]";
                return true;
            default:
                renderedValue = string.Empty;
                return false;
        }
    }

    private static string GetDomainAccessor(string domain)
    {
        return domain switch
        {
            "" => "model.Graph",
            "ai.onnx.ml" => "model.Graph.ML",
            "com.microsoft" => "model.Graph.Microsoft",
            "com.microsoft.nhwc" => "model.Graph.Microsoft.NHWC",
            "com.microsoft.nchwc" => "model.Graph.Microsoft.NCHWc",
            "com.ms.internal.nhwc" => "model.Graph.Microsoft.Internal.NHWC",
            _ => throw new NotSupportedException($"Domain '{domain}' is not supported by Onnxify's typed wrapper surface."),
        };
    }

    private static string RenderValueType(OnnxValueType valueType)
    {
        return valueType switch
        {
            OnnxTensorType tensorType => RenderTensorType(tensorType),
            _ => throw new NotSupportedException($"Value type '{valueType.GetType().Name}' is not supported."),
        };
    }

    private static string RenderTensorType(OnnxTensorType tensorType)
    {
        var typeName = GetClrTypeName(tensorType.Type);
        if (tensorType.Shape is null)
        {
            return $"OnnxTensorType.Create<{typeName}>({AsStringLiteral(tensorType.Denotation)})";
        }

        var shape = string.Join(", ", tensorType.Shape.Dimensions.Select(RenderDimension));
        return $"OnnxTensorType.Create<{typeName}>([{shape}], {AsStringLiteral(tensorType.Denotation)})";
    }

    private static string RenderDimension(OnnxDimension dimension)
    {
        return dimension.GetValue() switch
        {
            long value => $"{value.ToString(CultureInfo.InvariantCulture)}L",
            string value => AsStringLiteral(value),
            _ => throw new NotSupportedException($"Dimension type '{dimension.GetValue().GetType().Name}' is not supported."),
        };
    }

    private static string RenderAttributes(IEnumerable<OnnxAttribute> attributes, GenerationContext context)
    {
        var rendered = attributes.Select(attribute => RenderAttribute(attribute, context)).ToArray();
        return rendered.Length == 0 ? "[]" : $"[{string.Join(", ", rendered)}]";
    }

    private static string RenderAttribute(OnnxAttribute attribute, GenerationContext context)
    {
        return attribute switch
        {
            OnnxAttribute<float> x => $"new OnnxAttribute<float>({AsStringLiteral(x.Name)}, {RenderFloat(x.Value)})",
            OnnxAttribute<long> x => $"new OnnxAttribute<long>({AsStringLiteral(x.Name)}, {x.Value.ToString(CultureInfo.InvariantCulture)}L)",
            OnnxAttribute<string> x => $"new OnnxAttribute<string>({AsStringLiteral(x.Name)}, {AsStringLiteral(x.Value)})",
            OnnxAttribute<float[]> x => $"new OnnxAttribute<float[]>({AsStringLiteral(x.Name)}, {RenderFloatArray(x.Value)})",
            OnnxAttribute<long[]> x => $"new OnnxAttribute<long[]>({AsStringLiteral(x.Name)}, {RenderLongArray(x.Value)})",
            OnnxAttribute<string[]> x => $"new OnnxAttribute<string[]>({AsStringLiteral(x.Name)}, {RenderStringArray(x.Value)})",
            OnnxAttribute<OnnxValueType> x => $"new OnnxAttribute<OnnxValueType>({AsStringLiteral(x.Name)}, {RenderValueType(x.Value)})",
            OnnxAttribute<OnnxValueType[]> x => $"new OnnxAttribute<OnnxValueType[]>({AsStringLiteral(x.Name)}, [{string.Join(", ", x.Value.Select(RenderValueType))}])",
            OnnxAttribute<OnnxTensor> x => $"new OnnxAttribute<OnnxTensor>({AsStringLiteral(x.Name)}, {RenderAttributeTensor(x.Value, context)})",
            _ => throw new NotSupportedException($"Attribute '{attribute.Name}' of type '{attribute.GetType().Name}' is not supported."),
        };
    }

    private static string RenderAttributeTensor(OnnxTensor tensor, GenerationContext context)
    {
        var assetRelativePath = ExportTensorAsset(tensor, context);
        return $"CreateDetachedTensor({AsStringLiteral(tensor.Name)}, {RenderLongArray(tensor.Shape)}, {RenderTensorLoadExpression(tensor, assetRelativePath)})";
    }

    private static string RenderEdgeList(IEnumerable<string> variableNames)
    {
        var values = variableNames.ToArray();
        return values.Length == 0 ? "[]" : $"[{string.Join(", ", values)}]";
    }

    private static string RenderFloat(float value)
    {
        if (float.IsNaN(value))
        {
            return "float.NaN";
        }

        if (float.IsPositiveInfinity(value))
        {
            return "float.PositiveInfinity";
        }

        if (float.IsNegativeInfinity(value))
        {
            return "float.NegativeInfinity";
        }

        return value.ToString("R", CultureInfo.InvariantCulture) + "F";
    }

    private static string RenderDouble(double value)
    {
        if (double.IsNaN(value))
        {
            return "double.NaN";
        }

        if (double.IsPositiveInfinity(value))
        {
            return "double.PositiveInfinity";
        }

        if (double.IsNegativeInfinity(value))
        {
            return "double.NegativeInfinity";
        }

        return value.ToString("R", CultureInfo.InvariantCulture) + "D";
    }

    private static string RenderFloatArray(IEnumerable<float> values)
    {
        var rendered = values.Select(RenderFloat).ToArray();
        return rendered.Length == 0 ? "[]" : $"[{string.Join(", ", rendered)}]";
    }

    private static string RenderLongArray(IEnumerable<long> values)
    {
        var rendered = values.Select(value => value.ToString(CultureInfo.InvariantCulture) + "L").ToArray();
        return rendered.Length == 0 ? "[]" : $"[{string.Join(", ", rendered)}]";
    }

    private static string RenderStringArray(IEnumerable<string> values)
    {
        var rendered = values.Select(AsStringLiteral).ToArray();
        return rendered.Length == 0 ? "[]" : $"[{string.Join(", ", rendered)}]";
    }

    private static string AsStringLiteral(string? value)
    {
        value ??= string.Empty;

        var escaped = value
            .Replace("\\", "\\\\", StringComparison.Ordinal)
            .Replace("\"", "\\\"", StringComparison.Ordinal)
            .Replace("\r", "\\r", StringComparison.Ordinal)
            .Replace("\n", "\\n", StringComparison.Ordinal)
            .Replace("\t", "\\t", StringComparison.Ordinal);

        return $"\"{escaped}\"";
    }

    private static string GetClrTypeName(Type type)
    {
        if (type == typeof(float)) return "float";
        if (type == typeof(double)) return "double";
        if (type == typeof(sbyte)) return "sbyte";
        if (type == typeof(byte)) return "byte";
        if (type == typeof(short)) return "short";
        if (type == typeof(ushort)) return "ushort";
        if (type == typeof(int)) return "int";
        if (type == typeof(uint)) return "uint";
        if (type == typeof(long)) return "long";
        if (type == typeof(ulong)) return "ulong";
        if (type == typeof(bool)) return "bool";
        if (type == typeof(string)) return "string";
        if (type == typeof(Half)) return nameof(Half);
        if (type == typeof(BFloat16)) return nameof(BFloat16);
        if (type == typeof(Float8E4M3FN)) return nameof(Float8E4M3FN);
        if (type == typeof(Float8E4M3FNUZ)) return nameof(Float8E4M3FNUZ);
        if (type == typeof(Float8E5M2)) return nameof(Float8E5M2);
        if (type == typeof(Float8E5M2FNUZ)) return nameof(Float8E5M2FNUZ);
        if (type == typeof(Float4E2M1)) return nameof(Float4E2M1);
        if (type == typeof(Float8E8M0)) return nameof(Float8E8M0);
        if (type == typeof(UInt4)) return nameof(UInt4);
        if (type == typeof(Int4)) return nameof(Int4);
        if (type == typeof(UInt2)) return nameof(UInt2);
        if (type == typeof(Int2)) return nameof(Int2);
        if (type == typeof(Complex64)) return nameof(Complex64);
        if (type == typeof(Complex128)) return nameof(Complex128);

        throw new NotSupportedException($"CLR type '{type}' is not supported.");
    }

    private static NotSupportedException CreateUnsupportedTensorException(OnnxTensor tensor)
    {
        return new($"Tensor '{tensor.Name}' uses unsupported data type '{tensor.DataType}'.");
    }

    private sealed class GenerationContext
    {
        private readonly HashSet<string> _relativePaths = new(StringComparer.OrdinalIgnoreCase);

        public ProjectGeneratorOptions Options { get; }
        public string OutputDirectoryPath { get; }
        public List<string> TensorFilePaths { get; } = [];
        public List<string> Warnings { get; } = [];

        public GenerationContext(ProjectGeneratorOptions options, string outputDirectoryPath)
        {
            Options = options;
            OutputDirectoryPath = outputDirectoryPath;
        }

        public string WriteBinaryAsset(string baseName, string extension, byte[] bytes)
        {
            return WriteAsset(baseName, extension, stream => stream.Write(bytes, 0, bytes.Length));
        }

        public string WriteTextAsset(string baseName, string extension, string text)
        {
            return WriteAsset(baseName, extension, stream =>
            {
                using var writer = new StreamWriter(stream, Encoding.UTF8, 1024, leaveOpen: true);
                writer.Write(text);
                writer.Flush();
            });
        }

        private string WriteAsset(string baseName, string extension, Action<FileStream> writer)
        {
            var relativePath = GetUniqueRelativePath(baseName, extension);
            var fullPath = Path.Combine(OutputDirectoryPath, relativePath);
            Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);

            if (File.Exists(fullPath) && !Options.Overwrite)
            {
                throw new IOException($"File already exists at '{fullPath}'.");
            }

            using var stream = File.Create(fullPath);
            writer(stream);
            TensorFilePaths.Add(fullPath);
            return relativePath.Replace('\\', '/');
        }

        private string GetUniqueRelativePath(string baseName, string extension)
        {
            var counter = 0;

            while (true)
            {
                var suffix = counter == 0 ? string.Empty : "_" + counter.ToString(CultureInfo.InvariantCulture);
                var candidate = Path.Combine(Options.TensorDirectoryName, $"{baseName}{suffix}.{extension}");

                if (_relativePaths.Add(candidate))
                {
                    return candidate;
                }

                counter++;
            }
        }
    }

    private sealed class ReferenceEqualityComparer<T> : IEqualityComparer<T> where T : class
    {
        public static readonly ReferenceEqualityComparer<T> Instance = new();

        public bool Equals(T? x, T? y) => ReferenceEquals(x, y);
        public int GetHashCode(T obj) => RuntimeHelpers.GetHashCode(obj);
    }
}
