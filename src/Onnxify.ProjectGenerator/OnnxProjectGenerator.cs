using System.Globalization;
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
        var className = SanitizeIdentifier(options.ProgramClassName, "Program");
        var factoryMethodName = SanitizeIdentifier(options.FactoryMethodName, "CreateModel");
        var outputFileName = $"{Path.GetFileNameWithoutExtension(options.InputModelPath)}.generated.onnx";
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

            var candidate = SanitizeIdentifier(baseName, $"{edge.GetType().Name}_{graphVarNames.Count}");
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
        using Onnx;
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
                var model = OnnxModel.Create(new OnnxModelCreationOptions());

                {{Indent(flow.ToString(), 4)}}

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
                model.ProducerName = {{AsStringLiteral(model.ProducerName)}};
                model.ProducerVersion = {{AsStringLiteral(model.ProducerVersion)}};
                model.ModelVersion = {{model.ModelVersion.ToString(CultureInfo.InvariantCulture)}}L;
                model.IrVersion = {{model.IrVersion.ToString(CultureInfo.InvariantCulture)}}L;
                model.Domain = {{AsStringLiteral(model.Domain)}};
                model.Document = {{AsStringLiteral(model.Document)}};
                model.MetadataProps.Clear();
                model.OpsetImport.Clear();

            """);

        foreach (var metadataProp in model.MetadataProps)
        {
            flow.AppendLine(
                $$"""
                    model.MetadataProps.Add(new StringStringEntryProto
                    {
                        Key = {{AsStringLiteral(metadataProp.Key)}},
                        Value = {{AsStringLiteral(metadataProp.Value)}},
                    });

                """);
        }

        foreach (var opsetImport in model.OpsetImport)
        {
            flow.AppendLine(
                $$"""
                    model.OpsetImport.Add(new OperatorSetIdProto
                    {
                        Domain = {{AsStringLiteral(opsetImport.Domain)}},
                        Version = {{opsetImport.Version.ToString(CultureInfo.InvariantCulture)}}L,
                    });

                """);
        }
    }

    private static string ExportTensorAsset(OnnxTensor tensor, GenerationContext context)
    {
        var safeName = SanitizeFileName(string.IsNullOrWhiteSpace(tensor.Name) ? "tensor" : tensor.Name);
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
            OnnxTensor<float> => $"OnnxExternalDataProvider.Instance.ReadTensorArray<float>({resolvedPath})",
            OnnxTensor<double> => $"OnnxExternalDataProvider.Instance.ReadTensorArray<double>({resolvedPath})",
            OnnxTensor<sbyte> => $"OnnxExternalDataProvider.Instance.ReadTensorArray<sbyte>({resolvedPath})",
            OnnxTensor<byte> => $"OnnxExternalDataProvider.Instance.ReadTensorArray<byte>({resolvedPath})",
            OnnxTensor<short> => $"OnnxExternalDataProvider.Instance.ReadTensorArray<short>({resolvedPath})",
            OnnxTensor<ushort> => $"OnnxExternalDataProvider.Instance.ReadTensorArray<ushort>({resolvedPath})",
            OnnxTensor<int> => $"OnnxExternalDataProvider.Instance.ReadTensorArray<int>({resolvedPath})",
            OnnxTensor<uint> => $"OnnxExternalDataProvider.Instance.ReadTensorArray<uint>({resolvedPath})",
            OnnxTensor<long> => $"OnnxExternalDataProvider.Instance.ReadTensorArray<long>({resolvedPath})",
            OnnxTensor<ulong> => $"OnnxExternalDataProvider.Instance.ReadTensorArray<ulong>({resolvedPath})",
            OnnxTensor<bool> => $"OnnxExternalDataProvider.Instance.ReadTensorArray<bool>({resolvedPath})",
            OnnxTensor<Half> => $"OnnxExternalDataProvider.Instance.ReadTensorArray<Half>({resolvedPath})",
            OnnxTensor<BFloat16> => $"OnnxExternalDataProvider.Instance.ReadTensorArray<BFloat16>({resolvedPath})",
            OnnxTensor<Float8E4M3FN> => $"OnnxExternalDataProvider.Instance.ReadTensorArray<Float8E4M3FN>({resolvedPath})",
            OnnxTensor<Float8E4M3FNUZ> => $"OnnxExternalDataProvider.Instance.ReadTensorArray<Float8E4M3FNUZ>({resolvedPath})",
            OnnxTensor<Float8E5M2> => $"OnnxExternalDataProvider.Instance.ReadTensorArray<Float8E5M2>({resolvedPath})",
            OnnxTensor<Float8E5M2FNUZ> => $"OnnxExternalDataProvider.Instance.ReadTensorArray<Float8E5M2FNUZ>({resolvedPath})",
            OnnxTensor<Float4E2M1> => $"OnnxExternalDataProvider.Instance.ReadTensorArray<Float4E2M1>({resolvedPath}, {GetTensorElementCount(tensor).ToString(CultureInfo.InvariantCulture)}L)",
            OnnxTensor<Float8E8M0> => $"OnnxExternalDataProvider.Instance.ReadTensorArray<Float8E8M0>({resolvedPath})",
            OnnxTensor<UInt4> => $"OnnxExternalDataProvider.Instance.ReadTensorArray<UInt4>({resolvedPath}, {GetTensorElementCount(tensor).ToString(CultureInfo.InvariantCulture)}L)",
            OnnxTensor<Int4> => $"OnnxExternalDataProvider.Instance.ReadTensorArray<Int4>({resolvedPath}, {GetTensorElementCount(tensor).ToString(CultureInfo.InvariantCulture)}L)",
            OnnxTensor<UInt2> => $"OnnxExternalDataProvider.Instance.ReadTensorArray<UInt2>({resolvedPath}, {GetTensorElementCount(tensor).ToString(CultureInfo.InvariantCulture)}L)",
            OnnxTensor<Int2> => $"OnnxExternalDataProvider.Instance.ReadTensorArray<Int2>({resolvedPath}, {GetTensorElementCount(tensor).ToString(CultureInfo.InvariantCulture)}L)",
            OnnxTensor<Complex64> => $"OnnxExternalDataProvider.Instance.ReadTensorArray<Complex64>({resolvedPath})",
            OnnxTensor<Complex128> => $"OnnxExternalDataProvider.Instance.ReadTensorArray<Complex128>({resolvedPath})",
            OnnxTensor<string> => $"OnnxExternalDataProvider.Instance.ReadStringArray({resolvedPath})",
            _ => throw CreateUnsupportedTensorException(tensor),
        };
    }

    private static long GetTensorElementCount(OnnxTensor tensor)
    {
        return tensor.Shape.Aggregate(1L, static (count, dimension) => checked(count * dimension));
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
        if (type == typeof(Half)) return "Half";
        if (type == typeof(BFloat16)) return "BFloat16";
        if (type == typeof(Float8E4M3FN)) return "Float8E4M3FN";
        if (type == typeof(Float8E4M3FNUZ)) return "Float8E4M3FNUZ";
        if (type == typeof(Float8E5M2)) return "Float8E5M2";
        if (type == typeof(Float8E5M2FNUZ)) return "Float8E5M2FNUZ";
        if (type == typeof(Float4E2M1)) return "Float4E2M1";
        if (type == typeof(Float8E8M0)) return "Float8E8M0";
        if (type == typeof(UInt4)) return "UInt4";
        if (type == typeof(Int4)) return "Int4";
        if (type == typeof(UInt2)) return "UInt2";
        if (type == typeof(Int2)) return "Int2";
        if (type == typeof(Complex64)) return "Complex64";
        if (type == typeof(Complex128)) return "Complex128";

        throw new NotSupportedException($"CLR type '{type}' is not supported.");
    }

    private static string SanitizeIdentifier(string value, string fallback)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return fallback;
        }

        var builder = new StringBuilder(value.Length + 1);

        if (!IsIdentifierStart(value[0]))
        {
            builder.Append('_');
        }

        foreach (var ch in value)
        {
            builder.Append(IsIdentifierPart(ch) ? ch : '_');
        }

        var result = builder.ToString();
        return _keywords.Contains(result) ? "@" + result : result;
    }

    private static string SanitizeFileName(string value)
    {
        var invalidChars = Path.GetInvalidFileNameChars();
        var chars = value.Select(ch => invalidChars.Contains(ch) ? '_' : ch).ToArray();
        var result = new string(chars).Trim();
        return string.IsNullOrWhiteSpace(result) ? "tensor" : result;
    }

    private static string Indent(string text, int indentLevel)
    {
        var indent = new string(' ', indentLevel * 4);
        var normalized = text.Replace("\r\n", "\n", StringComparison.Ordinal).TrimEnd();
        var lines = normalized.Split('\n');
        return string.Join(Environment.NewLine, lines.Select(line => string.IsNullOrWhiteSpace(line) ? string.Empty : indent + line));
    }

    private static NotSupportedException CreateUnsupportedTensorException(OnnxTensor tensor)
    {
        return new($"Tensor '{tensor.Name}' uses unsupported data type '{tensor.DataType}'.");
    }

    private static bool IsIdentifierStart(char ch) => ch == '_' || char.IsLetter(ch);
    private static bool IsIdentifierPart(char ch) => ch == '_' || char.IsLetterOrDigit(ch);

    private static readonly HashSet<string> _keywords =
    [
        "class",
        "namespace",
        "public",
        "private",
        "protected",
        "internal",
        "static",
        "void",
        "string",
        "int",
        "long",
        "float",
        "double",
        "bool",
        "object",
        "return",
        "new",
        "base",
        "this",
        "params",
        "out",
        "ref",
        "in",
        "var",
    ];

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
