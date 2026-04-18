using System.Globalization;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
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
                        shape: {{RenderLongArray(GetTensorShape(tensor))}},
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
        using System.Linq;
        using System.Numerics;
        using System.Runtime.InteropServices;
        using System.Text.Json;
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
        }

        internal static class TensorDataLoader
        {
            public static T[] LoadArray<T>(string relativePath, long? expectedElementCount = null) where T : struct
            {
                var path = ResolvePath(relativePath);
                var bytes = File.ReadAllBytes(path);

                if (typeof(T) == typeof(float))
                {
                    return Cast<float, T>(MemoryMarshal.Cast<byte, float>(bytes).ToArray());
                }

                if (typeof(T) == typeof(double))
                {
                    return Cast<double, T>(MemoryMarshal.Cast<byte, double>(bytes).ToArray());
                }

                if (typeof(T) == typeof(sbyte))
                {
                    return Cast<sbyte, T>(MemoryMarshal.Cast<byte, sbyte>(bytes).ToArray());
                }

                if (typeof(T) == typeof(byte))
                {
                    return Cast<byte, T>(bytes);
                }

                if (typeof(T) == typeof(short))
                {
                    return Cast<short, T>(MemoryMarshal.Cast<byte, short>(bytes).ToArray());
                }

                if (typeof(T) == typeof(ushort))
                {
                    return Cast<ushort, T>(MemoryMarshal.Cast<byte, ushort>(bytes).ToArray());
                }

                if (typeof(T) == typeof(int))
                {
                    return Cast<int, T>(MemoryMarshal.Cast<byte, int>(bytes).ToArray());
                }

                if (typeof(T) == typeof(uint))
                {
                    return Cast<uint, T>(MemoryMarshal.Cast<byte, uint>(bytes).ToArray());
                }

                if (typeof(T) == typeof(long))
                {
                    return Cast<long, T>(MemoryMarshal.Cast<byte, long>(bytes).ToArray());
                }

                if (typeof(T) == typeof(ulong))
                {
                    return Cast<ulong, T>(MemoryMarshal.Cast<byte, ulong>(bytes).ToArray());
                }

                if (typeof(T) == typeof(bool))
                {
                    return Cast<bool, T>(bytes.Select(static x => x != 0).ToArray());
                }

                if (typeof(T) == typeof(Half))
                {
                    var values = MemoryMarshal.Cast<byte, ushort>(bytes)
                        .Select(static x => BitConverter.UInt16BitsToHalf(x))
                        .ToArray();
                    return Cast<Half, T>(values);
                }

                if (typeof(T) == typeof(BFloat16))
                {
                    var values = MemoryMarshal.Cast<byte, ushort>(bytes)
                        .Select(static x =>
                        {
                            var bits = (uint)x << 16;
                            return new BFloat16(BitConverter.Int32BitsToSingle((int)bits));
                        })
                        .ToArray();
                    return Cast<BFloat16, T>(values);
                }

                if (typeof(T) == typeof(Float8E4M3FN))
                {
                    return Cast<Float8E4M3FN, T>(MemoryMarshal.Cast<byte, Float8E4M3FN>(bytes).ToArray());
                }

                if (typeof(T) == typeof(Float8E4M3FNUZ))
                {
                    return Cast<Float8E4M3FNUZ, T>(MemoryMarshal.Cast<byte, Float8E4M3FNUZ>(bytes).ToArray());
                }

                if (typeof(T) == typeof(Float8E5M2))
                {
                    return Cast<Float8E5M2, T>(MemoryMarshal.Cast<byte, Float8E5M2>(bytes).ToArray());
                }

                if (typeof(T) == typeof(Float8E5M2FNUZ))
                {
                    return Cast<Float8E5M2FNUZ, T>(MemoryMarshal.Cast<byte, Float8E5M2FNUZ>(bytes).ToArray());
                }

                if (typeof(T) == typeof(Float4E2M1))
                {
                    return Cast<Float4E2M1, T>(UnpackNibbles<Float4E2M1>(bytes, expectedElementCount));
                }

                if (typeof(T) == typeof(Float8E8M0))
                {
                    return Cast<Float8E8M0, T>(MemoryMarshal.Cast<byte, Float8E8M0>(bytes).ToArray());
                }

                if (typeof(T) == typeof(UInt4))
                {
                    return Cast<UInt4, T>(UnpackNibbles<UInt4>(bytes, expectedElementCount));
                }

                if (typeof(T) == typeof(Int4))
                {
                    return Cast<Int4, T>(UnpackNibbles<Int4>(bytes, expectedElementCount));
                }

                if (typeof(T) == typeof(UInt2))
                {
                    return Cast<UInt2, T>(UnpackTwoBitValues<UInt2>(bytes, expectedElementCount));
                }

                if (typeof(T) == typeof(Int2))
                {
                    return Cast<Int2, T>(UnpackTwoBitValues<Int2>(bytes, expectedElementCount));
                }

                if (typeof(T) == typeof(Complex64))
                {
                    var raw = MemoryMarshal.Cast<byte, float>(bytes).ToArray();
                    var values = new Complex64[raw.Length / 2];

                    for (var index = 0; index < values.Length; index++)
                    {
                        values[index] = new Complex64(raw[index * 2], raw[index * 2 + 1]);
                    }

                    return Cast<Complex64, T>(values);
                }

                if (typeof(T) == typeof(Complex128))
                {
                    var raw = MemoryMarshal.Cast<byte, double>(bytes).ToArray();
                    var values = new Complex128[raw.Length / 2];

                    for (var index = 0; index < values.Length; index++)
                    {
                        values[index] = new Complex128(raw[index * 2], raw[index * 2 + 1]);
                    }

                    return Cast<Complex128, T>(values);
                }

                throw new NotSupportedException($"Tensor data type '{typeof(T)}' is not supported.");
            }

            public static string[] LoadStringArray(string relativePath)
            {
                var path = ResolvePath(relativePath);
                return JsonSerializer.Deserialize<string[]>(File.ReadAllText(path))
                    ?? throw new InvalidOperationException($"Could not deserialize tensor data from '{path}'.");
            }

            private static string ResolvePath(string relativePath)
            {
                return Path.Combine(AppContext.BaseDirectory, relativePath);
            }

            private static TByteBacked[] UnpackNibbles<TByteBacked>(byte[] bytes, long? expectedElementCount)
                where TByteBacked : struct
            {
                var rawValues = new byte[bytes.Length * 2];
                var index = 0;

                foreach (var value in bytes)
                {
                    rawValues[index++] = (byte)(value & 0x0F);
                    rawValues[index++] = (byte)(value >> 4);
                }

                return TrimByteBacked<TByteBacked>(rawValues, expectedElementCount);
            }

            private static TByteBacked[] UnpackTwoBitValues<TByteBacked>(byte[] bytes, long? expectedElementCount)
                where TByteBacked : struct
            {
                var rawValues = new byte[bytes.Length * 4];
                var index = 0;

                foreach (var value in bytes)
                {
                    rawValues[index++] = (byte)(value & 0x03);
                    rawValues[index++] = (byte)((value >> 2) & 0x03);
                    rawValues[index++] = (byte)((value >> 4) & 0x03);
                    rawValues[index++] = (byte)((value >> 6) & 0x03);
                }

                return TrimByteBacked<TByteBacked>(rawValues, expectedElementCount);
            }

            private static TByteBacked[] TrimByteBacked<TByteBacked>(byte[] rawValues, long? expectedElementCount)
                where TByteBacked : struct
            {
                var values = MemoryMarshal.Cast<byte, TByteBacked>(rawValues).ToArray();
                if (expectedElementCount is not long count)
                {
                    return values;
                }

                if (count < 0 || count > values.LongLength)
                {
                    throw new InvalidOperationException($"Tensor payload length mismatch. Expected {count} items, got {values.LongLength}.");
                }

                return values.AsSpan(0, checked((int)count)).ToArray();
            }

            private static TTarget[] Cast<TSource, TTarget>(TSource[] values)
                where TSource : struct
                where TTarget : struct
            {
                return values.Cast<TTarget>().ToArray();
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

        return tensor switch
        {
            OnnxTensor<float> x => context.WriteBinaryAsset(safeName, "bin", Pack(x.Value.ToArray())),
            OnnxTensor<double> x => context.WriteBinaryAsset(safeName, "bin", Pack(x.Value.ToArray())),
            OnnxTensor<sbyte> x => context.WriteBinaryAsset(safeName, "bin", MemoryMarshal.AsBytes(x.Value.ToArray().AsSpan()).ToArray()),
            OnnxTensor<byte> x => context.WriteBinaryAsset(safeName, "bin", x.Value.ToArray()),
            OnnxTensor<short> x => context.WriteBinaryAsset(safeName, "bin", Pack(x.Value.ToArray())),
            OnnxTensor<ushort> x => context.WriteBinaryAsset(safeName, "bin", Pack(x.Value.ToArray())),
            OnnxTensor<int> x => context.WriteBinaryAsset(safeName, "bin", Pack(x.Value.ToArray())),
            OnnxTensor<uint> x => context.WriteBinaryAsset(safeName, "bin", Pack(x.Value.ToArray())),
            OnnxTensor<long> x => context.WriteBinaryAsset(safeName, "bin", Pack(x.Value.ToArray())),
            OnnxTensor<ulong> x => context.WriteBinaryAsset(safeName, "bin", Pack(x.Value.ToArray())),
            OnnxTensor<bool> x => context.WriteBinaryAsset(safeName, "bin", x.Value.Select(static value => (byte)(value ? 1 : 0)).ToArray()),
            OnnxTensor<Half> x => context.WriteBinaryAsset(safeName, "bin", PackHalf(x.Value.ToArray())),
            OnnxTensor<BFloat16> x => context.WriteBinaryAsset(safeName, "bin", PackBFloat16(x.Value.ToArray())),
            OnnxTensor<Float8E4M3FN> x => context.WriteBinaryAsset(safeName, "bin", PackByteBacked(x.Value.ToArray(), static value => value.Value)),
            OnnxTensor<Float8E4M3FNUZ> x => context.WriteBinaryAsset(safeName, "bin", PackByteBacked(x.Value.ToArray(), static value => value.Value)),
            OnnxTensor<Float8E5M2> x => context.WriteBinaryAsset(safeName, "bin", PackByteBacked(x.Value.ToArray(), static value => value.Value)),
            OnnxTensor<Float8E5M2FNUZ> x => context.WriteBinaryAsset(safeName, "bin", PackByteBacked(x.Value.ToArray(), static value => value.Value)),
            OnnxTensor<Float4E2M1> x => context.WriteBinaryAsset(safeName, "bin", PackNibbles(x.Value.ToArray(), static value => value.Value)),
            OnnxTensor<Float8E8M0> x => context.WriteBinaryAsset(safeName, "bin", PackByteBacked(x.Value.ToArray(), static value => value.Value)),
            OnnxTensor<UInt4> x => context.WriteBinaryAsset(safeName, "bin", PackNibbles(x.Value.ToArray(), static value => value.Value)),
            OnnxTensor<Int4> x => context.WriteBinaryAsset(safeName, "bin", PackNibbles(x.Value.ToArray(), static value => value.Value)),
            OnnxTensor<UInt2> x => context.WriteBinaryAsset(safeName, "bin", PackTwoBitValues(x.Value.ToArray(), static value => value.Value)),
            OnnxTensor<Int2> x => context.WriteBinaryAsset(safeName, "bin", PackTwoBitValues(x.Value.ToArray(), static value => value.Value)),
            OnnxTensor<Complex64> x => context.WriteBinaryAsset(safeName, "bin", PackComplex64(x.Value.ToArray())),
            OnnxTensor<Complex128> x => context.WriteBinaryAsset(safeName, "bin", PackComplex128(x.Value.ToArray())),
            OnnxTensor<string> x => context.WriteTextAsset(safeName, "json", JsonSerializer.Serialize(x.Value.ToArray())),
            _ => throw CreateUnsupportedTensorException(tensor),
        };
    }

    private static byte[] Pack<T>(T[] values) where T : struct
    {
        return MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
    }

    private static byte[] PackHalf(Half[] values)
    {
        var buffer = new byte[values.Length * sizeof(ushort)];
        for (var index = 0; index < values.Length; index++)
        {
            BitConverter.TryWriteBytes(buffer.AsSpan(index * sizeof(ushort), sizeof(ushort)), BitConverter.HalfToUInt16Bits(values[index]));
        }

        return buffer;
    }

    private static byte[] PackBFloat16(BFloat16[] values)
    {
        var buffer = new byte[values.Length * sizeof(ushort)];
        for (var index = 0; index < values.Length; index++)
        {
            BitConverter.TryWriteBytes(buffer.AsSpan(index * sizeof(ushort), sizeof(ushort)), values[index].Value);
        }

        return buffer;
    }

    private static byte[] PackByteBacked<T>(T[] values, Func<T, byte> selector) where T : struct
    {
        return values.Select(selector).ToArray();
    }

    private static byte[] PackNibbles<T>(T[] values, Func<T, byte> selector) where T : struct
    {
        var buffer = new byte[(values.Length + 1) / 2];

        for (var index = 0; index < values.Length; index += 2)
        {
            var low = (byte)(selector(values[index]) & 0x0F);
            var high = index + 1 < values.Length
                ? (byte)((selector(values[index + 1]) & 0x0F) << 4)
                : (byte)0;

            buffer[index / 2] = (byte)(low | high);
        }

        return buffer;
    }

    private static byte[] PackTwoBitValues<T>(T[] values, Func<T, byte> selector) where T : struct
    {
        var buffer = new byte[(values.Length + 3) / 4];

        for (var index = 0; index < values.Length; index += 4)
        {
            var packed = (byte)(selector(values[index]) & 0x03);

            if (index + 1 < values.Length)
            {
                packed |= (byte)((selector(values[index + 1]) & 0x03) << 2);
            }

            if (index + 2 < values.Length)
            {
                packed |= (byte)((selector(values[index + 2]) & 0x03) << 4);
            }

            if (index + 3 < values.Length)
            {
                packed |= (byte)((selector(values[index + 3]) & 0x03) << 6);
            }

            buffer[index / 4] = packed;
        }

        return buffer;
    }

    private static byte[] PackComplex64(Complex64[] values)
    {
        var buffer = new float[values.Length * 2];
        for (var index = 0; index < values.Length; index++)
        {
            buffer[index * 2] = values[index].Real;
            buffer[index * 2 + 1] = values[index].Imaginary;
        }

        return Pack(buffer);
    }

    private static byte[] PackComplex128(Complex128[] values)
    {
        var buffer = new double[values.Length * 2];
        for (var index = 0; index < values.Length; index++)
        {
            buffer[index * 2] = values[index].Real;
            buffer[index * 2 + 1] = values[index].Imaginary;
        }

        return Pack(buffer);
    }

    private static long[] GetTensorShape(OnnxTensor tensor)
    {
        return tensor switch
        {
            OnnxTensor<float> x => x.Shape,
            OnnxTensor<double> x => x.Shape,
            OnnxTensor<sbyte> x => x.Shape,
            OnnxTensor<byte> x => x.Shape,
            OnnxTensor<short> x => x.Shape,
            OnnxTensor<ushort> x => x.Shape,
            OnnxTensor<int> x => x.Shape,
            OnnxTensor<uint> x => x.Shape,
            OnnxTensor<long> x => x.Shape,
            OnnxTensor<ulong> x => x.Shape,
            OnnxTensor<bool> x => x.Shape,
            OnnxTensor<Half> x => x.Shape,
            OnnxTensor<BFloat16> x => x.Shape,
            OnnxTensor<Float8E4M3FN> x => x.Shape,
            OnnxTensor<Float8E4M3FNUZ> x => x.Shape,
            OnnxTensor<Float8E5M2> x => x.Shape,
            OnnxTensor<Float8E5M2FNUZ> x => x.Shape,
            OnnxTensor<Float4E2M1> x => x.Shape,
            OnnxTensor<Float8E8M0> x => x.Shape,
            OnnxTensor<UInt4> x => x.Shape,
            OnnxTensor<Int4> x => x.Shape,
            OnnxTensor<UInt2> x => x.Shape,
            OnnxTensor<Int2> x => x.Shape,
            OnnxTensor<Complex64> x => x.Shape,
            OnnxTensor<Complex128> x => x.Shape,
            OnnxTensor<string> x => x.Shape,
            _ => throw CreateUnsupportedTensorException(tensor),
        };
    }

    private static string RenderTensorLoadExpression(OnnxTensor tensor, string relativePath)
    {
        var literalPath = AsStringLiteral(relativePath.Replace('\\', '/'));

        return tensor switch
        {
            OnnxTensor<float> => $"TensorDataLoader.LoadArray<float>({literalPath})",
            OnnxTensor<double> => $"TensorDataLoader.LoadArray<double>({literalPath})",
            OnnxTensor<sbyte> => $"TensorDataLoader.LoadArray<sbyte>({literalPath})",
            OnnxTensor<byte> => $"TensorDataLoader.LoadArray<byte>({literalPath})",
            OnnxTensor<short> => $"TensorDataLoader.LoadArray<short>({literalPath})",
            OnnxTensor<ushort> => $"TensorDataLoader.LoadArray<ushort>({literalPath})",
            OnnxTensor<int> => $"TensorDataLoader.LoadArray<int>({literalPath})",
            OnnxTensor<uint> => $"TensorDataLoader.LoadArray<uint>({literalPath})",
            OnnxTensor<long> => $"TensorDataLoader.LoadArray<long>({literalPath})",
            OnnxTensor<ulong> => $"TensorDataLoader.LoadArray<ulong>({literalPath})",
            OnnxTensor<bool> => $"TensorDataLoader.LoadArray<bool>({literalPath})",
            OnnxTensor<Half> => $"TensorDataLoader.LoadArray<Half>({literalPath})",
            OnnxTensor<BFloat16> => $"TensorDataLoader.LoadArray<BFloat16>({literalPath})",
            OnnxTensor<Float8E4M3FN> => $"TensorDataLoader.LoadArray<Float8E4M3FN>({literalPath})",
            OnnxTensor<Float8E4M3FNUZ> => $"TensorDataLoader.LoadArray<Float8E4M3FNUZ>({literalPath})",
            OnnxTensor<Float8E5M2> => $"TensorDataLoader.LoadArray<Float8E5M2>({literalPath})",
            OnnxTensor<Float8E5M2FNUZ> => $"TensorDataLoader.LoadArray<Float8E5M2FNUZ>({literalPath})",
            OnnxTensor<Float4E2M1> => $"TensorDataLoader.LoadArray<Float4E2M1>({literalPath}, {GetTensorElementCount(tensor).ToString(CultureInfo.InvariantCulture)}L)",
            OnnxTensor<Float8E8M0> => $"TensorDataLoader.LoadArray<Float8E8M0>({literalPath})",
            OnnxTensor<UInt4> => $"TensorDataLoader.LoadArray<UInt4>({literalPath}, {GetTensorElementCount(tensor).ToString(CultureInfo.InvariantCulture)}L)",
            OnnxTensor<Int4> => $"TensorDataLoader.LoadArray<Int4>({literalPath}, {GetTensorElementCount(tensor).ToString(CultureInfo.InvariantCulture)}L)",
            OnnxTensor<UInt2> => $"TensorDataLoader.LoadArray<UInt2>({literalPath}, {GetTensorElementCount(tensor).ToString(CultureInfo.InvariantCulture)}L)",
            OnnxTensor<Int2> => $"TensorDataLoader.LoadArray<Int2>({literalPath}, {GetTensorElementCount(tensor).ToString(CultureInfo.InvariantCulture)}L)",
            OnnxTensor<Complex64> => $"TensorDataLoader.LoadArray<Complex64>({literalPath})",
            OnnxTensor<Complex128> => $"TensorDataLoader.LoadArray<Complex128>({literalPath})",
            OnnxTensor<string> => $"TensorDataLoader.LoadStringArray({literalPath})",
            _ => throw CreateUnsupportedTensorException(tensor),
        };
    }

    private static long GetTensorElementCount(OnnxTensor tensor)
    {
        return GetTensorShape(tensor).Aggregate(1L, static (count, dimension) => checked(count * dimension));
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
        return $"CreateDetachedTensor({AsStringLiteral(tensor.Name)}, {RenderLongArray(GetTensorShape(tensor))}, {RenderTensorLoadExpression(tensor, assetRelativePath)})";
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
