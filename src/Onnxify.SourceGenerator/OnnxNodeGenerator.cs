using System.Text;
using System.Text.Json;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Text;
using Onnxify.SourceGenerator.Models;

namespace Onnxify.SourceGenerator
{
    [Generator]
    public class OnnxNodeGenerator : IIncrementalGenerator
    {
        public void Initialize(IncrementalGeneratorInitializationContext context)
        {
            var assemblyName = context.CompilationProvider.Select((c, _) => c.AssemblyName);
            var texts = context.AdditionalTextsProvider;
            var combined = texts.Combine(assemblyName);

            context.RegisterSourceOutput(
                combined,
                (productionContext, sourceContext) =>
                {
                    Generate(
                        context: productionContext,
                        file: sourceContext.Left
                    );
                }
            );
        }

        public void Generate(
            SourceProductionContext context,
            AdditionalText file
        )
        {
            if (Path.GetFileName(file.Path) != "onnx_operators.json")
            {
                return;
            }

            context.ReportDiagnostic(
                Diagnostic.Create(
                    new DiagnosticDescriptor("GEN001", "test", $"Generator running: {file.Path}", "gen", DiagnosticSeverity.Info, true),
                    Location.None
                )
            );

            var json = file.GetText()?.ToString() ?? string.Empty;
            var root = JsonSerializer.Deserialize<OperatorSchemaRoot>(json) ?? throw new Exception();

            var classes = new StringBuilder();

            foreach (var op in root.Operators)
            {
                context.ReportDiagnostic(
                    Diagnostic.Create(
                        new DiagnosticDescriptor("GEN001", "test", op.Name, "gen", DiagnosticSeverity.Warning, true),
                        Location.None
                    )
                );

                var className = $"{op.Name}";

                var list1 = new List<string>();

                list1.AddRange(op.Inputs.Select(x => $"{InputName(x.Name)} = options.{InputName(x.Name)}"));
                list1.AddRange(op.Attributes.Select(x => $"{AttributeName(x.Name)} = options.{AttributeName(x.Name)}"));
                list1.AddRange(op.Outputs.Select(x => $"{OutputName(x.Name)} = edge{OutputName(x.Name)}"));

                var optionsInputs = op.Inputs.Select(x => $"options.{InputName(x.Name)}").ToArray();
                var optionsOutputs = op.Outputs.Select(x => $"options.{OutputName(x.Name)}").ToArray();
                var optionsAttributes = op.Attributes
                    .Select(x =>
                    {
                        var type = FromProto((AttributeType)x.Type);
                        return $"() => options.{AttributeName(x.Name)} is not null ? new OnnxAttribute<{type}>(\"{x.Name}\", ({type})options.{AttributeName(x.Name)}) : null";
                    })
                    .ToArray();

                var optionsAttributesList = optionsAttributes.Length > 0 ? $$"""
                OnnxHelper.NotNull<OnnxAttribute>(
                    new List<Func<OnnxAttribute?>>() 
                    { 
                        {{string.Join(",\n            ", optionsAttributes)}}
                    }
                    .Select(x => x())
                    .ToArray()
                )
                """ : "[]";

                classes.AppendLine($$"""
                    public class {{className}}InputOptions
                    {
                        {{GetFields(op.Inputs, x => InputName(x))}}

                        {{GetFields(op.Attributes, x => AttributeName(x))}}
                    }

                    public class {{className}}InputOutputOptions : {{className}}InputOptions
                    {
                        {{GetFields(op.Outputs, x => OutputName(x))}}
                    }

                    public class {{className}} : OnnxNode
                    {
                        public {{className}}(
                            string name,
                            {{className}}InputOutputOptions options
                        ) : base(
                            name: name,
                            opType: "{{op.Name}}",
                            domain: "",
                            docString: "",
                            inputs: OnnxHelper.NotNull<IOnnxGraphEdge>([{{string.Join(", ", optionsInputs)}}]),
                            outputs: OnnxHelper.NotNull<IOnnxGraphEdge>([{{string.Join(", ", optionsOutputs)}}]),
                            attributes: {{optionsAttributesList}}
                        ) { }
                    }

                    public static class {{className}}Extensions
                    {
                        public static IOnnxGraphEdge {{className}}(
                            this OnnxGraph graph,
                            string name,
                            {{className}}InputOptions options
                        )
                        {
                            {{string.Join("\n            ", (
                                op.Outputs.Select(x => (
                                    $"var edge{OutputName(x.Name)} = graph.AddEdge(name + \"_{x.Name}\");"
                                )))
                            )}}
                            
                            var op = new {{className}}(
                                name: name,
                                options: new {{className}}InputOutputOptions
                                {
                                    {{string.Join(", ", list1)}}
                                }
                            );

                            graph.AddNode(op);
                            // return {{string.Join(", ", op.Outputs.Select(x => $"options.{OutputName(x.Name)}"))}};
                            return null;
                        }

                        public static IOnnxGraphEdge {{className}}(
                            this OnnxGraph graph,
                            string name,
                            {{className}}InputOutputOptions options
                        )
                        {
                            var op = new {{className}}(
                                name: name,
                                options: options
                            );

                            graph.AddNode(op);
                            // return {{string.Join(", ", op.Outputs.Select(x => $"options.{OutputName(x.Name)}"))}};
                            return null;
                        }
                    }

                """);
            }

            var namespaceName = $"{nameof(Onnxify)}.Operators";

            var code = $$"""
            // <auto-generated/>
            #nullable enable
            
            using System;
            using System.Collections.Generic;
            using Onnxify;
            using Onnx;

            namespace {{namespaceName}}
            {
            {{classes}}
            }

            #nullable restore
            """;

            context.AddSource($"{namespaceName}.g.cs", SourceText.From(code, Encoding.UTF8));
        }

        private string GetFields(IEnumerable<OperatorParameter> items, Func<string, string> onName)
        {
            var sb = new StringBuilder();

            foreach (var x in items)
            {
                var required = x.Option == FormalParameterOption.Single ? " required " : " ";
                var nullable = x.Option == FormalParameterOption.Optional ? "?" : "";

                sb.AppendLine($$"""
                        public{{required}}{{MapType(x.Type)}}{{nullable}} {{onName(x.Name)}} { get; init; }
                """);
            }

            return sb.ToString().Trim();
        }
        
        private string GetFields(IEnumerable<OperatorAttribute> items, Func<string, string> onName)
        {
            var sb = new StringBuilder();

            foreach (var x in items)
            {
                var required = x.Required ? " required " : " ";
                var nullable = x.Required ? "" : "?";
                var typeEnum = (AttributeType)x.Type;

                sb.AppendLine($$"""
                        public{{required}}{{FromProto(typeEnum)}}{{nullable}} {{AttributeName(x.Name)}} { get; set; }
                """);
            }

            return sb.ToString().Trim();
        }

        private static string PascalCase(string s)
        {
            if (string.IsNullOrEmpty(s))
            {
                return s;
            }

            var words = s.Split(['_'], StringSplitOptions.RemoveEmptyEntries);

            var result = new StringBuilder();

            foreach (var word in words)
            {
                var newWord = char.ToUpperInvariant(word[0]) + word.Substring(1);
                result.Append(newWord);
            }

            return result.ToString();
        }

        public static string InputName(string name, string prefix = "")
        {
            var p = PascalCase(name);

            if (!string.IsNullOrEmpty(prefix) && p.Equals(prefix, StringComparison.OrdinalIgnoreCase))
            {
                return prefix;
            }

            return prefix + p;
        }

        public static string OutputName(string name, string prefix = "")
        {
            var p = PascalCase(name);

            if (!string.IsNullOrEmpty(prefix) && p.Equals(prefix, StringComparison.OrdinalIgnoreCase))
            {
                return prefix;
            }

            return prefix + p;
        }

        public static string AttributeName(string name, string prefix = "")
        {
            var p = PascalCase(name);

            if (!string.IsNullOrEmpty(prefix) && p.Equals(prefix, StringComparison.OrdinalIgnoreCase))
            {
                return prefix;
            }

            return prefix + p;
        }

        public static string MapType(string type)
        {
            return type switch
            {
                "T" => "IOnnxGraphEdge",
                "Tind" => "IOnnxGraphEdge",
                "T1" => "IOnnxGraphEdge",
                "tensor(int64)" => "OnnxTensor<long>",
                "tensor(int32)" => "OnnxTensor<int>",
                "tensor(float)" => "OnnxTensor<float>",
                "tensor(double)" => "OnnxTensor<double>",
                "tensor(bool)" => "OnnxTensor<bool>",

                _ => throw new NotSupportedException($"Unsupported ONNX type: {type}")
            };
        }

        public static string FromProto(AttributeType attributeType)
        {
            return attributeType switch
            {
                AttributeType.Float => "float",
                AttributeType.Int => "long",
                AttributeType.String => "string",

                AttributeType.Tensor => "OnnxTensor",
                AttributeType.Graph => "OnnxGraph",
                AttributeType.SparseTensor => "OnnxSparseTensorBase",

                AttributeType.Floats => "float[]",
                AttributeType.Ints => "long[]",
                AttributeType.Strings => "string[]",

                AttributeType.Tensors => "OnnxTensor[]",
                AttributeType.Graphs => "OnnxGraph[]",
                AttributeType.SparseTensors => "OnnxSparseTensorBase[]",

                _ => throw new NotImplementedException($"Not implemented for '{attributeType}'"),
            };
        }



        /*foreach (var op in root.Operators)
        {
            var propBuilder = new StringBuilder();

            foreach (var x in op.Inputs)
            {
                var required = x.Option == FormalParameterOption.Single ? " required " : " ";
                var nullable = x.Option == FormalParameterOption.Optional ? "?" : "";

                propBuilder.AppendLine($$"""
                    public{{required}}FormalParameter<{{MapType(x.Type)}}>{{nullable}} {{InputName(x.Name)}} { get; set; }
                """);
            }

            if (op.Inputs.Count != 0)
            {
                propBuilder.AppendLine();
            }

            foreach (var x in op.Outputs)
            {
                var required = x.Option == FormalParameterOption.Single ? " required " : " ";
                var nullable = x.Option == FormalParameterOption.Optional ? "?" : "";

                propBuilder.AppendLine($$"""
                    public{{required}}FormalParameter<{{MapType(x.Type)}}>{{nullable}} {{OutputName(x.Name)}} { get; set; }
                """);
            }

            if (op.Outputs.Count != 0)
            {
                propBuilder.AppendLine();
            }

            foreach (var x in op.Attributes)
            {
                var required = x.Required ? " required " : " ";
                var nullable = x.Required ? "" : "?";

                var typeEnum = x.Type;
            
                propBuilder.AppendLine($$"""
                    public{{required}}OperatorAttribute<{{OperatorHelpers.MapAttributeType(typeEnum)}}>{{nullable}} {{AttributeName(x.Name)}} { get; set; }
                """);
            }

            if (op.Attributes.Count != 0)
            {
                propBuilder.AppendLine();
            }

            sourceBuilder.AppendLine($$"""
            /// <summary>
            /// {{op.Name}} operator:
            /// <para>
            /// {{(op.Doc ?? "").Trim().Replace("\n", $"\n/// ")}}
            /// </para>
            /// </summary>
            public sealed class {{op.Name}} : Operator
            {
                public override string Name => "{{op.Name}}";
                public override string Domain => "{{op.Domain}}";
                public override int SinceVersion => {{op.SinceVersion}};

                {{propBuilder.ToString().TrimStart()}}
            }

            """);
        }

        var sourceCode = sourceBuilder.ToString();
        */
        // File.WriteAllText(outputPath, sourceCode);
    }
}

public enum AttributeType
{
    Undefined = 0,
    Float = 1,
    Int = 2,
    String = 3,
    Tensor = 4,
    Graph = 5,
    SparseTensor = 11,
    Floats = 6,
    Ints = 7,
    Strings = 8,
    Tensors = 9,
    Graphs = 10,
    SparseTensors = 12,
}
