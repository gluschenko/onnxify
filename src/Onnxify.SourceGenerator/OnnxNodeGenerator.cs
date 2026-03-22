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
        private readonly static HashSet<string> _reservedFieldNames = [];
        private readonly static Dictionary<string, string> _aliasFieldNames = new() 
        {
            ["Inputs"] = "In",
            ["Outputs"] = "Out",
        };

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
            var typeResolver = new List<string>();

            foreach (var op in root.Operators)
            {
                context.ReportDiagnostic(
                    Diagnostic.Create(
                        new DiagnosticDescriptor("GEN001", "test", op.Name, "gen", DiagnosticSeverity.Warning, true),
                        Location.None
                    )
                );

                _reservedFieldNames.Clear();
                var className = $"{op.Name}";
                _reservedFieldNames.Add(className);
                _reservedFieldNames.Add("Inputs");
                _reservedFieldNames.Add("Outputs");

                var optionsInputs = op.Inputs.Select(x => $"options.{InputName(x.Name)}").ToArray();
                var optionsOutputs = op.Outputs.Select(x => $"options.{OutputName(x.Name)}").ToArray();
                var optionsAttributes = op.Attributes
                    .Select(x =>
                    {
                        var type = FromProto((AttributeType)x.Type);
                        return $"() => options.{AttributeName(x.Name)} is {type} x ? new OnnxAttribute<{type}>(\"{x.Name}\", x) : null";
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
                """);

                var nodeFields = new StringBuilder();

                for (var i = 0; i < op.Inputs.Count(); i++)
                {
                    var x = op.Inputs[i];

                    nodeFields.AppendLine($$"""
                            /// <summary>
                            /// <para>
                            /// {{x.Name}}: {{(x.Description ?? "").Trim().Replace("\n", $"\n    /// ").Replace("<", "&lt;").Replace(">", "&gt;")}}
                            /// </para>
                            /// <para>
                            /// Allowed types: {{string.Join(", ", x.Types)}}
                            /// </para>
                            /// </summary>
                    """);

                    if (x.Option != FormalParameterOption.Optional)
                    {
                        nodeFields.AppendLine($$"""
                                public {{MapType(x.Types)}} {{InputName(x.Name)}}
                                {
                                    get => ({{MapType(x.Types)}})Inputs[{{i}}];
                                    set => SetInput({{i}}, value);
                                }
                        """);
                    }
                    else
                    {
                        nodeFields.AppendLine($$"""
                                public {{MapType(x.Types)}}? {{InputName(x.Name)}}
                                {
                                    get => Inputs.Count > {{i}} ? ({{MapType(x.Types)}})Inputs[{{i}}] : null;
                                    set => SetOptionalInput({{i}}, value);
                                }
                        """);
                    }
                }

                for (var i = 0; i < op.Outputs.Count(); i++)
                {
                    var x = op.Outputs[i];

                    nodeFields.AppendLine($$"""
                            /// <summary>
                            /// <para>
                            /// {{x.Name}}: {{(x.Description ?? "").Trim().Replace("\n", $"\n    /// ").Replace("<", "&lt;").Replace(">", "&gt;")}}
                            /// </para>
                            /// <para>
                            /// Allowed types: {{string.Join(", ", x.Types)}}
                            /// </para>
                            /// </summary>
                    """);

                    if (x.Option != FormalParameterOption.Optional)
                    {
                        nodeFields.AppendLine($$"""
                                public {{MapType(x.Types)}} {{OutputName(x.Name)}}
                                {
                                    get => ({{MapType(x.Types)}})Outputs[{{i}}];
                                    set => SetOutput({{i}}, value);
                                }
                        """);
                    }
                    else
                    {
                        nodeFields.AppendLine($$"""
                                public {{MapType(x.Types)}}? {{OutputName(x.Name)}}
                                {
                                    get => Outputs.Count > {{i}} && Outputs[{{i}}] is {{MapType(x.Types)}} x ? x : null;
                                    set => SetOptionalOutput({{i}}, value);
                                }
                        """);
                    }
                }

                for (var i = 0; i < op.Attributes.Count(); i++)
                {
                    var x = op.Attributes[i];

                    var type = FromProto((AttributeType)x.Type);

                    nodeFields.AppendLine($$"""
                            /// <summary>
                            /// <para>
                            /// {{x.Name}}: {{(x.Description ?? "").Trim().Replace("\n", $"\n    /// ").Replace("<", "&lt;").Replace(">", "&gt;")}}
                            /// </para>
                            /// <para>
                            /// Allowed types: {{string.Join(", ", [type])}}
                            /// </para>
                            /// </summary>
                    """);

                    if (x.Required)
                    {
                        nodeFields.AppendLine($$"""
                                public {{type}} {{AttributeName(x.Name)}}
                                {
                                    get => GetAttribute<{{type}}>("{{x.Name}}");
                                    set => SetAttribute<{{type}}>("{{x.Name}}", value);
                                }
                        """);
                    }
                    else
                    {
                        nodeFields.AppendLine($$"""
                                public {{type}}? {{AttributeName(x.Name)}}
                                {
                                    get => HasAttribute("{{x.Name}}") ? GetAttribute<{{type}}>("{{x.Name}}") : null;
                                    set {
                                        if (value is not null)
                                        {
                                            SetAttribute<{{type}}>("{{x.Name}}", ({{type}})value);
                                        }
                                        else
                                        {
                                            RemoveAttribute("{{x.Name}}");
                                        }
                                    }
                                }
                        """);
                    }
                }

                var comment = $$"""
                        /// <summary>
                        /// {{op.Name}} operator:
                        /// <para>
                        /// {{(op.Doc ?? "").Trim().Replace("\n", $"\n    /// ").Replace("<", "&lt;").Replace(">", "&gt;")}}
                        /// </para>
                        /// </summary>
                """;

                var list1 = new List<string>();

                list1.AddRange(op.Inputs.Select(x => $"{InputName(x.Name)} = options.{InputName(x.Name)}"));
                list1.AddRange(op.Attributes.Select(x => $"{AttributeName(x.Name)} = options.{AttributeName(x.Name)}"));
                list1.AddRange(op.Outputs.Select(x => $"{OutputName(x.Name)} = edge{OutputName(x.Name)}"));

                var list2 = new List<string>();

                list2.AddRange(op.Inputs.Select((x, i) =>
                {
                    if (x.Option != FormalParameterOption.Optional)
                    {
                        return $"{InputName(x.Name)} = inputs[{i}]";
                    }
                    else
                    {
                        return $"{InputName(x.Name)} = inputs.Length > {i} ? inputs[{i}] : null";
                    }
                }));
                list2.AddRange(op.Attributes.Select(x =>
                {
                    var type = FromProto((AttributeType)x.Type);

                    if (x.Required)
                    {
                        return $"{AttributeName(x.Name)} = ({type})(attributes.GetValueOrDefault(\"{x.Name}\") ?? throw new InvalidOperationException($\"Missing value '{x.Name}'\"))";
                    }
                    else
                    {
                        return $"{AttributeName(x.Name)} = ({type}?)attributes.GetValueOrDefault(\"{x.Name}\")";
                    }
                }));
                list2.AddRange(op.Outputs.Select((x, i) =>
                {
                    if (x.Option != FormalParameterOption.Optional)
                    {
                        return $"{OutputName(x.Name)} = outputs[{i}]";
                    }
                    else
                    {
                        return $"{OutputName(x.Name)} = outputs.Length > {i} ? outputs[{i}] : null";
                    }
                }));

                classes.AppendLine($$"""
                    {{comment}}
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
                            attributes: CreateAttributes(options)
                        ) { }

                        {{nodeFields.ToString().Trim()}}

                        internal static OnnxAttribute[] CreateAttributes({{className}}InputOutputOptions options)
                        {
                            return {{optionsAttributesList}};
                        }

                        internal static {{className}} {{className}}FromProto(NodeProto node, OnnxGraph graph)
                        {
                            if (node.OpType != "{{op.Name}}")
                            {
                                throw new InvalidOperationException($"Node type is not valid '{node.OpType}' != '{{op.Name}}'");
                            }
                            
                            var inputs = node.Input
                                .Select(x => graph.GetValue(x) ?? throw new InvalidOperationException($"Missing value '{x}'"))
                                .ToArray();

                            var outputs = node.Output
                                .Select(x => graph.GetValue(x) ?? throw new InvalidOperationException($"Missing value '{x}'"))
                                .ToArray();

                            var attributes = node.Attribute.ToDictionary(x => x.Name, x => x.GetValue());

                            var op = new {{className}}(
                                name: node.Name,
                                options: new {{className}}InputOutputOptions
                                {
                                    {{string.Join(",\n", list2)}},
                                }
                            );

                            return op;
                        }
                    }
                """);

                if (op.Outputs.Count() > 1)
                {
                    classes.AppendLine($$""""
                        public class {{className}}Output
                        {
                            {{GetFields(op.Outputs, x => OutputName(x))}}
                        }
                    """");
                }

                var extensionMethodReturnType = op.Outputs.Count <= 1 
                    ? "IOnnxGraphEdge" 
                    : $"{className}Output";

                var extensionMethodReturnValue = op.Outputs.Count <= 1 
                    ? string.Join(", ", op.Outputs.Select(x => $"op.{OutputName(x.Name)}")) 
                    : $$"""
                                new {{className}}Output 
                                { 
                                    {{string.Join(",\n", op.Outputs.Select(x => $"{OutputName(x.Name)} = op.{OutputName(x.Name)}"))}} 
                                }
                    """;

                classes.AppendLine($$"""
                    public static class {{className}}Extensions
                    {
                        {{comment}}
                        public static {{extensionMethodReturnType}} {{className}}(
                            this OnnxGraph graph,
                            string name,
                            {{className}}InputOptions options
                        )
                        {
                            {{string.Join("\n            ", (
                                op.Outputs.Select(x => (
                                    $"var edge{OutputName(x.Name)} = graph.AddEdge(name + \"_output_{x.Name.ToLower()}\");"
                                )))
                            )}}
                            
                            var op = new {{className}}(
                                name: name,
                                options: new {{className}}InputOutputOptions
                                {
                                    {{string.Join(",\n", list1)}}
                                }
                            );
                
                            graph.AddNode(op);
                
                            return {{extensionMethodReturnValue}};
                        }
                
                        {{comment}}
                        public static {{extensionMethodReturnType}} {{className}}(
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
                
                            return {{extensionMethodReturnValue}};
                        }
                    }
                """);

                typeResolver.Add($$"""
                "{{op.Name}}" => {{className}}.{{className}}FromProto(node, graph)
                """);
            }


            classes.AppendLine($$"""
            public static class OnnxNodeHelper
            {
                public static OnnxNode? TryFromProto(NodeProto node, OnnxGraph graph)
                {
                    return node.OpType switch
                    {
                        {{string.Join(",\n", typeResolver.Select(x => x.Trim()))}},
                        _ => null,
                    };
                }
            }
            """);

            var namespaceName = $"{nameof(Onnxify)}";

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
                var required = x.Option == FormalParameterOption.Optional ? " " : " required ";
                var nullable = x.Option == FormalParameterOption.Optional ? "?" : "";

                sb.AppendLine($$"""
                        /// <summary>
                        /// <para>
                        /// {{x.Name}}: {{(x.Description ?? "").Trim().Replace("\n", $"\n    /// ").Replace("<", "&lt;").Replace(">", "&gt;")}}
                        /// </para>
                        /// <para>
                        /// Allowed types: {{string.Join(", ", x.Types)}}
                        /// </para>
                        /// </summary>
                        public{{required}}{{MapType(x.Types)}}{{nullable}} {{onName(x.Name)}} { get; init; }
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
                var type = FromProto(typeEnum);

                sb.AppendLine($$"""
                        /// <summary>
                        /// <para>
                        /// {{x.Name}}: {{(x.Description ?? "").Trim().Replace("\n", $"\n    /// ").Replace("<", "&lt;").Replace(">", "&gt;")}}
                        /// </para>
                        /// <para>
                        /// Allowed types: {{string.Join(", ", [type])}}
                        /// </para>
                        /// </summary>
                        public{{required}}{{type}}{{nullable}} {{AttributeName(x.Name)}} { get; set; }
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

            if (_aliasFieldNames.TryGetValue(p, out var alias))
            {
                p = alias;
            }

            if (_reservedFieldNames.Contains(p))
            {
                prefix = "Input";
                return prefix + p;
            }

            if (!string.IsNullOrEmpty(prefix) && p.Equals(prefix, StringComparison.OrdinalIgnoreCase))
            {
                return prefix;
            }

            return prefix + p;
        }

        public static string OutputName(string name, string prefix = "")
        {
            var p = PascalCase(name);

            if (_aliasFieldNames.TryGetValue(p, out var alias))
            {
                p = alias;
            }

            if (_reservedFieldNames.Contains(p))
            {
                prefix = "Output";
                return prefix + p;
            }

            if (!string.IsNullOrEmpty(prefix) && p.Equals(prefix, StringComparison.OrdinalIgnoreCase))
            {
                return prefix;
            }

            return prefix + p;
        }

        public static string AttributeName(string name, string prefix = "")
        {
            var p = PascalCase(name);

            if (_aliasFieldNames.TryGetValue(p, out var alias))
            {
                p = alias;
            }

            if (_reservedFieldNames.Contains(p))
            {
                prefix = "Attribute";
                return prefix + p;
            }

            if (!string.IsNullOrEmpty(prefix) && p.Equals(prefix, StringComparison.OrdinalIgnoreCase))
            {
                return prefix;
            }

            return prefix + p;
        }

        public static string MapType(string[] types)
        {
            return "IOnnxGraphEdge";

            if (types.Length == 1)
            {
                return MapType(types[0]);
            }

            if (types.All(x => x.StartsWith("tensor")))
            {
                return "IOnnxGraphEdge"; // TODO: OnnxTensor??
            }

            throw new NotSupportedException($"Unsupported ONNX type: {string.Join(", ", types)}");
        }

        public static string MapType(string type)
        {
            return type switch
            {
                "tensor(uint32)" => "OnnxTensor<bool>",
                "tensor(float)" => "OnnxTensor<float>",
                "tensor(uint8)" => "OnnxTensor<byte>",
                "tensor(uint16)" => "OnnxTensor<ushort>",
                "tensor(int64)" => "OnnxTensor<long>",
                "tensor(uint64)" => "OnnxTensor<ulong>",
                "tensor(int16)" => "OnnxTensor<short>",
                "tensor(int8)" => "OnnxTensor<sbyte>",
                "tensor(int32)" => "OnnxTensor<int>",
                "tensor(bfloat16)" => "OnnxTensor<BFloat16>",
                "tensor(float16)" => "OnnxTensor<Half>",
                "tensor(double)" => "OnnxTensor<double>",
                "tensor(string)" => "OnnxTensor<string>",
                "tensor(bool)" => "OnnxTensor<bool>",
                "tensor(complex64)" => "OnnxTensor<Complex64>",
                "tensor(complex128)" => "OnnxTensor<Complex>",

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
