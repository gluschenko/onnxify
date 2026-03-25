using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
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

                var className = $"{op.Name}";

                _reservedFieldNames.Clear();
                _reservedFieldNames.Add(className);
                _reservedFieldNames.Add("Inputs");
                _reservedFieldNames.Add("Outputs");

                var hasVariadicOutput = op.Outputs.Any(IsVariadic);
                var optionsAttributes = op.Attributes
                    .Select(x =>
                    {
                        var type = FromProto((AttributeType)x.Type);
                        return $"() => options.{AttributeName(x.Name)} is {type} x ? new OnnxAttribute<{type}>(\"{x.Name}\", x) : null";
                    })
                    .ToArray();

                var optionsAttributesList = optionsAttributes.Length > 0 
                    ? $$"""
                    OnnxHelper.NotNull<OnnxAttribute>(
                        new List<Func<OnnxAttribute?>>() 
                        { 
                            {{string.Join(",\n", optionsAttributes).Indent(2)}}
                        }
                        .Select(x => x())
                        .ToArray()
                    )
                    """ 
                    : "[]";

                classes.AppendLine($$"""
                public class {{className}}InputOptions
                {
                    {{GetFields(op.Inputs, x => InputName(x)).Indent(1)}}
                
                    {{GetFields(op.Attributes, x => AttributeName(x)).Indent(1)}}
                }
                
                public class {{className}}InputOutputOptions : {{className}}InputOptions
                {
                    {{GetFields(op.Outputs, x => OutputName(x)).Indent(1)}}
                }
                """);

                var nodeFields = new StringBuilder();
                var createInputLines = new List<string>();
                var createOutputLines = new List<string>();

                for (var i = 0; i < op.Inputs.Count(); i++)
                {
                    var x = op.Inputs[i];

                    nodeFields.AppendLine(GetParameterComment(x));

                    if (IsVariadic(x))
                    {
                        nodeFields.AppendLine($$"""
                        public {{GetNodeParameterType(x)}} {{InputName(x.Name)}}
                        {
                            get => GetVariadicInputs({{i}});
                            set
                            {
                                ArgumentNullException.ThrowIfNull(value);
                                if (value.Count < {{x.MinArity}})
                                {
                                    throw new InvalidOperationException("Variadic input '{{InputName(x.Name)}}' requires at least {{x.MinArity}} value(s).");
                                }

                                SetVariadicInputs({{i}}, value);
                            }
                        }
                        """);

                        createInputLines.Add($$"""
                        if (options.{{InputName(x.Name)}}.Length < {{x.MinArity}})
                        {
                            throw new InvalidOperationException("Variadic input '{{InputName(x.Name)}}' requires at least {{x.MinArity}} value(s).");
                        }

                        inputs.AddRange(options.{{InputName(x.Name)}});
                        """);
                    }
                    else if (x.Option != FormalParameterOption.Optional)
                    {
                        nodeFields.AppendLine($$"""
                        public {{MapType(x.Types)}} {{InputName(x.Name)}}
                        {
                            get => ({{MapType(x.Types)}})Inputs[{{i}}];
                            set => SetInput({{i}}, value);
                        }
                        """);

                        createInputLines.Add($"inputs.Add(options.{InputName(x.Name)});");
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

                        createInputLines.Add($$"""
                        if (options.{{InputName(x.Name)}} is not null)
                        {
                            inputs.Add(options.{{InputName(x.Name)}});
                        }
                        """);
                    }
                }

                for (var i = 0; i < op.Outputs.Count(); i++)
                {
                    var x = op.Outputs[i];

                    nodeFields.AppendLine(GetParameterComment(x));

                    if (IsVariadic(x))
                    {
                        nodeFields.AppendLine($$"""
                        public {{GetNodeParameterType(x)}} {{OutputName(x.Name)}}
                        {
                            get => GetVariadicOutputs({{i}});
                            set
                            {
                                ArgumentNullException.ThrowIfNull(value);
                                if (value.Count < {{x.MinArity}})
                                {
                                    throw new InvalidOperationException("Variadic output '{{OutputName(x.Name)}}' requires at least {{x.MinArity}} value(s).");
                                }

                                SetVariadicOutputs({{i}}, value);
                            }
                        }
                        """);

                        createOutputLines.Add($$"""
                        if (options.{{OutputName(x.Name)}}.Length < {{x.MinArity}})
                        {
                            throw new InvalidOperationException("Variadic output '{{OutputName(x.Name)}}' requires at least {{x.MinArity}} value(s).");
                        }

                        outputs.AddRange(options.{{OutputName(x.Name)}});
                        """);
                    }
                    else if (x.Option != FormalParameterOption.Optional)
                    {
                        nodeFields.AppendLine($$"""
                        public {{MapType(x.Types)}} {{OutputName(x.Name)}}
                        {
                            get => ({{MapType(x.Types)}})Outputs[{{i}}];
                            set => SetOutput({{i}}, value);
                        }
                        """);

                        createOutputLines.Add($"outputs.Add(options.{OutputName(x.Name)});");
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

                        createOutputLines.Add($$"""
                        if (options.{{OutputName(x.Name)}} is not null)
                        {
                            outputs.Add(options.{{OutputName(x.Name)}});
                        }
                        """);
                    }
                }

                for (var i = 0; i < op.Attributes.Count(); i++)
                {
                    var x = op.Attributes[i];

                    var type = FromProto((AttributeType)x.Type);

                    nodeFields.AppendLine(GetAttributeComment(x));

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

                var list1 = new List<string>();

                list1.AddRange(op.Inputs.Select(x => $"{InputName(x.Name)} = options.{InputName(x.Name)}"));
                list1.AddRange(op.Attributes.Select(x => $"{AttributeName(x.Name)} = options.{AttributeName(x.Name)}"));
                list1.AddRange(op.Outputs.Where(x => !IsVariadic(x)).Select(x => $"{OutputName(x.Name)} = edge{OutputName(x.Name)}"));

                var list2 = new List<string>();

                list2.AddRange(op.Inputs.Select((x, i) =>
                {
                    if (IsVariadic(x))
                    {
                        return $"{InputName(x.Name)} = inputs.Skip({i}).OfType<IOnnxGraphEdge>().ToArray()";
                    }

                    if (x.Option != FormalParameterOption.Optional)
                    {
                        return $"{InputName(x.Name)} = inputs[{i}] ?? throw new InvalidOperationException(\"Missing required input '{x.Name}'\")";
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
                    if (IsVariadic(x))
                    {
                        return $"{OutputName(x.Name)} = outputs.Skip({i}).OfType<IOnnxGraphEdge>().ToArray()";
                    }

                    if (x.Option != FormalParameterOption.Optional)
                    {
                        return $"{OutputName(x.Name)} = outputs[{i}] ?? throw new InvalidOperationException(\"Missing required output '{x.Name}'\")";
                    }
                    else
                    {
                        return $"{OutputName(x.Name)} = outputs.Length > {i} ? outputs[{i}] : null";
                    }
                }));

                classes.AppendLine($$"""
                {{GetOperatorComment(op)}}
                public class {{className}} : OnnxNode
                {
                    public {{className}}(
                        string name,
                        {{className}}InputOutputOptions options
                    ) : base(
                        name: name,
                        opType: "{{op.Name}}",
                        domain: "{{op.Domain}}",
                        docString: "",
                        inputs: CreateInputs(options),
                        outputs: CreateOutputs(options),
                        attributes: CreateAttributes(options)
                    ) { }

                    {{nodeFields.ToString().Indent(1)}}

                    internal static IOnnxGraphEdge[] CreateInputs({{className}}InputOutputOptions options)
                    {
                        var inputs = new List<IOnnxGraphEdge>();
                        {{string.Join("\n\n", createInputLines).Indent(2)}}
                        return [.. inputs];
                    }

                    internal static IOnnxGraphEdge[] CreateOutputs({{className}}InputOutputOptions options)
                    {
                        var outputs = new List<IOnnxGraphEdge>();
                        {{string.Join("\n\n", createOutputLines).Indent(2)}}
                        return [.. outputs];
                    }

                    internal static OnnxAttribute[] CreateAttributes({{className}}InputOutputOptions options)
                    {
                        return {{optionsAttributesList.Indent(2)}};
                    }

                    internal static {{className}} {{className}}FromProto(NodeProto node, OnnxGraph graph)
                    {
                        var options = graph.GetOptions();
                        
                        if (node.OpType != "{{op.Name}}")
                        {
                            throw new InvalidOperationException($"Node type is not valid '{node.OpType}' != '{{op.Name}}'");
                        }
                            
                        var inputs = node.Input
                            .Select(x => string.IsNullOrEmpty(x) ? null : graph.GetValue(x) ?? throw new InvalidOperationException($"Missing value '{x}'"))
                            .ToArray();

                        var outputs = node.Output
                            .Select(x => string.IsNullOrEmpty(x) ? null : graph.GetValue(x) ?? throw new InvalidOperationException($"Missing value '{x}'"))
                            .ToArray();

                        var attributes = node.Attribute.ToDictionary(x => x.Name, x => x.GetValue(options));

                        var op = new {{className}}(
                            name: node.Name,
                            options: new {{className}}InputOutputOptions
                            {
                                {{string.Join(",\n", list2).Indent(4)}},
                            }
                        );

                        return op;
                    }
                }
                """);

                if (op.Outputs.Count() > 1)
                {
                    classes.AppendLine($$"""
                    public class {{className}}Output
                    {
                        {{GetResultFields(op.Outputs, x => OutputName(x)).Indent(1)}}
                    }
                    """);
                }

                var extensionMethodReturnType = op.Outputs.Count switch
                {
                    0 => className,
                    1 => GetNodeParameterType(op.Outputs[0]),
                    _ => $"{className}Output",
                };

                var extensionMethodReturnValue = op.Outputs.Count switch
                {
                    0 => "op",
                    1 => string.Join(", ", op.Outputs.Select(x => $"op.{OutputName(x.Name)}")),
                    _ => $$"""
                    new {{className}}Output 
                    { 
                        {{string.Join(",\n", op.Outputs.Select(x => $"{OutputName(x.Name)} = op.{OutputName(x.Name)}")).Indent(1)}} 
                    }
                    """,
                };

                if (!hasVariadicOutput)
                {
                    classes.AppendLine($$"""
                    public static partial class {{className}}Extensions
                    {
                        {{GetOperatorComment(op).Indent(1)}}
                        public static {{extensionMethodReturnType}} {{className}}(
                            this OnnxGraph graph,
                            string name,
                            {{className}}InputOptions options
                        )
                        {
                            {{string.Join("\n", (
                                op.Outputs.Select(x => (
                                    $"var edge{OutputName(x.Name)} = graph.AddEdge(name + \"_output_{x.Name.ToLower()}\");"
                                )))
                            ).Indent(2)}}
                                
                            var op = new {{className}}(
                                name: name,
                                options: new {{className}}InputOutputOptions
                                {
                                    {{string.Join(",\n", list1).Indent(4)}}
                                }
                            );
                    
                            graph.AddNode(op);
                    
                            return {{extensionMethodReturnValue.Indent(2)}};
                        }
                    }
                    """);
                }

                classes.AppendLine($$"""
                public static partial class {{className}}Extensions
                {
                    {{GetOperatorComment(op).Indent(1)}}
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
                
                        return {{extensionMethodReturnValue.Indent(2)}};
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
                        {{string.Join(",\n", typeResolver.Select(x => x.Trim())).Indent(3)}},
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
                {{classes.ToString().Indent(1)}}
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
                var type = GetOptionsParameterType(x);

                sb.AppendLine($$"""
                {{GetParameterComment(x)}}
                public{{required}}{{type}} {{onName(x.Name)}} { get; init; }
                """);
            }

            return sb.ToString().Trim();
        }

        private string GetResultFields(IEnumerable<OperatorParameter> items, Func<string, string> onName)
        {
            var sb = new StringBuilder();

            foreach (var x in items)
            {
                var type = GetNodeParameterType(x);

                sb.AppendLine($$"""
                {{GetParameterComment(x)}}
                public required {{type}} {{onName(x.Name)}} { get; init; }
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
                {{GetAttributeComment(x)}}
                public{{required}}{{type}}{{nullable}} {{AttributeName(x.Name)}} { get; init; }{{(x.Default is not null ? $" = ({type}){JsonSerializer.Serialize(x.Default)};" : "")}}
                """);
            }

            return sb.ToString().Trim();
        }

        public static string GetOperatorComment(OperatorSchema op)
        {
            return $$"""
            /// <summary>
            /// <b>{{op.Name}} (operator):</b>
            /// 
            /// <para>Domain: {{op.Domain ?? "ai.onnx"}}</para>
            /// <para>Since version: {{op.SinceVersion}}</para>
            /// 
            /// {{(op.Doc ?? "").Comment()}}
            /// </summary>
            """;
        }

        public static string GetParameterComment(OperatorParameter x)
        {
            var allowedTypes = x.Types
                .Select(x =>
                {
                    var type = MapType(x).Replace("<", "{").Replace(">", "}");
                    var name = MapType(x).Replace("<", "&lt;").Replace(">", "&gt;");

                    return $"<see cref=\"{type}\">{name}</see>";
                })
                .ToArray();

            var allowedTypeString = string.Join(", ", allowedTypes);

            return $$"""
            /// <summary>
            /// <b>{{x.Name}} (parameter):</b>
            /// 
            /// {{(x.Description ?? "").Comment()}}
            /// 
            /// <para>Allowed types: {{allowedTypeString}}</para>
            /// <para>Type: {{x.Option}}</para>
            /// </summary>
            """;
        }

        public static string GetAttributeComment(OperatorAttribute x)
        {
            var typeEnum = (AttributeType)x.Type;
            string[] types = [FromProto(typeEnum)];

            var allowedTypes = types
                .Select(x =>
                {
                    var type = MapType(x).Replace("<", "{").Replace(">", "}");
                    var name = MapType(x).Replace("<", "&lt;").Replace(">", "&gt;");

                    return $"<see cref=\"{type}\">{name}</see>";
                })
                .ToArray();

            var allowedTypeString = string.Join(", ", allowedTypes);

            return $$"""
            /// <summary>
            /// <b>{{x.Name}} (attribute):</b>
            /// 
            /// {{(x.Description ?? "").Comment()}}
            /// 
            /// <para>Allowed types: {{allowedTypeString}}</para>
            /// <para>Default: {{(x.Default is not null ? JsonSerializer.Serialize(x.Default) : "[null]")}}</para>
            /// </summary>
            """;
        }

        public static string GetFieldName(string name, string prefix = "")
        {
            var newName = name.PascalCase();

            if (_aliasFieldNames.TryGetValue(newName, out var alias))
            {
                newName = alias;
            }

            if (_reservedFieldNames.Contains(newName))
            {
                prefix = "Output";
                return prefix + newName;
            }

            if (!string.IsNullOrEmpty(prefix) && newName.Equals(prefix, StringComparison.OrdinalIgnoreCase))
            {
                return prefix;
            }

            return newName;
        }

        public static string InputName(string name)
        {
            return GetFieldName(name, "Input");
        }

        public static string OutputName(string name)
        {
            return GetFieldName(name, "Output");
        }

        public static string AttributeName(string name)
        {
            return GetFieldName(name, "Attribute");
        }

        public static bool IsVariadic(OperatorParameter parameter)
        {
            return parameter.Option == FormalParameterOption.Variadic;
        }

        public static string GetOptionsParameterType(OperatorParameter parameter)
        {
            var type = MapType(parameter.Types);

            return parameter.Option switch
            {
                FormalParameterOption.Optional => $"{type}?",
                FormalParameterOption.Variadic => $"{type}[]",
                _ => type,
            };
        }

        public static string GetNodeParameterType(OperatorParameter parameter)
        {
            var type = MapType(parameter.Types);

            return parameter.Option switch
            {
                FormalParameterOption.Optional => $"{type}?",
                FormalParameterOption.Variadic => $"IReadOnlyList<{type}>",
                _ => type,
            };
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

                _ => type, // throw new NotSupportedException($"Unsupported ONNX type: {type}")
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
                AttributeType.SparseTensor => "OnnxSparseTensor",

                AttributeType.Floats => "float[]",
                AttributeType.Ints => "long[]",
                AttributeType.Strings => "string[]",

                AttributeType.Tensors => "OnnxTensor[]",
                AttributeType.Graphs => "OnnxGraph[]",
                AttributeType.SparseTensors => "OnnxSparseTensor[]",

                AttributeType.TypeProto => "OnnxValueType",
                AttributeType.TypeProtos => "OnnxValueType[]",

                _ => throw new NotImplementedException($"Not implemented for '{attributeType}'"),
            };
        }
    }

    public static class TextHelper
    {
        public static string PascalCase(this string s)
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

        public static string Comment(this string text)
        {
            return text.Trim().ToXmlDoc().Replace("\n", $"\n/// ").Trim();
        }

        public static string Indent(this string text, int tabs)
        {
            var indent = new string(' ', tabs * 4);
            return text.Trim().Replace("\n", $"\n{indent}").Trim();
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
    TypeProto = 13,
    TypeProtos = 14,
}

public static class MarkdownHelper
{
    private static readonly Regex _codeBlockRegex = new(
        @"```(?:\w+)?\s*([\s\S]*?)```",
        RegexOptions.Compiled
    );

    private static readonly Regex _inlineCodeRegex = new(
        @"`([^`\n]+?)`",
        RegexOptions.Compiled
    );

    public static string ToXmlDoc(this string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return string.Empty;
        }

        var blocks = new List<string>();

        text = _codeBlockRegex.Replace(text, m =>
        {
            var content = Escape(m.Groups[1].Value.Trim());
            blocks.Add($"<code>{content}</code>");
            return $"@@CODEBLOCK_{blocks.Count - 1}@@";
        });

        text = _inlineCodeRegex.Replace(text, m =>
        {
            var content = Escape(m.Groups[1].Value);
            blocks.Add($"<c>{content}</c>");
            return $"@@CODEBLOCK_{blocks.Count - 1}@@";
        });

        text = Escape(text);

        for (var i = 0; i < blocks.Count; i++)
        {
            text = text.Replace($"@@CODEBLOCK_{i}@@", blocks[i]);
        }

        // 5. Параграфы
        var paragraphs = text
            .Split(["\r\n\r\n", "\n\n"], StringSplitOptions.RemoveEmptyEntries)
            .ToArray();

        return string.Join("\n", paragraphs.Length > 1 ? paragraphs.Select(x => $"<para>\n{x.Trim()}\n</para>") : paragraphs);
    }

    private static string Escape(string text)
    {
        return text
            .Replace("&", "&amp;")
            .Replace("<", "&lt;")
            .Replace(">", "&gt;");
    }
}
