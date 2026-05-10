using System.Text;
using System.Text.Json;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Text;
using Onnxify.SourceGenerator.Models;

namespace Onnxify.SourceGenerator;

[Generator]
public class OnnxNodeGenerator : IIncrementalGenerator
{
    private readonly static Dictionary<string, string> _aliasFieldNames = new()
    {
        ["Inputs"] = "In",
        ["Outputs"] = "Out",
    };

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var operatorSchemaFiles = context.AdditionalTextsProvider
            .Where(static file => string.Equals(Path.GetFileName(file.Path), "onnx_operators.json", StringComparison.OrdinalIgnoreCase));

        context.RegisterSourceOutput(
            operatorSchemaFiles,
            (productionContext, file) =>
            {
                Generate(
                    context: productionContext,
                    file: file
                );
            }
        );
    }

    public void Generate(
        SourceProductionContext context,
        AdditionalText file
    )
    {
        var json = file.GetText(context.CancellationToken)?.ToString() ?? string.Empty;
        var root = JsonSerializer.Deserialize<OperatorSchemaRoot>(json) ?? throw new Exception();

        var typeResolver = new List<string>();

        var domains = root.Operators
            .GroupBy(x => x.Domain)
            .Select(x =>
            {
                return new
                {
                    Name = x.Key,
                    Operators = x.OrderBy(x => x.Name).ToArray(),
                };
            })
            .ToArray();

        var rootNamespace = $"{nameof(Onnxify)}";
        var graphType = $"{rootNamespace}.OnnxGraph";

        foreach (var domain in domains)
        {
            var domainClasses = new StringBuilder();

            var currentNamespace = domain.Name switch
            {
                "" => rootNamespace,
                "ai.onnx.ml" => $"{rootNamespace}.ML",
                "com.microsoft" => $"{rootNamespace}.Microsoft",
                "com.microsoft.nhwc" => $"{rootNamespace}.Microsoft.NHWC",
                "com.microsoft.nchwc" => $"{rootNamespace}.Microsoft.NCHWc",
                "com.ms.internal.nhwc" => $"{rootNamespace}.Microsoft.Internal.NHWC",
                _ => throw new NotImplementedException($"Not implemented for '{domain}'"),
            };

            var extensionAnchor = domain.Name switch
            {
                "" => graphType,
                "ai.onnx.ml" => $"{rootNamespace}.MLDomain",
                "com.microsoft" => $"{rootNamespace}.MicrosoftDomain",
                "com.microsoft.nhwc" => $"{rootNamespace}.MicrosoftNHWCDomain",
                "com.microsoft.nchwc" => $"{rootNamespace}.MicrosoftNCHWcDomain",
                "com.ms.internal.nhwc" => $"{rootNamespace}.MicrosoftInternalNHWCDomain",
                _ => throw new NotImplementedException($"Not implemented for '{domain}'"),
            };

            foreach (var op in domain.Operators)
            {
                var operatorClasses = new StringBuilder();
                var extensionMethods = new StringBuilder();

                var className = $"{op.Name}";
                var reservedFieldNames = CreateReservedFieldNames(op, className);

                var hasVariadicOutput = op.Outputs.Any(IsVariadic);
                var optionsAttributes = op.Attributes
                    .Select(x =>
                    {
                        var type = FromProto((AttributeType)x.Type);
                        return $"() => options.{AttributeName(x.Name, reservedFieldNames)} is {type} x ? new OnnxAttribute<{type}>(\"{x.Name}\", x) : null";
                    })
                    .ToArray();

                var optionsAttributesList = optionsAttributes.Length > 0
                    ? $$"""
                    TypeHelper.NotNull<OnnxAttribute>(
                        new List<Func<OnnxAttribute?>>() 
                        { 
                            {{string.Join(",\n", optionsAttributes).Indent(2)}}
                        }
                        .Select(x => x())
                        .ToArray()
                    )
                    """
                    : "[]";

                operatorClasses.AppendLine($$"""
                /// <summary>
                /// Collects the input wires and attributes used to construct a {{className}} node.
                /// </summary>
                /// <remarks>
                /// Use this options type when outputs are supplied by another construction path, such as loading an existing ONNX node.
                /// </remarks>
                public class {{className}}InputOptions
                {
                    {{GetFields(op.Inputs, x => InputName(x, reservedFieldNames)).Indent(1)}}
                
                    {{GetFields(op.Attributes, x => AttributeName(x, reservedFieldNames)).Indent(1)}}
                }
                
                /// <summary>
                /// Collects the complete input, output, and attribute wiring for a {{className}} node.
                /// </summary>
                /// <remarks>
                /// Properties follow ONNX schema order so optional and variadic parameters remain aligned with serialized node inputs and outputs.
                /// </remarks>
                public class {{className}}InputOutputOptions : {{className}}InputOptions
                {
                    {{GetFields(op.Outputs, x => OutputName(x, reservedFieldNames)).Indent(1)}}
                }
                """);

                var nodeFields = new StringBuilder();
                var createInputLines = new List<string>();
                var createOutputLines = new List<string>();

                for (var i = 0; i < op.Inputs.Count(); i++)
                {
                    var x = op.Inputs[i];

                    nodeFields.AppendLine(GetParameterComment(x));
                    nodeFields.AppendLine(GetAcceptTypeAttributes(x));

                    if (IsVariadic(x))
                    {
                        nodeFields.AppendLine($$"""
                        public {{GetNodeParameterType(x)}} {{InputName(x.Name, reservedFieldNames)}}
                        {
                            get => GetVariadicInputs({{i}});
                            set
                            {
                                ArgumentNullException.ThrowIfNull(value);
                                if (value.Count < {{x.MinArity}})
                                {
                                    throw new InvalidOperationException("Variadic input '{{InputName(x.Name, reservedFieldNames)}}' requires at least {{x.MinArity}} value(s).");
                                }

                                SetVariadicInputs({{i}}, value);
                            }
                        }
                        """);

                        createInputLines.Add($$"""
                        if (options.{{InputName(x.Name, reservedFieldNames)}}.Length < {{x.MinArity}})
                        {
                            throw new InvalidOperationException("Variadic input '{{InputName(x.Name, reservedFieldNames)}}' requires at least {{x.MinArity}} value(s).");
                        }

                        inputs.AddRange(options.{{InputName(x.Name, reservedFieldNames)}});
                        """);
                    }
                    else if (x.Option != FormalParameterOption.Optional)
                    {
                        nodeFields.AppendLine($$"""
                        public {{MapType(x.Types)}} {{InputName(x.Name, reservedFieldNames)}}
                        {
                            get => ({{MapType(x.Types)}})Inputs[{{i}}];
                            set => SetInput({{i}}, value);
                        }
                        """);

                        createInputLines.Add($"inputs.Add(options.{InputName(x.Name, reservedFieldNames)});");
                    }
                    else
                    {
                        nodeFields.AppendLine($$"""
                        public {{MapType(x.Types)}}? {{InputName(x.Name, reservedFieldNames)}}
                        {
                            get => Inputs.Count > {{i}} ? ({{MapType(x.Types)}})Inputs[{{i}}] : null;
                            set => SetOptionalInput({{i}}, value);
                        }
                        """);

                        createInputLines.Add($$"""
                        if (options.{{InputName(x.Name, reservedFieldNames)}} is not null)
                        {
                            inputs.Add(options.{{InputName(x.Name, reservedFieldNames)}});
                        }
                        """);
                    }
                }

                for (var i = 0; i < op.Outputs.Count(); i++)
                {
                    var x = op.Outputs[i];

                    nodeFields.AppendLine(GetParameterComment(x));
                    nodeFields.AppendLine(GetAcceptTypeAttributes(x));

                    if (IsVariadic(x))
                    {
                        nodeFields.AppendLine($$"""
                        public {{GetNodeParameterType(x)}} {{OutputName(x.Name, reservedFieldNames)}}
                        {
                            get => GetVariadicOutputs({{i}});
                            set
                            {
                                ArgumentNullException.ThrowIfNull(value);
                                if (value.Count < {{x.MinArity}})
                                {
                                    throw new InvalidOperationException("Variadic output '{{OutputName(x.Name, reservedFieldNames)}}' requires at least {{x.MinArity}} value(s).");
                                }

                                SetVariadicOutputs({{i}}, value);
                            }
                        }
                        """);

                        createOutputLines.Add($$"""
                        if (options.{{OutputName(x.Name, reservedFieldNames)}}.Length < {{x.MinArity}})
                        {
                            throw new InvalidOperationException("Variadic output '{{OutputName(x.Name, reservedFieldNames)}}' requires at least {{x.MinArity}} value(s).");
                        }

                        outputs.AddRange(options.{{OutputName(x.Name, reservedFieldNames)}});
                        """);
                    }
                    else if (x.Option != FormalParameterOption.Optional)
                    {
                        nodeFields.AppendLine($$"""
                        public {{MapType(x.Types)}} {{OutputName(x.Name, reservedFieldNames)}}
                        {
                            get => ({{MapType(x.Types)}})Outputs[{{i}}];
                            set => SetOutput({{i}}, value);
                        }
                        """);

                        createOutputLines.Add($"outputs.Add(options.{OutputName(x.Name, reservedFieldNames)});");
                    }
                    else
                    {
                        nodeFields.AppendLine($$"""
                        public {{MapType(x.Types)}}? {{OutputName(x.Name, reservedFieldNames)}}
                        {
                            get => Outputs.Count > {{i}} && Outputs[{{i}}] is {{MapType(x.Types)}} x ? x : null;
                            set => SetOptionalOutput({{i}}, value);
                        }
                        """);

                        createOutputLines.Add($$"""
                        if (options.{{OutputName(x.Name, reservedFieldNames)}} is not null)
                        {
                            outputs.Add(options.{{OutputName(x.Name, reservedFieldNames)}});
                        }
                        """);
                    }
                }

                for (var i = 0; i < op.Attributes.Count(); i++)
                {
                    var x = op.Attributes[i];

                    var type = FromProto((AttributeType)x.Type);

                    nodeFields.AppendLine(GetAttributeComment(x));

                    if (!x.IsNullable())
                    {
                        nodeFields.AppendLine($$"""
                        public {{type}} {{AttributeName(x.Name, reservedFieldNames)}}
                        {
                            get => GetAttribute<{{type}}>("{{x.Name}}");
                            set => SetAttribute<{{type}}>("{{x.Name}}", value);
                        }
                        """);
                    }
                    else
                    {
                        nodeFields.AppendLine($$"""
                        public {{type}}? {{AttributeName(x.Name, reservedFieldNames)}}
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

                list1.AddRange(op.Inputs.Select(x => $"{InputName(x.Name, reservedFieldNames)} = options.{InputName(x.Name, reservedFieldNames)}"));
                list1.AddRange(op.Attributes.Select(x => $"{AttributeName(x.Name, reservedFieldNames)} = options.{AttributeName(x.Name, reservedFieldNames)}"));
                list1.AddRange(op.Outputs.Where(x => !IsVariadic(x)).Select(x => $"{OutputName(x.Name, reservedFieldNames)} = edge{OutputName(x.Name, reservedFieldNames)}"));

                var list2 = new List<string>();

                list2.AddRange(op.Inputs.Select((x, i) =>
                {
                    if (IsVariadic(x))
                    {
                        return $"{InputName(x.Name, reservedFieldNames)} = inputs.Skip({i}).OfType<IOnnxGraphEdge>().ToArray()";
                    }

                    if (x.Option != FormalParameterOption.Optional)
                    {
                        return $"{InputName(x.Name, reservedFieldNames)} = inputs[{i}] ?? throw new InvalidOperationException(\"Missing required input '{x.Name}'\")";
                    }
                    else
                    {
                        return $"{InputName(x.Name, reservedFieldNames)} = inputs.Length > {i} ? inputs[{i}] : null";
                    }
                }));

                list2.AddRange(op.Attributes
                    .Where(x => x.Required || x.Default is null)
                    .Select(x =>
                    {
                        var type = FromProto((AttributeType)x.Type);

                        if (!x.IsNullable())
                        {
                            return $"{AttributeName(x.Name, reservedFieldNames)} = ({type})(attributes.GetValueOrDefault(\"{x.Name}\") ?? throw new InvalidOperationException($\"Missing value '{x.Name}'\"))";
                        }
                        else
                        {
                            return $"{AttributeName(x.Name, reservedFieldNames)} = ({type}?)attributes.GetValueOrDefault(\"{x.Name}\")";
                        }
                    }));

                list2.AddRange(op.Outputs.Select((x, i) =>
                {
                    if (IsVariadic(x))
                    {
                        return $"{OutputName(x.Name, reservedFieldNames)} = outputs.Skip({i}).OfType<IOnnxGraphEdge>().ToArray()";
                    }

                    if (x.Option != FormalParameterOption.Optional)
                    {
                        return $"{OutputName(x.Name, reservedFieldNames)} = outputs[{i}] ?? throw new InvalidOperationException(\"Missing required output '{x.Name}'\")";
                    }
                    else
                    {
                        return $"{OutputName(x.Name, reservedFieldNames)} = outputs.Length > {i} ? outputs[{i}] : null";
                    }
                }));

                operatorClasses.AppendLine($$"""
                {{GetOperatorComment(op)}}
                public class {{className}} : OnnxNode
                {
                    /// <summary>
                    /// Creates a {{className}} node from explicit ONNX graph wiring.
                    /// </summary>
                    /// <param name="name">Graph-local node name.</param>
                    /// <param name="options">Input, output, and attribute values arranged according to the ONNX {{op.Name}} schema.</param>
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
                    operatorClasses.AppendLine($$"""
                    /// <summary>
                    /// Groups the multiple output wires produced by a {{className}} helper call.
                    /// </summary>
                    public class {{className}}Output
                    {
                        {{GetResultFields(op.Outputs, x => OutputName(x, reservedFieldNames)).Indent(1)}}
                    }
                    """);
                }

                var extensionMethodReturnType = op.Outputs.Count switch
                {
                    0 => $"{currentNamespace}.{className}",
                    1 => GetNodeParameterType(op.Outputs[0]),
                    _ => $"{currentNamespace}.{className}Output",
                };

                var extensionMethodReturnValue = op.Outputs.Count switch
                {
                    0 => "op",
                    1 => string.Join(", ", op.Outputs.Select(x => $"op.{OutputName(x.Name, reservedFieldNames)}")),
                    _ => $$"""
                    new {{currentNamespace}}.{{className}}Output 
                    { 
                        {{string.Join(",\n", op.Outputs.Select(x => $"{OutputName(x.Name, reservedFieldNames)} = op.{OutputName(x.Name, reservedFieldNames)}")).Indent(1)}} 
                    }
                    """,
                };

                if (!hasVariadicOutput)
                {
                    var edgeCreators = op.Outputs
                        .Select(x => $"var edge{OutputName(x.Name, reservedFieldNames)} = graph.AddEdge(name + \"_output_{x.Name.ToLower()}\");")
                        .ToArray();

                    extensionMethods.AppendLine($$"""
                    {{GetOperatorComment(op)}}
                    /// <param name="domain">Graph or domain accessor that determines where the node is added.</param>
                    /// <param name="name">Graph-local node name and prefix for automatically created outputs.</param>
                    /// <param name="options">Input and attribute values arranged according to the ONNX {{op.Name}} schema.</param>
                    /// <returns>The created node when there are no outputs, the single output wire when there is one output, or an output grouping object when the operator has multiple outputs.</returns>
                    public static {{extensionMethodReturnType}} {{className}}(
                        this {{extensionAnchor}} domain,
                        string name,
                        {{currentNamespace}}.{{className}}InputOptions options
                    )
                    {
                        var graph = {{(extensionAnchor == graphType ? "domain" : "domain.Graph")}};
                            
                        {{string.Join("\n", edgeCreators).Indent(1)}}
                                
                        var op = new {{currentNamespace}}.{{className}}(
                            name: name,
                            options: new {{currentNamespace}}.{{className}}InputOutputOptions
                            {
                                {{string.Join(",\n", list1).Indent(3)}}
                            }
                        );
                    
                        graph.AddNode(op);
                    
                        return {{extensionMethodReturnValue.Indent(1)}};
                    }
                    """);
                }

                extensionMethods.AppendLine($$"""
                {{GetOperatorComment(op)}}
                /// <param name="domain">Graph or domain accessor that determines where the node is added.</param>
                /// <param name="name">Graph-local node name.</param>
                /// <param name="options">Complete input, output, and attribute values arranged according to the ONNX {{op.Name}} schema.</param>
                /// <returns>The created node when there are no outputs, the single output wire when there is one output, or an output grouping object when the operator has multiple outputs.</returns>
                public static {{extensionMethodReturnType}} {{className}}(
                    this {{extensionAnchor}} domain,
                    string name,
                    {{currentNamespace}}.{{className}}InputOutputOptions options
                )
                {
                    var graph = {{(extensionAnchor == graphType ? "domain" : "domain.Graph")}};

                    var op = new {{currentNamespace}}.{{className}}(
                        name: name,
                        options: options
                    );
                    
                    graph.AddNode(op);
                    
                    return {{extensionMethodReturnValue.Indent(1)}};
                }
                """);

                typeResolver.Add($$"""
                ("{{op.Name}}", "{{op.Domain}}") => {{currentNamespace}}.{{className}}.{{className}}FromProto(node, graph)
                """);

                domainClasses.AppendLine($$"""
                namespace {{currentNamespace}}
                {
                    {{operatorClasses.ToString().Indent(1)}}
                }

                namespace {{rootNamespace}}
                {
                    /// <summary>
                    /// Provides generated convenience methods for adding ONNX operator nodes to graphs and domain accessors.
                    /// </summary>
                    public static partial class OnnxifyExtensions
                    {
                        {{extensionMethods.ToString().Indent(2)}}
                    }
                }
                """);
            }

            var code = $$"""
            // <auto-generated/>
            #nullable enable
            
            using System;
            using System.Collections.Generic;
            using Onnx;
            using {{nameof(Onnxify)}}.Helpers;
            using {{nameof(Onnxify)}}.Data;
            using {{nameof(Onnxify)}}.Data.Numerics;

            {{domainClasses}}

            #nullable restore
            """;

            context.AddSource($"{currentNamespace}.g.cs", SourceText.From(code, Encoding.UTF8));
        }

        if (true)
        {
            var namespaceName = $"{nameof(Onnxify)}.Helpers";
            var classes = new StringBuilder();

            if (typeResolver.Count() > 0)
            {
                classes.AppendLine($$"""
                internal static class OnnxNodeHelper
                {
                    internal static OnnxNode? TryFromProto(NodeProto node, OnnxGraph graph)
                    {
                        return (node.OpType, node.Domain) switch
                        {
                            {{string.Join(",\n", typeResolver.Select(x => x.Trim())).Indent(3)}},
                            _ => null,
                        };
                    }
                }
                """);
            }

            var code = $$"""
            // <auto-generated/>
            #nullable enable
            
            using System;
            using System.Collections.Generic;
            using Onnx;
            using {{namespaceName}};
            using {{nameof(Onnxify)}}.Data;
            using {{nameof(Onnxify)}}.Data.Numerics;

            namespace {{namespaceName}}
            {
                {{classes.ToString().Indent(1)}}
            }

            #nullable restore
            """;

            context.AddSource($"{namespaceName}.g.cs", SourceText.From(code, Encoding.UTF8));
        }
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
            {{GetAcceptTypeAttributes(x)}}
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
            {{GetAcceptTypeAttributes(x)}}
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
            var nullable = x.IsNullable() ? "?" : "";
            var typeEnum = (AttributeType)x.Type;
            var type = FromProto(typeEnum);

            var initializer = GetLiteral(typeEnum, x.Default);

            sb.AppendLine($$"""
            {{GetAttributeComment(x)}}
            public{{required}}{{type}}{{nullable}} {{onName(x.Name)}} { get; init; }{{(initializer is not null ? $" = {initializer};" : "")}}
            """);
        }

        return sb.ToString().Trim();
    }

    private static string? GetLiteral(AttributeType type, object? value)
    {
        if (value is null)
        {
            return null;
        }

        if (type == AttributeType.Float)
        {
            return $"{value}f";
        }

        if (type == AttributeType.Floats)
        {
            if (value is JsonElement element && element.ValueKind == JsonValueKind.Array)
            {
                var values = element.EnumerateArray().Select(x => $"{x}f").ToArray();
                return $"[{string.Join(", ", values)}]";
            }
            else
            {
                throw new NotImplementedException($"Not implemented for '{value.GetType().Name}'");
            }
        }

        return JsonSerializer.Serialize(value);
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
            .Select(x => FormatDocType(MapType(x)))
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

    public static string GetAttributeComment()
    {
        var typeEnum = (AttributeType)x.Type;
        string[] types = [FromProto(typeEnum)];

        var allowedTypes = types
            .Select(x => FormatDocType(MapType(x)))
            .ToArray();

        var allowedTypeString = string.Join(", ", allowedTypes);

        return $$"""
        /// <summary>
        /// <b>{{x.Name}} (attribute):</b>
        /// 
        /// {{(x.Description ?? "").Comment()}}
        /// 
        /// <para>Allowed types: {{allowedTypeString}}</para>
        /// <para>Default: {{GetLiteral(typeEnum, x.Default) ?? "[null]"}}</para>
        /// </summary>
        """;
    }

    public static string GetAcceptTypeAttributes(OperatorParameter x)
    {
        var allowedTypes = x.Types
            .Select(x => MapType(x))
            .Select(x => $"[AcceptType<{x}>]")
            .ToArray();

        return string.Join("\n", allowedTypes);
    }

    public static string GetAcceptTypeAttributes(OperatorAttribute x)
    {
        var allowedTypes = x.Types
            .Select(x => MapType(x))
            .Select(x => $"[AcceptType<{x}>]")
            .ToArray();

        return string.Join("\n", allowedTypes);
    }

    private static string FormatDocType(string type)
    {
        var name = type.Replace("<", "&lt;").Replace(">", "&gt;");
        return $"<c>{name}</c>";
    }

    private static HashSet<string> CreateReservedFieldNames(OperatorSchema op, string className)
    {
        var reservedFieldNames = new HashSet<string>(StringComparer.Ordinal)
        {
            className,
            "Inputs",
            "Outputs"
        };

        if (op.Inputs.Any(x => x.Name == "shape") && op.Attributes.Any(x => x.Name == "shape"))
        {
            reservedFieldNames.Add("Shape");
        }

        return reservedFieldNames;
    }

    public static string GetFieldName(string name, string prefix)
    {
        return GetFieldName(name, prefix, reservedFieldNames: null);
    }

    private static string GetFieldName(string name, string prefix, ISet<string>? reservedFieldNames)
    {
        var newName = name.PascalCase();

        if (_aliasFieldNames.TryGetValue(newName, out var alias))
        {
            newName = alias;
        }

        if (reservedFieldNames?.Contains(newName) == true)
        {
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
        return InputName(name, reservedFieldNames: null);
    }

    private static string InputName(string name, ISet<string>? reservedFieldNames)
    {
        return GetFieldName(name, "Input", reservedFieldNames);
    }

    public static string OutputName(string name)
    {
        return OutputName(name, reservedFieldNames: null);
    }

    private static string OutputName(string name, ISet<string>? reservedFieldNames)
    {
        return GetFieldName(name, "Output", reservedFieldNames);
    }

    public static string AttributeName(string name)
    {
        return AttributeName(name, reservedFieldNames: null);
    }

    private static string AttributeName(string name, ISet<string>? reservedFieldNames)
    {
        return GetFieldName(name, "Attribute", reservedFieldNames);
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
    }

    public static string MapType(string type)
    {
        return type switch
        {
            "tensor(int8)" => "OnnxTensor<sbyte>",
            "tensor(uint8)" => "OnnxTensor<byte>",
            "tensor(int16)" => "OnnxTensor<short>",
            "tensor(uint16)" => "OnnxTensor<ushort>",
            "tensor(int32)" => "OnnxTensor<int>",
            "tensor(uint32)" => "OnnxTensor<uint>",
            "tensor(int64)" => "OnnxTensor<long>",
            "tensor(uint64)" => "OnnxTensor<ulong>",

            "tensor(float)" => "OnnxTensor<float>",
            "tensor(double)" => "OnnxTensor<double>",
            "tensor(float16)" => "OnnxTensor<Half>",
            "tensor(bfloat16)" => "OnnxTensor<BFloat16>",

            "tensor(float8e4m3fn)" => "OnnxTensor<Float8E4M3FN>",
            "tensor(float8e4m3fnuz)" => "OnnxTensor<Float8E4M3FNUZ>",
            "tensor(float8e5m2)" => "OnnxTensor<Float8E5M2>",
            "tensor(float8e5m2fnuz)" => "OnnxTensor<Float8E5M2FNUZ>",
            "tensor(float8e8m0)" => "OnnxTensor<Float8E8M0>",

            "tensor(float4e2m1)" => "OnnxTensor<Float4E2M1>",

            "tensor(uint4)" => "OnnxTensor<UInt4>",
            "tensor(int4)" => "OnnxTensor<Int4>",
            "tensor(uint2)" => "OnnxTensor<UInt2>",
            "tensor(int2)" => "OnnxTensor<Int2>",

            "tensor(bool)" => "OnnxTensor<bool>",
            "tensor(string)" => "OnnxTensor<string>",

            "tensor(complex64)" => "OnnxTensor<Complex64>",
            "tensor(complex128)" => "OnnxTensor<Complex128>",

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
