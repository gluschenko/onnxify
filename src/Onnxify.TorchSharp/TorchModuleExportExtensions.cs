using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using ICSharpCode.Decompiler;
using ICSharpCode.Decompiler.CSharp;
using ICSharpCode.Decompiler.CSharp.Syntax;
using ICSharpCode.Decompiler.Metadata;

namespace Onnxify.TorchSharp;

public static class TorchModuleExportExtensions
{
    public static OnnxModel Export(
        this TorchModule module,
        OnnxTensorType input,
        OnnxTensorType output,
        OnnxModelCreationOptions options
    )
    {
        ArgumentNullException.ThrowIfNull(module);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(options);

        var onnxModel = OnnxModel.Create(options);
        var graph = onnxModel.Graph;

        var inputName = graph.NextName("input");
        var outputName = graph.NextName("output");

        var inputEdge = graph.AddInput(inputName, input);
        graph.AddOutput(outputName, output);

        var assemblyPath = module.GetType().Assembly.Location;

        var decompiler = new CSharpDecompiler(
            fileName: assemblyPath,
            settings: new DecompilerSettings
            {
                ThrowOnAssemblyResolveErrors = false,
            }
        );

        var method = module.GetType().GetMethod("forward");

        if (method is null)
        {
            throw new InvalidOperationException(
                $"Could not find forward method on '{module.GetType().FullName}'."
            );
        }

        var metadataToken = MetadataTokenHelpers.EntityHandleOrNil(method.MetadataToken);

        var syntaxTree = decompiler.Decompile(metadataToken);
        var forward = syntaxTree
            .Descendants
            .OfType<MethodDeclaration>()
            .SingleOrDefault(x => string.Equals(x.Name, method.Name, StringComparison.Ordinal))
            ?? throw new InvalidOperationException(
                $"Could not find decompiled method '{method.Name}' on '{module.GetType().FullName}'."
            );

        var result = ExportForwardBody(module, graph, inputEdge, forward);
        var outputEdge = graph.AddEdge(outputName);

        graph.Identity(
            name: graph.NextName("output_identity"),
            options: new IdentityInputOutputOptions
            {
                Input = result,
                Output = outputEdge,
            }
        );

        return onnxModel;
    }

    private static IOnnxGraphEdge ExportForwardBody(
        TorchModule module,
        OnnxGraph graph,
        IOnnxGraphEdge input,
        MethodDeclaration forward
    )
    {
        var context = new ForwardExportContext(module, graph);
        context.Values["input"] = new ExportValue(input);

        foreach (var parameter in forward.Parameters)
        {
            context.Values.TryAdd(parameter.Name, new ExportValue(input));
        }

        IOnnxGraphEdge? result = null;
        foreach (var statement in forward.Body.Statements)
        {
            result = ExportStatement(context, statement) ?? result;
        }

        return result
            ?? throw new NotSupportedException(
                $"Method '{forward.Name}' did not return a supported ONNX graph edge."
            );
    }

    private static IOnnxGraphEdge? ExportStatement(
        ForwardExportContext context,
        Statement statement
    )
    {
        if (TryExportDeconstructionStatement(context, statement))
        {
            return null;
        }

        switch (statement)
        {
            case VariableDeclarationStatement variableDeclaration:
                ExportVariableDeclaration(context, variableDeclaration);
                return null;

            case ExpressionStatement { Expression: AssignmentExpression assignment }:
                AssignValue(context, assignment.Left, ExportExpression(context, assignment.Right));
                return null;

            case ReturnStatement returnStatement:
                return ExportExpression(context, returnStatement.Expression).GetRequiredEdge(statement);

            default:
                throw new NotSupportedException(
                    $"Unsupported forward statement '{statement.GetType().Name}': {statement}"
                );
        }
    }

    private static void ExportVariableDeclaration(
        ForwardExportContext context,
        VariableDeclarationStatement declaration
    )
    {
        foreach (var variable in declaration.Variables)
        {
            if (variable.Initializer.IsNull)
            {
                continue;
            }

            context.Values[variable.Name] = ExportExpression(context, variable.Initializer);
        }
    }

    private static bool TryExportDeconstructionStatement(
        ForwardExportContext context,
        Statement statement
    )
    {
        var text = statement.ToString().Trim();
        var match = Regex.Match(
            text,
            @"^(?:var\s+)?\((?<names>[^)]*)\)\s*=\s*(?<value>.*);$",
            RegexOptions.CultureInvariant
        );

        if (!match.Success)
        {
            return false;
        }

        var invocation = statement.Descendants.OfType<InvocationExpression>().SingleOrDefault()
            ?? throw new NotSupportedException($"Unsupported deconstruction initializer: {statement}");

        var value = ExportExpression(context, invocation);
        var names = match.Groups["names"].Value
            .Split(',')
            .Select(static x => x.Trim())
            .ToArray();

        for (var index = 0; index < names.Length; index++)
        {
            if (string.IsNullOrWhiteSpace(names[index]) || names[index] == "_")
            {
                continue;
            }

            context.Values[names[index]] = GetTupleElement(value, index, statement);
        }

        return true;
    }

    private static ExportValue ExportExpression(
        ForwardExportContext context,
        Expression expression
    )
    {
        switch (expression)
        {
            case IdentifierExpression identifier:
                return context.Values.TryGetValue(identifier.Identifier, out var value)
                    ? value
                    : throw new NotSupportedException($"Unknown forward value '{identifier.Identifier}'.");

            case InvocationExpression invocation:
                return ExportInvocation(context, invocation);

            case MemberReferenceExpression memberReference:
                return ExportMemberReference(context, memberReference);

            case ParenthesizedExpression parenthesized:
                return ExportExpression(context, parenthesized.Expression);

            case CastExpression cast:
                return ExportExpression(context, cast.Expression);

            case PrimitiveExpression primitive:
                return new ExportValue(primitive.Value);

            default:
                throw new NotSupportedException(
                    $"Unsupported forward expression '{expression.GetType().Name}': {expression}"
                );
        }
    }

    private static ExportValue ExportInvocation(
        ForwardExportContext context,
        InvocationExpression invocation
    )
    {
        if (invocation.Target is not MemberReferenceExpression memberReference)
        {
            throw new NotSupportedException($"Unsupported invocation target: {invocation}");
        }

        if (string.Equals(memberReference.MemberName, "forward", StringComparison.Ordinal))
        {
            return ExportModuleForwardCall(context, memberReference, invocation);
        }

        if (string.Equals(memberReference.MemberName, "sum", StringComparison.Ordinal)
            && IsTorchReference(memberReference.Target))
        {
            return ExportTorchSum(context, invocation);
        }

        throw new NotSupportedException($"Unsupported forward invocation: {invocation}");
    }

    private static ExportValue ExportModuleForwardCall(
        ForwardExportContext context,
        MemberReferenceExpression target,
        InvocationExpression invocation
    )
    {
        if (invocation.Arguments.Count < 1
            || invocation.Arguments.Skip(1).Any(static x => x is not NullReferenceExpression))
        {
            throw new NotSupportedException(
                $"Only module.forward(input) and module.forward(input, null, ...) calls are supported by deep export: {invocation}"
            );
        }

        var torchModule = ResolveTorchModule(context.RootModule, target.Target);
        var input = ExportExpression(context, invocation.Arguments.First()).GetRequiredEdge(invocation);
        var output = InvokeModuleExport(torchModule, context.Graph, input);
        return new ExportValue(output);
    }

    private static ExportValue ExportTorchSum(
        ForwardExportContext context,
        InvocationExpression invocation
    )
    {
        if (invocation.Arguments.Count < 1 || invocation.Arguments.Count > 3)
        {
            throw new NotSupportedException($"Unsupported torch.sum argument count: {invocation}");
        }

        var data = ExportExpression(context, invocation.Arguments.ElementAt(0)).GetRequiredEdge(invocation);
        var axis = invocation.Arguments.Count >= 2
            ? Convert.ToInt64(ExportExpression(context, invocation.Arguments.ElementAt(1)).Value)
            : throw new NotSupportedException("Deep export requires torch.sum dimension to be specified.");

        var keepdims = invocation.Arguments.Count >= 3
            && Convert.ToBoolean(ExportExpression(context, invocation.Arguments.ElementAt(2)).Value);

        var name = context.Graph.NextName("sum");
        var output = context.Graph.ReduceSum(
            name: name,
            options: new ReduceSumInputOptions
            {
                Data = data,
                Axes = context.Graph.AddTensor<long>($"{name}_axes", [1], [axis]),
                Keepdims = keepdims ? 1 : 0,
            }
        );

        return new ExportValue(output);
    }

    private static ExportValue ExportMemberReference(
        ForwardExportContext context,
        MemberReferenceExpression memberReference
    )
    {
        var target = ExportExpression(context, memberReference.Target);
        var value = target.Value
            ?? throw new NotSupportedException($"Cannot access member '{memberReference.MemberName}' on an ONNX edge.");

        if (TryGetTupleItemIndex(memberReference.MemberName, out var itemIndex))
        {
            return GetTupleElement(new ExportValue(value), itemIndex, memberReference);
        }

        return new ExportValue(GetRequiredMemberValue(value, memberReference.MemberName));
    }

    private static void AssignValue(
        ForwardExportContext context,
        Expression target,
        ExportValue value
    )
    {
        if (target is IdentifierExpression identifier)
        {
            context.Values[identifier.Identifier] = value;
            return;
        }

        throw new NotSupportedException($"Unsupported assignment target: {target}");
    }

    private static object ResolveTorchModule(
        TorchModule root,
        Expression expression
    )
    {
        return ResolveMemberExpression(root, expression);
    }

    private static object ResolveMemberExpression(
        object root,
        Expression expression
    )
    {
        return expression switch
        {
            IdentifierExpression identifier => GetRequiredMemberValue(root, identifier.Identifier),
            MemberReferenceExpression { Target: ThisReferenceExpression, MemberName: var memberName } =>
                GetRequiredMemberValue(root, memberName),
            MemberReferenceExpression memberReference =>
                GetRequiredMemberValue(ResolveMemberExpression(root, memberReference.Target), memberReference.MemberName),
            ThisReferenceExpression => root,
            _ => throw new NotSupportedException($"Unsupported module reference: {expression}"),
        };
    }

    private static object InvokeModuleExport(
        object module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        const BindingFlags FLAGS = BindingFlags.Public | BindingFlags.Static;

        var exportMethod = typeof(TorchModuleExtensions)
            .GetMethods(FLAGS)
            .Where(static x => string.Equals(x.Name, "Export", StringComparison.Ordinal))
            .Select(x => new
            {
                Method = x,
                Parameters = x.GetParameters(),
            })
            .Where(x => x.Parameters.Length == 3
                && x.Parameters[1].ParameterType == typeof(OnnxGraph)
                && x.Parameters[2].ParameterType == typeof(IOnnxGraphEdge)
                && x.Parameters[0].ParameterType.IsAssignableFrom(module.GetType()))
            .OrderByDescending(x => GetInheritanceDistance(module.GetType(), x.Parameters[0].ParameterType))
            .Select(x => x.Method)
            .FirstOrDefault()
            ?? throw new NotSupportedException(
                $"No TorchModuleExtensions.Export overload was found for '{module.GetType().FullName}'."
            );

        return exportMethod.Invoke(null, [module, graph, input])
            ?? throw new InvalidOperationException(
                $"Export overload '{exportMethod}' returned null for '{module.GetType().FullName}'."
            );
    }

    private static int GetInheritanceDistance(Type type, Type candidate)
    {
        if (type == candidate)
        {
            return int.MaxValue;
        }

        var distance = 0;
        var current = type;
        while (current is not null)
        {
            if (current == candidate)
            {
                return int.MaxValue - distance;
            }

            current = current.BaseType;
            distance++;
        }

        return 0;
    }

    private static ExportValue GetTupleElement(
        ExportValue value,
        int index,
        AstNode source
    )
    {
        if (value.Value is null)
        {
            throw new NotSupportedException($"Cannot deconstruct ONNX edge result from: {source}");
        }

        var memberName = index switch
        {
            0 => "Y",
            1 => "YH",
            2 => "YC",
            _ => $"Item{index + 1}",
        };

        if (TryGetMemberValue(value.Value, memberName, out var memberValue))
        {
            return new ExportValue(memberValue);
        }

        if (value.Value is ITuple tuple && index < tuple.Length)
        {
            return new ExportValue(tuple[index]);
        }

        throw new NotSupportedException($"Cannot read deconstruction item {index} from '{value.Value.GetType().FullName}'.");
    }

    private static bool TryGetTupleItemIndex(string memberName, out int index)
    {
        if (memberName.Length > 4
            && memberName.StartsWith("Item", StringComparison.Ordinal)
            && int.TryParse(memberName[4..], out var oneBasedIndex)
            && oneBasedIndex > 0)
        {
            index = oneBasedIndex - 1;
            return true;
        }

        index = -1;
        return false;
    }

    private static object GetRequiredMemberValue(object instance, string name)
    {
        if (TryGetMemberValue(instance, name, out var value))
        {
            return value!;
        }

        throw new NotSupportedException(
            $"Member '{name}' was not found on '{instance.GetType().FullName}'."
        );
    }

    private static bool TryGetMemberValue(object instance, string name, out object? value)
    {
        const BindingFlags FLAGS = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

        var type = instance.GetType();
        var property = type.GetProperty(name, FLAGS);
        if (property is not null)
        {
            value = property.GetValue(instance);
            return true;
        }

        var field = type.GetField(name, FLAGS);
        if (field is not null)
        {
            value = field.GetValue(instance);
            return true;
        }

        value = null;
        return false;
    }

    private static bool IsTorchReference(Expression expression)
    {
        return string.Equals(expression.ToString(), "torch", StringComparison.Ordinal)
            || expression.ToString().EndsWith(".torch", StringComparison.Ordinal);
    }

    private sealed class ForwardExportContext(
        TorchModule rootModule,
        OnnxGraph graph
    )
    {
        public TorchModule RootModule { get; } = rootModule;

        public OnnxGraph Graph { get; } = graph;

        public Dictionary<string, ExportValue> Values { get; } = new(StringComparer.Ordinal);
    }

    private readonly record struct ExportValue(object? Value)
    {
        public IOnnxGraphEdge GetRequiredEdge(AstNode source)
        {
            return Value as IOnnxGraphEdge
                ?? throw new NotSupportedException(
                    $"Expression '{source}' did not produce an ONNX graph edge."
                );
        }
    }
}
