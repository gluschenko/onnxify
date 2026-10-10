using System.Globalization;
using Onnxify;
using Onnxify.Compiler;

namespace Onnxify.Compiler.Operators;

internal abstract class CompilerOperator
{
    protected CompilerOperator(CompilerOperatorIdentity identity)
    {
        Identity = identity;
    }

    public CompilerOperatorIdentity Identity { get; }
    public string OnnxName => Identity.Name;
    public string OnnxDomain => Identity.Domain;
    public CompilerOperatorDescriptor Descriptor => new(Identity.Name, Identity.Domain, Capability, Constraints);
    public IReadOnlyList<string> TorchSharpNames => TorchSharpForms.Select(static form => form.Name).ToArray();
    public abstract Type NodeType { get; }
    public abstract IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; }
    public virtual CompilerOperationCapability Capability => CompilerOperationCapability.Bidirectional;
    public virtual int MinimumInputCount => 1;
    public virtual int MaximumInputCount => MinimumInputCount;
    public virtual int InputCount => MinimumInputCount;
    public virtual string? BinaryOperator => TorchSharpForms.FirstOrDefault(static form => form.Kind == CompilerTorchSharpFormKind.Binary)?.Name;
    public virtual string? UnaryOperator => TorchSharpForms.FirstOrDefault(static form => form.Kind == CompilerTorchSharpFormKind.Unary)?.Name;
    public virtual string? TorchSharpMethod => null;
    public virtual IReadOnlyList<string> AttributeNames => Array.Empty<string>();
    public virtual IReadOnlyList<float> FixedTorchSharpArguments => Array.Empty<float>();
    public virtual bool SupportsMultidirectionalBroadcast => false;
    public virtual IReadOnlyList<CompilerElementType?> InputElementTypes => Array.Empty<CompilerElementType?>();
    public virtual CompilerElementType? OutputElementType => null;
    public virtual CompilerElementType? GetTorchSharpOutputElementType(CompilerInvocationExpression invocation)
    {
        return OutputElementType;
    }

    protected virtual bool AllowsEmptyOptionalInputs => false;
    public virtual IReadOnlyCollection<string> Constraints =>
    [
        MinimumInputCount == MaximumInputCount
            ? $"Inputs: {MinimumInputCount}; attributes: {string.Join(", ", AttributeNames)}"
            : $"Inputs: {MinimumInputCount}-{MaximumInputCount}; attributes: {string.Join(", ", AttributeNames)}",
        $"TorchSharp forms: {string.Join(", ", TorchSharpForms.Select(static form => form.Name))}",
    ];

    public abstract CompilerOnnxStep CreateNode(
        string name,
        IEnumerable<CompilerValueReference> inputs,
        IEnumerable<CompilerValueReference> outputs,
        IEnumerable<CompilerAttribute>? attributes = null,
        CompilerSourceSpan? span = null);

    public bool Accepts(CompilerOnnxStep operation)
    {
        return Accepts(operation.Node)
            && operation.Inputs.Count >= MinimumInputCount
            && operation.Inputs.Count <= MaximumInputCount
            && operation.Outputs.Count == 1
            && (AllowsEmptyOptionalInputs || operation.Inputs.All(static input => !input.IsEmptyOptional))
            && operation.Outputs.All(static output => !output.IsEmptyOptional)
            && operation.Node.Attributes.All(attribute => AttributeNames.Contains(attribute.Name, StringComparer.Ordinal));
    }

    public bool Accepts(OnnxNode node)
    {
        return (node.GetType() == NodeType || node.GetType() == typeof(OnnxNode))
            && string.Equals(node.Domain, OnnxDomain, StringComparison.Ordinal)
            && string.Equals(node.OpType, OnnxName, StringComparison.Ordinal)
            && node.Inputs.Count >= MinimumInputCount
            && node.Inputs.Count <= MaximumInputCount
            && node.Outputs.Count == 1
            && node.Attributes.All(attribute => AttributeNames.Contains(attribute.Name, StringComparer.Ordinal));
    }

    public abstract string PrintTorchSharp(CompilerOnnxStep operation);

    public virtual void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        graph.AddNode(operation.Node);
    }

    public virtual bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        return false;
    }

    public virtual CompilerOnnxStep ScanOnnx(OnnxNode node, string name, CompilerSourceSpan? span)
    {
        if (node.GetType() != NodeType)
        {
            throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                $"ONNX operator '{OnnxDomain}::{OnnxName}' requires generated node type '{NodeType.Name}', but received '{node.GetType().Name}'.");
        }

        return new CompilerOnnxStep(node, Descriptor, span, name);
    }

    public virtual string? ValidateOnnxNode(
        OnnxNode node,
        IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        return null;
    }

    protected static string? ValidateBroadcast(
        OnnxNode node,
        IReadOnlyDictionary<string, CompilerType> knownTypes
    )
    {
        var shapes = new List<long[]>();
        foreach (var input in node.Inputs)
        {
            if (!knownTypes.TryGetValue(input.Name, out var inputType)
                || inputType is not CompilerTensorType tensorType
                || tensorType.Dimensions is not { } inputDimensions
                || inputDimensions.Any(static dimension => dimension is not CompilerFixedDimension))
            {
                return null;
            }

            var dimensions = inputDimensions
                .OfType<CompilerFixedDimension>()
                .Select(static dimension => dimension.Value)
                .ToArray();
            shapes.Add(dimensions);
        }

        if (shapes.Count < 2)
        {
            return null;
        }

        var rank = shapes.Max(static shape => shape.Length);
        for (var offset = 1; offset <= rank; offset++)
        {
            var nonUnitDimensionCount = shapes
                .Select(shape => offset <= shape.Length ? shape[shape.Length - offset] : 1)
                .Where(static dimension => dimension != 1)
                .Distinct()
                .Take(2)
                .Count();
            if (nonUnitDimensionCount > 1)
            {
                return $"ONNX operator '{node.OpType}' has input shapes that cannot be broadcast together.";
            }
        }

        return null;
    }

    protected static string Float(float value)
    {
        return value.ToString("R", CultureInfo.InvariantCulture) + "f";
    }

    protected static CSharpCompilerDiagnosticException Unsupported(CompilerSourceSpan? span, string message)
    {
        return new(CompilerDiagnosticCodes.Unsupported, message, CompilerDiagnosticStage.Emit, span);
    }
}

internal abstract class CompilerOperator<TNode>(CompilerOperatorIdentity identity) : CompilerOperator(identity)
    where TNode : OnnxNode
{
    public sealed override Type NodeType => typeof(TNode);

    public sealed override CompilerOnnxStep CreateNode(
        string name,
        IEnumerable<CompilerValueReference> inputs,
        IEnumerable<CompilerValueReference> outputs,
        IEnumerable<CompilerAttribute>? attributes = null,
        CompilerSourceSpan? span = null)
    {
        var inputEdges = inputs
            .Select(static input => (IOnnxGraphEdge?)new OnnxEdge(input.IsEmptyOptional ? string.Empty : input.Name))
            .ToArray();
        var outputEdges = outputs
            .Select(static output => (IOnnxGraphEdge?)new OnnxEdge(output.IsEmptyOptional ? string.Empty : output.Name))
            .ToArray();
        var onnxAttributes = (attributes ?? Array.Empty<CompilerAttribute>())
            .Select(attribute => OnnxCompilerBackend.CreateOnnxAttribute(attribute, [], null))
            .ToArray();
        TNode node;
        try
        {
            node = CompilerTypedNodeFactory.Create<TNode>(name, inputEdges, outputEdges, onnxAttributes);
        }
        catch (Exception exception)
        {
            throw Unsupported(
                span,
                $"ONNX operator '{OnnxName}' cannot be constructed from the supplied inputs or attributes: {exception.Message}");
        }

        return new CompilerOnnxStep(node, Descriptor, span, name);
    }

    public sealed override string PrintTorchSharp(CompilerOnnxStep operation)
    {
        if (operation.Node is not TNode typedNode)
        {
            throw Unsupported(operation.Span, $"Operator '{OnnxName}' requires generated node type '{typeof(TNode).Name}'.");
        }

        return PrintTorchSharp(typedNode, operation.Span);
    }

    protected abstract string PrintTorchSharp(TNode node, CompilerSourceSpan? span);

    public sealed override CompilerOnnxStep ScanOnnx(OnnxNode node, string name, CompilerSourceSpan? span)
    {
        if (node.GetType() != typeof(OnnxNode) && node is not TNode)
        {
            throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                $"ONNX operator '{OnnxDomain}::{OnnxName}' requires generated node type '{typeof(TNode).Name}', but received '{node.GetType().Name}'.");
        }

        TNode typedNode;
        try
        {
            typedNode = CompilerTypedNodeFactory.Create<TNode>(name, node.Inputs, node.Outputs, node.Attributes);
        }
        catch (Exception exception)
        {
            throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                $"ONNX operator '{OnnxDomain}::{OnnxName}' cannot be represented by generated node type '{typeof(TNode).Name}': {exception.Message}");
        }

        return new CompilerOnnxStep(typedNode, Descriptor, span, name);
    }
}
