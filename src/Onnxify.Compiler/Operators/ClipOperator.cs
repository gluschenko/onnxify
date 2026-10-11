using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::clamp")]
[CompilerTorchOperator("aten::clamp.Tensor")]
internal sealed class ClipOperator() : CompilerOperator<Onnxify.Clip>(CompilerOperatorIdentity.Onnx("Clip"))
{
    public override CompilerOperationCapability Capability => CompilerOperationCapability.Bidirectional;
    public override int MinimumInputCount => 1;
    public override int MaximumInputCount => 3;
    protected override bool AllowsEmptyOptionalInputs => true;
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } =
    [
        CompilerTorchSharpForm.Call("torch.clamp"),
        CompilerTorchSharpForm.Call("Tensor.clamp"),
    ];

    public override bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        if (expression is not CompilerInvocationExpression invocation)
        {
            return false;
        }

        var operands = (context.Receiver is null
                ? Array.Empty<CompilerExpression>()
                : [context.Receiver])
            .Concat(invocation.Arguments)
            .ToArray();
        if (operands.Length != 3)
        {
            throw context.Unsupported(
                expression,
                "TorchSharp clamp requires an input, a minimum, and a maximum; use null for an omitted bound.");
        }

        var input = context.RequireTensorReference(operands[0], OnnxName);
        var minimum = AddBound(operands[1], "min");
        var maximum = AddBound(operands[2], "max");
        var node = new Onnxify.Clip(
            OnnxName.ToLowerInvariant(),
            new Onnxify.ClipInputOutputOptions
            {
                Input = new OnnxEdge(input.Name),
                Min = minimum,
                Max = maximum,
                Output = context.RequireSingleOutputEdge(this),
            });
        context.AddOperation(this, node, expression.Span);
        return true;

        OnnxEdge? AddBound(CompilerExpression bound, string label)
        {
            if (bound is CompilerLiteralExpression { Literal: CompilerNullLiteral })
            {
                return null;
            }

            if (bound is CompilerInvocationExpression
                {
                    Target: CompilerMemberAccessExpression
                    {
                        Target: CompilerReferenceExpression reference,
                        MemberName: "item",
                    },
                    Arguments.Count: 0,
                })
            {
                return new OnnxEdge(context.RequireTensorReference(reference, OnnxName).Name);
            }

            var value = ReadFloatLiteral(bound);
            var name = $"__clip_{label}_{context.Outputs[0].Name}";
            context.Builder.AddStateMember(
                new CompilerStateMember(
                    name,
                    CompilerStateMemberKind.Initializer,
                    new CompilerTensorType(CompilerElementType.Float32, Array.Empty<CompilerDimension>()),
                    new CompilerTensorLiteral(
                        CompilerElementType.Float32,
                        Array.Empty<CompilerDimension>(),
                        [new CompilerFloatingPointLiteral(CompilerElementType.Float32, value)])));
            return new OnnxEdge(name);
        }

        float ReadFloatLiteral(CompilerExpression bound)
        {
            if (bound is CompilerUnaryExpression
                {
                    Operator: "-",
                    Expression: CompilerLiteralExpression { Literal: CompilerFloatingPointLiteral negative },
                })
            {
                return -(float)negative.Value;
            }

            if (bound is CompilerUnaryExpression
                {
                    Operator: "-",
                    Expression: CompilerLiteralExpression { Literal: CompilerSignedIntegerLiteral negativeInteger },
                })
            {
                return -negativeInteger.Value;
            }

            if (bound is CompilerLiteralExpression { Literal: CompilerFloatingPointLiteral floatingPoint })
            {
                return (float)floatingPoint.Value;
            }

            if (bound is CompilerLiteralExpression { Literal: CompilerSignedIntegerLiteral integer })
            {
                return integer.Value;
            }

            if (bound is CompilerLiteralExpression { Literal: CompilerUnsignedIntegerLiteral unsignedInteger })
            {
                return unsignedInteger.Value;
            }

            throw context.Unsupported(bound, "TorchSharp clamp bounds must be Float32 compile-time scalar values or null.");
        }
    }

    protected override string PrintTorchSharp(Onnxify.Clip node, CompilerSourceSpan? span)
    {
        var min = node.Min is null
            ? "null"
            : $"{CompilerCSharpNaming.Identifier(node.Min.Name)}.item<float>()";
        var max = node.Max is null
            ? "null"
            : $"{CompilerCSharpNaming.Identifier(node.Max.Name)}.item<float>()";
        return $"{CompilerCSharpNaming.Identifier(node.Input.Name)}.clamp({min}, {max})";
    }

    public override string PrintTorchSharp(
        CompilerOnnxStep operation,
        CompilerTorchSharpPrintContext context)
    {
        if (operation.Node is not Onnxify.Clip clip)
        {
            throw Unsupported(operation.Span, $"Clip requires generated node type '{nameof(Onnxify.Clip)}'.");
        }

        var input = CompilerCSharpNaming.Identifier(clip.Input.Name);
        var min = PrintBound(clip.Min);
        var max = PrintBound(clip.Max);
        return $"{input}.clamp({min}, {max})";

        string PrintBound(IOnnxGraphEdge? bound)
        {
            if (bound is null)
            {
                return "null";
            }

            if (context.TryGetTensorInitializer(bound.Name, out var initializer)
                && initializer is { Dimensions.Count: 0, Values.Count: 1 }
                && initializer.Values[0] is CompilerFloatingPointLiteral value
                && value.ElementType == CompilerElementType.Float32)
            {
                return Float((float)value.Value);
            }

            return $"{CompilerCSharpNaming.Identifier(bound.Name)}.item<float>()";
        }
    }

    public override string? ValidateOnnxNode(
        OnnxNode node,
        IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        if (node is not Onnxify.Clip clip)
        {
            return $"Clip requires the generated node type '{nameof(Onnxify.Clip)}'.";
        }

        if (clip.Input is null)
        {
            return "ONNX Clip requires its input tensor.";
        }

        foreach (var bound in new[] { clip.Min, clip.Max })
        {
            if (bound is null || !knownTypes.TryGetValue(bound.Name, out var type))
            {
                continue;
            }

            if (type is not CompilerTensorType tensorType
                || tensorType.ElementType != CompilerElementType.Float32
                || tensorType.Dimensions is { Count: > 0 })
            {
                return "Compiler Clip supports Float32 scalar min and max inputs only.";
            }
        }

        return null;
    }
}
