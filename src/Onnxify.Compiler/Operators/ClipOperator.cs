using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::clamp")]
[CompilerTorchOperator("aten::clamp.Tensor")]
[CompilerTorchOperator("aten::clamp_min")]
[CompilerTorchOperator("aten::clamp_min.Tensor")]
[CompilerTorchOperator("aten::clamp_max")]
[CompilerTorchOperator("aten::clamp_max.Tensor")]
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
        CompilerTorchSharpForm.Call("torch.clamp_min"),
        CompilerTorchSharpForm.Call("Tensor.clamp_min"),
        CompilerTorchSharpForm.Call("torch.clamp_max"),
        CompilerTorchSharpForm.Call("Tensor.clamp_max"),
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
        var clampMinimum = context.CallName is "torch.clamp_min" or "Tensor.clamp_min";
        var clampMaximum = context.CallName is "torch.clamp_max" or "Tensor.clamp_max";
        if ((clampMinimum || clampMaximum) && operands.Length != 2)
        {
            throw context.Unsupported(expression, "TorchSharp clamp_min and clamp_max require an input and exactly one bound.");
        }

        if (!clampMinimum && !clampMaximum && operands.Length != 3)
        {
            throw context.Unsupported(
                expression,
                "TorchSharp clamp requires an input, a minimum, and a maximum; use null for an omitted bound.");
        }

        var input = context.RequireTensorReference(operands[0], OnnxName);
        CompilerExpression? minimumExpression = null;
        CompilerExpression? maximumExpression = null;
        if (clampMinimum)
        {
            minimumExpression = operands[1];
        }
        else if (clampMaximum)
        {
            maximumExpression = operands[1];
        }
        else
        {
            minimumExpression = operands[1];
            maximumExpression = operands[2];
        }

        var minimumTensor = TryGetTensorReference(minimumExpression);
        var maximumTensor = TryGetTensorReference(maximumExpression);
        if (minimumTensor is not null || maximumTensor is not null)
        {
            if ((minimumExpression is not null && minimumTensor is null && !IsNull(minimumExpression))
                || (maximumExpression is not null && maximumTensor is null && !IsNull(maximumExpression)))
            {
                throw context.Unsupported(expression, "Tensor clamp overloads require tensor bounds or null; scalar clamp overloads require scalar bounds.");
            }

            AddTensorClampOperations(input, minimumTensor, maximumTensor);
            return true;
        }

        var minimum = clampMinimum ? AddBound(operands[1], "min") : clampMaximum ? null : AddBound(operands[1], "min");
        var maximum = clampMaximum ? AddBound(operands[1], "max") : clampMinimum ? null : AddBound(operands[2], "max");
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

        CompilerValueReference? TryGetTensorReference(CompilerExpression? bound)
        {
            if (bound is CompilerReferenceExpression reference
                && context.Inputs.Any(input => string.Equals(input.Name, reference.Name, StringComparison.Ordinal)
                    && input.Type is CompilerTensorType))
            {
                return new CompilerValueReference(reference.Name);
            }

            return null;
        }

        bool IsNull(CompilerExpression bound) => bound is CompilerLiteralExpression { Literal: CompilerNullLiteral };

        void AddTensorClampOperations(
            CompilerValueReference inputReference,
            CompilerValueReference? minReference,
            CompilerValueReference? maxReference)
        {
            var inputType = context.Inputs
                .First(input => string.Equals(input.Name, inputReference.Name, StringComparison.Ordinal))
                .Type;
            var current = inputReference;

            if (minReference is not null)
            {
                current = AddOperation(
                    new MaxOperator(),
                    "min",
                    current,
                    minReference,
                    maxReference is null ? context.Outputs[0].Name : $"{context.Outputs[0].Name}__after_min",
                    inputType);
            }

            if (maxReference is not null)
            {
                AddOperation(
                    new MinOperator(),
                    "max",
                    current,
                    maxReference,
                    context.Outputs[0].Name,
                    inputType);
            }
        }

        CompilerValueReference AddOperation(
            CompilerOperator compilerOperator,
            string suffix,
            CompilerValueReference left,
            CompilerValueReference right,
            string outputName,
            CompilerType outputType)
        {
            if (!context.Builder.HasOutput(outputName))
            {
                context.Builder.AddIntermediateValue(new CompilerValue(outputName, outputType, expression.Span));
            }

            context.Builder.AddOperation(compilerOperator.CreateNode(
                $"{outputName}__{suffix}",
                [left, right],
                [new CompilerValueReference(outputName)],
                span: expression.Span));
            return new CompilerValueReference(outputName);
        }

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

        CompilerElementType? inputElementType = knownTypes.TryGetValue(clip.Input.Name, out var inputType)
            && inputType is CompilerTensorType inputTensor
                ? inputTensor.ElementType
                : null;

        foreach (var bound in new[] { clip.Min, clip.Max })
        {
            if (bound is null || !knownTypes.TryGetValue(bound.Name, out var type))
            {
                continue;
            }

            if (type is not CompilerTensorType tensorType
                || tensorType.ElementType != CompilerElementType.Float32
                || tensorType.Dimensions is null
                || tensorType.Dimensions.Any(static dimension => dimension is not CompilerFixedDimension { Value: 1 }))
            {
                return "Compiler Clip supports Float32 scalar or statically known single-element min and max inputs only.";
            }

            if (inputElementType is { } knownInputElementType && knownInputElementType != CompilerElementType.Float32)
            {
                return $"Compiler Clip scalar bounds are Float32, but the input tensor is '{knownInputElementType}'.";
            }
        }

        return null;
    }
}
