using System.Globalization;
using Onnxify;

namespace Onnxify.Compiler;

/// <summary>Compiler-owned semantic mapping shared by ONNX and TorchSharp frontends/backends.</summary>
internal sealed class CompilerOperatorMapping
{
    public CompilerOperatorMapping(
        string onnxName,
        IEnumerable<string> torchSharpNames,
        int inputCount = 1,
        IEnumerable<string>? attributeNames = null,
        IEnumerable<float>? fixedTorchSharpArguments = null,
        string? binaryOperator = null,
        string? unaryOperator = null,
        string? torchSharpMethod = null,
        bool supportsMultidirectionalBroadcast = false,
        IEnumerable<CompilerElementType?>? inputElementTypes = null,
        CompilerElementType? outputElementType = null,
        CompilerOperationCapability capability = CompilerOperationCapability.Bidirectional,
        int? minimumInputCount = null,
        int? maximumInputCount = null
    )
    {
        OnnxName = onnxName;
        OnnxDomain = string.Empty;
        TorchSharpNames = torchSharpNames.ToArray();
        InputCount = inputCount;
        MinimumInputCount = minimumInputCount ?? inputCount;
        MaximumInputCount = maximumInputCount ?? inputCount;
        AttributeNames = (attributeNames ?? Array.Empty<string>()).ToArray();
        FixedTorchSharpArguments = (fixedTorchSharpArguments ?? Array.Empty<float>()).ToArray();
        BinaryOperator = binaryOperator;
        UnaryOperator = unaryOperator;
        TorchSharpMethod = torchSharpMethod;
        SupportsMultidirectionalBroadcast = supportsMultidirectionalBroadcast;
        InputElementTypes = (inputElementTypes ?? Array.Empty<CompilerElementType?>()).ToArray();
        OutputElementType = outputElementType;
        Capability = capability;
        Descriptor = new CompilerOperatorDescriptor(
            onnxName,
            OnnxDomain,
            capability,
            [
                MinimumInputCount == MaximumInputCount
                    ? $"Inputs: {InputCount}; attributes: {string.Join(", ", AttributeNames)}"
                    : $"Inputs: {MinimumInputCount}-{MaximumInputCount}; attributes: {string.Join(", ", AttributeNames)}",
                $"TorchSharp forms: {string.Join(", ", TorchSharpNames)}",
            ]);
    }

    public string OnnxName { get; }

    public string OnnxDomain { get; }

    public IReadOnlyList<string> TorchSharpNames { get; }

    public IReadOnlyList<string> AttributeNames { get; }

    public IReadOnlyList<float> FixedTorchSharpArguments { get; }

    public int InputCount { get; }

    public int MinimumInputCount { get; }

    public int MaximumInputCount { get; }

    public string? BinaryOperator { get; }

    public string? UnaryOperator { get; }

    public string? TorchSharpMethod { get; }

    public bool SupportsMultidirectionalBroadcast { get; }

    public IReadOnlyList<CompilerElementType?> InputElementTypes { get; }

    public CompilerElementType? OutputElementType { get; }

    public CompilerOperationCapability Capability { get; }

    public CompilerOperatorDescriptor Descriptor { get; }

    public bool MatchesTorchSharpName(string name) => TorchSharpNames.Contains(name, StringComparer.Ordinal);

    public bool Accepts(CompilerOperation operation)
    {
        return operation.Inputs.Count >= MinimumInputCount
            && operation.Inputs.Count <= MaximumInputCount
            && operation.Outputs.Count == 1
            && (OnnxName == "Gemm" || operation.Inputs.All(input => !input.IsEmptyOptional))
            && operation.Outputs.All(output => !output.IsEmptyOptional)
            && operation.Attributes.All(attribute => AttributeNames.Contains(attribute.Name, StringComparer.Ordinal));
    }

    public bool Accepts(OnnxNode node)
    {
        return node.Inputs.Count >= MinimumInputCount
            && node.Inputs.Count <= MaximumInputCount
            && node.Outputs.Count == 1
            && node.Attributes.All(attribute => AttributeNames.Contains(attribute.Name, StringComparer.Ordinal));
    }

    public string EmitTorchSharpExpression(CompilerOperation operation)
    {
        if (!Accepts(operation))
        {
            throw new CSharpCompilerDiagnosticException(
                CompilerDiagnosticCodes.Unsupported,
                $"ONNX operator '{OnnxName}' has an unsupported input or attribute configuration.",
                CompilerDiagnosticStage.Emit,
                operation.Span);
        }

        var inputs = operation.Inputs.Select(reference => reference.Name).ToArray();
        var x = inputs[0];
        var y = inputs.Length > 1 ? inputs[1] : string.Empty;
        var z = inputs.Length > 2 ? inputs[2] : string.Empty;
        if (OnnxName == "MatMul")
        {
            return $"torch.matmul({x}, {y})";
        }
        if (OnnxName == "Gemm")
        {
            return EmitGemmExpression(operation, x, y, z);
        }
        if (OnnxName == "Where")
        {
            return $"torch.where({x}, {y}, {z})";
        }
        if (OnnxName == "Cast")
        {
            var targetType = GetCastTargetType(operation);
            return $"{x}.to_type(torch.ScalarType.{targetType})";
        }
        if (BinaryOperator is not null)
        {
            return $"({x} {BinaryOperator} {y})";
        }

        if (UnaryOperator is not null)
        {
            return $"({UnaryOperator}{x})";
        }

        if (TorchSharpMethod is not null)
        {
            return InputCount == 1
                ? $"{x}.{TorchSharpMethod}()"
                : $"{x}.{TorchSharpMethod}({y})";
        }

        return OnnxName switch
        {
            "Celu" => $"torch.nn.functional.celu({x}, alpha: {GetFloatAttribute(operation, "alpha", 1f)}f)",
            "Elu" => $"torch.nn.functional.elu({x}, alpha: {GetFloatAttribute(operation, "alpha", 1f)}f)",
            "Gelu" => GeluExpression(x, operation),
            "HardSigmoid" => $"({x} * {GetFloatAttribute(operation, "alpha", 0.2f)}f + {GetFloatAttribute(operation, "beta", 0.5f)}f).clamp(0.0f, 1.0f)",
            "HardSwish" => $"({x} * ({x} + 3.0f).clamp(0.0f, 6.0f) / 6.0f)",
            "LeakyRelu" => $"torch.nn.functional.leaky_relu({x}, negative_slope: {GetFloatAttribute(operation, "alpha", 0.01f)}f)",
            "Mish" => $"({x} * {x}.softplus().tanh())",
            "PRelu" => $"torch.nn.functional.prelu({x}, {y})",
            "Pow" => $"{x}.pow({y})",
            "Relu" => $"torch.nn.functional.relu({x})",
            "Selu" => SeluExpression(x, operation),
            "Sigmoid" => $"{x}.sigmoid()",
            "Softplus" => $"{x}.softplus()",
            "Softsign" => $"({x} / (1.0f + {x}.abs()))",
            "Swish" => SwishExpression(x, operation),
            "Tanh" => $"{x}.tanh()",
            "ThresholdedRelu" => $"torch.nn.functional.threshold({x}, {GetFloatAttribute(operation, "alpha", 1f)}f, 0.0f)",
            _ => throw new CSharpCompilerDiagnosticException(
                CompilerDiagnosticCodes.Unsupported,
                $"No TorchSharp emitter is registered for ONNX operator '{OnnxName}'.",
                CompilerDiagnosticStage.Emit,
                operation.Span),
        };
    }

    private static string EmitGemmExpression(CompilerOperation operation, string a, string b, string c)
    {
        var transA = GetIntegerAttribute(operation, "transA", 0) == 1;
        var transB = GetIntegerAttribute(operation, "transB", 0) == 1;
        if (transA)
        {
            a = $"{a}.transpose(0, 1)";
        }

        if (transB)
        {
            b = $"{b}.transpose(0, 1)";
        }

        var alpha = GetFloatAttribute(operation, "alpha", 1f).ToString("R", CultureInfo.InvariantCulture) + "f";
        var beta = GetFloatAttribute(operation, "beta", 1f).ToString("R", CultureInfo.InvariantCulture) + "f";
        var product = $"torch.matmul({a}, {b})";
        product = alpha == "1f" ? product : $"({alpha} * {product})";
        return string.IsNullOrEmpty(c) ? product : $"({product} + ({beta} * {c}))";
    }

    private static long GetIntegerAttribute(CompilerOperation operation, string name, long defaultValue)
    {
        var attribute = operation.Attributes.FirstOrDefault(value => string.Equals(value.Name, name, StringComparison.Ordinal));
        return attribute?.Value switch
        {
            null => defaultValue,
            CompilerSignedIntegerLiteral signed => signed.Value,
            CompilerUnsignedIntegerLiteral unsigned => checked((long)unsigned.Value),
            _ => throw new CSharpCompilerDiagnosticException(
                CompilerDiagnosticCodes.Unsupported,
                $"Gemm attribute '{name}' must be an integer.",
                CompilerDiagnosticStage.Emit,
                operation.Span),
        };
    }

    private static string GeluExpression(string input, CompilerOperation operation)
    {
        var approximate = StringAttribute(operation, "approximate", "none");
        if (approximate is not ("none" or "tanh"))
        {
            throw new CSharpCompilerDiagnosticException(
                CompilerDiagnosticCodes.Unsupported,
                $"Gelu approximate mode '{approximate}' is unsupported.",
                CompilerDiagnosticStage.Emit,
                operation.Span);
        }

        var approximateValue = approximate == "tanh"
            ? "global::TorchSharp.Modules.GELU.Approximate.tanh"
            : "global::TorchSharp.Modules.GELU.Approximate.none";
        return $"torch.nn.functional.gelu({input}, approximate: {approximateValue})";
    }

    private static string SeluExpression(string input, CompilerOperation operation)
    {
        var alpha = GetFloatAttribute(operation, "alpha", 1.6732631921768188f);
        var gamma = GetFloatAttribute(operation, "gamma", 1.0507010221481323f);
        return alpha == 1.6732631921768188f && gamma == 1.0507010221481323f
            ? $"torch.nn.functional.selu({input})"
            : $"torch.where({input} > 0.0f, ({gamma.ToString("R", CultureInfo.InvariantCulture)}f * {input}), ({alpha.ToString("R", CultureInfo.InvariantCulture)}f * {gamma.ToString("R", CultureInfo.InvariantCulture)}f * ({input}.exp() - 1.0f)))";
    }

    private static string SwishExpression(string input, CompilerOperation operation)
    {
        var alpha = GetFloatAttribute(operation, "alpha", 1f);
        return alpha == 1f
            ? $"({input} * {input}.sigmoid())"
            : $"({input} * ({input} * {alpha.ToString("R", CultureInfo.InvariantCulture)}f).sigmoid())";
    }

    public static float GetFloatAttribute(CompilerOperation operation, string name, float defaultValue)
    {
        var attribute = operation.Attributes.FirstOrDefault(value => string.Equals(value.Name, name, StringComparison.Ordinal));
        return attribute?.Value switch
        {
            null => defaultValue,
            CompilerFloatingPointLiteral floatingPoint => (float)floatingPoint.Value,
            CompilerSignedIntegerLiteral integer => integer.Value,
            CompilerUnsignedIntegerLiteral integer => integer.Value,
            _ => throw new CSharpCompilerDiagnosticException(
                CompilerDiagnosticCodes.Unsupported,
                $"Activation attribute '{name}' must be numeric.",
                CompilerDiagnosticStage.Emit,
                operation.Span),
        };
    }

    private static string GetCastTargetType(CompilerOperation operation)
    {
        var attribute = operation.Attributes.FirstOrDefault(static value => value.Name == "to");
        var onnxType = attribute?.Value switch
        {
            CompilerSignedIntegerLiteral signed => signed.Value,
            CompilerUnsignedIntegerLiteral unsigned => checked((long)unsigned.Value),
            _ => 0,
        };
        return onnxType switch
        {
            1 => "Float32",
            2 => "UInt8",
            3 => "Int8",
            4 => "UInt16",
            5 => "Int16",
            6 => "Int32",
            7 => "Int64",
            9 => "Bool",
            10 => "Float16",
            11 => "Float64",
            12 => "UInt32",
            13 => "UInt64",
            16 => "BFloat16",
            _ => throw new CSharpCompilerDiagnosticException(
                CompilerDiagnosticCodes.Unsupported,
                $"ONNX Cast target element type '{onnxType}' is unsupported.",
                CompilerDiagnosticStage.Emit,
                operation.Span),
        };
    }

    private static string StringAttribute(CompilerOperation operation, string name, string defaultValue)
    {
        var attribute = operation.Attributes.FirstOrDefault(value => string.Equals(value.Name, name, StringComparison.Ordinal));
        return attribute?.Value switch
        {
            null => defaultValue,
            CompilerStringLiteral text => text.Value,
            _ => throw new CSharpCompilerDiagnosticException(
                CompilerDiagnosticCodes.Unsupported,
                $"Activation attribute '{name}' must be a string.",
                CompilerDiagnosticStage.Emit,
                operation.Span),
        };
    }
}

/// <summary>Single registry for compiler operator identity, syntax forms, and directional capability.</summary>
internal static class CompilerOperatorMappingRegistry
{
    private static readonly CompilerOperatorMapping[] MAPPINGS =
    [
        new("Abs", ["torch.abs", "Tensor.abs"], torchSharpMethod: "abs"),
        new("Add", ["torch.add", "Tensor.add"], inputCount: 2, binaryOperator: "+", supportsMultidirectionalBroadcast: true),
        new("Acos", ["torch.acos", "Tensor.acos"], torchSharpMethod: "acos"),
        new("Acosh", ["torch.acosh", "Tensor.acosh"], torchSharpMethod: "acosh"),
        new("Asin", ["torch.asin", "Tensor.asin"], torchSharpMethod: "asin"),
        new("Asinh", ["torch.asinh", "Tensor.asinh"], torchSharpMethod: "asinh"),
        new("Atan", ["torch.atan", "Tensor.atan"], torchSharpMethod: "atan"),
        new("Atanh", ["torch.atanh", "Tensor.atanh"], torchSharpMethod: "atanh"),
        new("Ceil", ["torch.ceil", "Tensor.ceil"], torchSharpMethod: "ceil"),
        new("Cast", ["Tensor.to_type"], attributeNames: ["to"]),
        new("Cos", ["torch.cos", "Tensor.cos"], torchSharpMethod: "cos"),
        new("Cosh", ["torch.cosh", "Tensor.cosh"], torchSharpMethod: "cosh"),
        new("Exp", ["torch.exp", "Tensor.exp"], torchSharpMethod: "exp"),
        new("Floor", ["torch.floor", "Tensor.floor"], torchSharpMethod: "floor"),
        new("Log", ["torch.log", "Tensor.log"], torchSharpMethod: "log"),
        new("Mod", ["torch.remainder", "Tensor.remainder"], inputCount: 2, torchSharpMethod: "remainder", supportsMultidirectionalBroadcast: true),
        new("MatMul", ["torch.matmul", "torch.mm", "torch.bmm", "Tensor.matmul", "Tensor.mm", "Tensor.bmm"], inputCount: 2),
        new("Gemm", ["torch.addmm", "Tensor.addmm", "torch.nn.functional.linear"], inputCount: 3, attributeNames: ["alpha", "beta", "transA", "transB"], minimumInputCount: 2, maximumInputCount: 3),
        new("Neg", ["torch.neg", "Tensor.neg"], unaryOperator: "-"),
        new("Sub", ["torch.sub", "Tensor.sub"], inputCount: 2, binaryOperator: "-", supportsMultidirectionalBroadcast: true),
        new("Sin", ["torch.sin", "Tensor.sin"], torchSharpMethod: "sin"),
        new("Sinh", ["torch.sinh", "Tensor.sinh"], torchSharpMethod: "sinh"),
        new("Sqrt", ["torch.sqrt", "Tensor.sqrt"], torchSharpMethod: "sqrt"),
        new("Tan", ["torch.tan", "Tensor.tan"], torchSharpMethod: "tan"),
        new("Mul", ["torch.mul", "Tensor.mul"], inputCount: 2, binaryOperator: "*", supportsMultidirectionalBroadcast: true),
        new("Div", ["torch.div", "Tensor.div"], inputCount: 2, binaryOperator: "/", supportsMultidirectionalBroadcast: true),
        new("Equal", ["torch.eq", "Tensor.eq"], inputCount: 2, binaryOperator: "==", torchSharpMethod: "eq", supportsMultidirectionalBroadcast: true, outputElementType: CompilerElementType.Boolean),
        new("Pow", ["torch.pow", "Tensor.pow"], inputCount: 2, supportsMultidirectionalBroadcast: true),
        new("Greater", ["torch.gt", "Tensor.gt"], inputCount: 2, binaryOperator: ">", torchSharpMethod: "gt", supportsMultidirectionalBroadcast: true, outputElementType: CompilerElementType.Boolean),
        new("GreaterOrEqual", ["torch.ge", "Tensor.ge"], inputCount: 2, binaryOperator: ">=", torchSharpMethod: "ge", supportsMultidirectionalBroadcast: true, outputElementType: CompilerElementType.Boolean),
        new("Less", ["torch.lt", "Tensor.lt"], inputCount: 2, binaryOperator: "<", torchSharpMethod: "lt", supportsMultidirectionalBroadcast: true, outputElementType: CompilerElementType.Boolean),
        new("LessOrEqual", ["torch.le", "Tensor.le"], inputCount: 2, binaryOperator: "<=", torchSharpMethod: "le", supportsMultidirectionalBroadcast: true, outputElementType: CompilerElementType.Boolean),
        new("Max", ["torch.maximum", "Tensor.maximum"], inputCount: 2, torchSharpMethod: "maximum", supportsMultidirectionalBroadcast: true),
        new("Min", ["torch.minimum", "Tensor.minimum"], inputCount: 2, torchSharpMethod: "minimum", supportsMultidirectionalBroadcast: true),
        new("And", ["torch.logical_and", "Tensor.logical_and"], inputCount: 2, torchSharpMethod: "logical_and", supportsMultidirectionalBroadcast: true, inputElementTypes: [CompilerElementType.Boolean, CompilerElementType.Boolean], outputElementType: CompilerElementType.Boolean),
        new("Or", ["torch.logical_or", "Tensor.logical_or"], inputCount: 2, torchSharpMethod: "logical_or", supportsMultidirectionalBroadcast: true, inputElementTypes: [CompilerElementType.Boolean, CompilerElementType.Boolean], outputElementType: CompilerElementType.Boolean),
        new("Xor", ["torch.logical_xor", "Tensor.logical_xor"], inputCount: 2, torchSharpMethod: "logical_xor", supportsMultidirectionalBroadcast: true, inputElementTypes: [CompilerElementType.Boolean, CompilerElementType.Boolean], outputElementType: CompilerElementType.Boolean),
        new("Not", ["torch.logical_not", "Tensor.logical_not"], unaryOperator: "!", torchSharpMethod: "logical_not", inputElementTypes: [CompilerElementType.Boolean], outputElementType: CompilerElementType.Boolean),
        new("Where", ["torch.where"], inputCount: 3, supportsMultidirectionalBroadcast: true, inputElementTypes: [CompilerElementType.Boolean, null, null]),
        new("Reciprocal", ["torch.reciprocal", "Tensor.reciprocal"], torchSharpMethod: "reciprocal"),
        new("Round", ["torch.round", "Tensor.round"], torchSharpMethod: "round"),
        new("Sign", ["torch.sign", "Tensor.sign"], torchSharpMethod: "sign"),
        new("Trunc", ["torch.trunc", "Tensor.trunc"], torchSharpMethod: "trunc"),
        new("Erf", ["torch.erf", "Tensor.erf"], torchSharpMethod: "erf"),
        new("IsNaN", ["torch.isnan", "Tensor.isnan"], torchSharpMethod: "isnan", outputElementType: CompilerElementType.Boolean),
        new("Celu", ["torch.nn.functional.celu"], attributeNames: ["alpha"]),
        new("Elu", ["torch.nn.functional.elu"], attributeNames: ["alpha"]),
        new("Gelu", ["torch.nn.functional.gelu"], attributeNames: ["approximate"]),
        new("HardSigmoid", ["torch.nn.functional.hardsigmoid"], attributeNames: ["alpha", "beta"]),
        new("HardSwish", ["torch.nn.functional.hardswish"]),
        new("LeakyRelu", ["torch.nn.functional.leaky_relu"], attributeNames: ["alpha"]),
        new("Mish", ["torch.nn.functional.mish"]),
        new("PRelu", ["torch.nn.functional.prelu"], inputCount: 2),
        new("Relu", ["torch.nn.functional.relu"]),
        new("Selu", ["torch.nn.functional.selu"], attributeNames: ["alpha", "gamma"]),
        new("Sigmoid", ["torch.nn.functional.sigmoid", "Tensor.sigmoid"]),
        new("Softplus", ["torch.nn.functional.softplus", "Tensor.softplus"]),
        new("Softsign", ["torch.nn.functional.softsign"]),
        new("Swish", ["torch.nn.functional.silu"], attributeNames: ["alpha"]),
        new("Tanh", ["torch.nn.functional.tanh", "Tensor.tanh"]),
        new("ThresholdedRelu", ["torch.nn.functional.threshold"], attributeNames: ["alpha"], fixedTorchSharpArguments: [0f]),
    ];

    public static bool TryGetOnnx(
        string domain,
        string name,
        out CompilerOperatorMapping? mapping
    )
    {
        mapping = MAPPINGS.FirstOrDefault(candidate =>
            IsSameOnnxDomain(candidate.OnnxDomain, domain)
            && string.Equals(candidate.OnnxName, name, StringComparison.Ordinal));
        return mapping is not null;
    }

    public static bool TryGetTorchSharp(
        string name,
        out CompilerOperatorMapping? mapping
    )
    {
        mapping = MAPPINGS.FirstOrDefault(candidate => candidate.MatchesTorchSharpName(name));
        return mapping is not null;
    }

    public static bool TryGetTorchSharpBinaryOperator(
        string operation,
        out CompilerOperatorMapping? mapping
    )
    {
        mapping = MAPPINGS.FirstOrDefault(candidate =>
            string.Equals(candidate.BinaryOperator, operation, StringComparison.Ordinal));
        return mapping is not null;
    }

    public static bool TryGetTorchSharpUnaryOperator(
        string operation,
        out CompilerOperatorMapping? mapping
    )
    {
        mapping = MAPPINGS.FirstOrDefault(candidate =>
            string.Equals(candidate.UnaryOperator, operation, StringComparison.Ordinal));
        return mapping is not null;
    }

    public static bool TryGetTorchSharpCall(
        CompilerExpression target,
        out CompilerOperatorMapping? mapping,
        out CompilerExpression? receiver
    )
    {
        receiver = null;
        mapping = null;
        if (TryGetMemberPath(target, out var name)
            && TryGetTorchSharp(name, out mapping))
        {
            return true;
        }

        if (target is CompilerMemberAccessExpression member
            && member.Target is CompilerReferenceExpression
            && TryGetTorchSharp($"Tensor.{member.MemberName}", out mapping))
        {
            receiver = member.Target;
            return true;
        }

        return false;
    }

    public static string GetTorchSharpCallName(CompilerExpression target, CompilerExpression? receiver)
    {
        if (receiver is not null && target is CompilerMemberAccessExpression member)
        {
            return $"Tensor.{member.MemberName}";
        }

        return TryGetMemberPath(target, out var name) ? name : string.Empty;
    }

    private static bool IsSameOnnxDomain(string expected, string actual)
    {
        var normalizedActual = string.Equals(actual, "ai.onnx", StringComparison.Ordinal)
            ? string.Empty
            : actual ?? string.Empty;
        return string.Equals(expected, normalizedActual, StringComparison.Ordinal);
    }

    private static bool TryGetMemberPath(CompilerExpression expression, out string name)
    {
        if (expression is CompilerReferenceExpression reference)
        {
            name = reference.Name;
            return true;
        }

        if (expression is CompilerMemberAccessExpression member
            && TryGetMemberPath(member.Target, out var prefix))
        {
            name = $"{prefix}.{member.MemberName}";
            return true;
        }

        name = string.Empty;
        return false;
    }
}
