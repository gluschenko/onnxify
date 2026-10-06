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
        CompilerOperationCapability capability = CompilerOperationCapability.Bidirectional
    )
    {
        OnnxName = onnxName;
        OnnxDomain = string.Empty;
        TorchSharpNames = torchSharpNames.ToArray();
        InputCount = inputCount;
        AttributeNames = (attributeNames ?? Array.Empty<string>()).ToArray();
        FixedTorchSharpArguments = (fixedTorchSharpArguments ?? Array.Empty<float>()).ToArray();
        Capability = capability;
        Descriptor = new CompilerOperatorDescriptor(
            onnxName,
            OnnxDomain,
            capability,
            [
                $"Inputs: {InputCount}; attributes: {string.Join(", ", AttributeNames)}",
                $"TorchSharp forms: {string.Join(", ", TorchSharpNames)}",
            ]);
    }

    public string OnnxName { get; }

    public string OnnxDomain { get; }

    public IReadOnlyList<string> TorchSharpNames { get; }

    public IReadOnlyList<string> AttributeNames { get; }

    public IReadOnlyList<float> FixedTorchSharpArguments { get; }

    public int InputCount { get; }

    public CompilerOperationCapability Capability { get; }

    public CompilerOperatorDescriptor Descriptor { get; }

    public bool MatchesTorchSharpName(string name) => TorchSharpNames.Contains(name, StringComparer.Ordinal);

    public bool Accepts(CompilerOperation operation)
    {
        return operation.Inputs.Count == InputCount
            && operation.Outputs.Count == 1
            && operation.Inputs.All(input => !input.IsEmptyOptional)
            && operation.Outputs.All(output => !output.IsEmptyOptional)
            && operation.Attributes.All(attribute => AttributeNames.Contains(attribute.Name, StringComparer.Ordinal));
    }

    public bool Accepts(OnnxNode node)
    {
        return node.Inputs.Count == InputCount
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

/// <summary>Single registry for compiler activation identity and directional capability.</summary>
internal static class CompilerOperatorMappingRegistry
{
    private static readonly CompilerOperatorMapping[] MAPPINGS =
    [
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
