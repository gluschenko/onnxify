using System.Globalization;
using Onnxify;

namespace Onnxify.Compiler.Operators;

/// <summary>Imports ONNX Loop as a runtime TorchSharp loop with carried tensor state.</summary>
internal sealed class LoopOperator() : CompilerOperator(CompilerOperatorIdentity.Onnx("Loop"))
{
    private const string TENSOR_TYPE = "global::TorchSharp.torch.Tensor";

    public override Type NodeType => typeof(OnnxNode);
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } = [];
    public override CompilerOperationCapability Capability => CompilerOperationCapability.ImportOnly;
    public override int MinimumInputCount => 2;
    public override int MaximumInputCount => int.MaxValue;
    public override int MinimumOutputCount => 0;
    public override int MaximumOutputCount => int.MaxValue;
    public override IReadOnlyList<string> AttributeNames { get; } = ["body"];
    protected override bool AllowsEmptyOptionalInputs => true;

    public override CompilerOnnxStep CreateNode(
        string name,
        IEnumerable<CompilerValueReference> inputs,
        IEnumerable<CompilerValueReference> outputs,
        IEnumerable<CompilerAttribute>? attributes = null,
        CompilerSourceSpan? span = null)
    {
        var node = new OnnxNode(
            name,
            OnnxName,
            OnnxDomain,
            string.Empty,
            inputs.Select(input => (IOnnxGraphEdge)new OnnxEdge(input.IsEmptyOptional ? string.Empty : input.Name)),
            outputs.Select(output => (IOnnxGraphEdge)new OnnxEdge(output.IsEmptyOptional ? string.Empty : output.Name)),
            (attributes ?? []).Select(attribute => OnnxCompilerBackend.CreateOnnxAttribute(attribute, [], null)));
        return new CompilerOnnxStep(node, Descriptor, span, name);
    }

    public override string PrintTorchSharp(CompilerOnnxStep operation)
    {
        var node = operation.Node;
        var body = node.Attributes.FirstOrDefault(attribute => string.Equals(attribute.Name, "body", StringComparison.Ordinal))?.GetValue() as OnnxGraph
            ?? throw Unsupported(operation.Span, "ONNX Loop requires a graph-valued 'body' attribute.");
        var carriedCount = node.Inputs.Count - 2;
        var scanCount = node.Outputs.Count - carriedCount;
        if (carriedCount < 0 || scanCount < 0 || body.Inputs.Count != carriedCount + 2
            || body.Outputs.Count != carriedCount + scanCount + 1)
        {
            throw Unsupported(operation.Span, "ONNX Loop body inputs or outputs do not match its loop-carried and scan values.");
        }

        var limitInput = string.IsNullOrEmpty(node.Inputs[0].Name)
            ? "null"
            : CompilerCSharpNaming.Identifier(node.Inputs[0].Name);
        var conditionInput = string.IsNullOrEmpty(node.Inputs[1].Name)
            ? "null"
            : CompilerCSharpNaming.Identifier(node.Inputs[1].Name);
        var lines = new List<string>
        {
            limitInput == "null"
                ? "long? __loop_limit = null;"
                : $"long? __loop_limit = {limitInput}.item<long>();",
            $"var __loop_condition = {conditionInput} ?? torch.tensor(new[] {{ true }}, Array.Empty<long>(), dtype: torch.ScalarType.Bool);",
        };

        var carriedNames = new string[carriedCount];
        for (var index = 0; index < carriedCount; index++)
        {
            carriedNames[index] = CompilerCSharpNaming.Identifier($"__loop_state_{index}");
            var initial = CompilerCSharpNaming.Identifier(node.Inputs[index + 2].Name);
            lines.Add($"{TENSOR_TYPE} {carriedNames[index]} = {initial};");
        }

        var scanNames = new string[scanCount];
        for (var index = 0; index < scanCount; index++)
        {
            scanNames[index] = CompilerCSharpNaming.Identifier($"__loop_scan_{index}");
            lines.Add($"var {scanNames[index]} = new global::System.Collections.Generic.List<{TENSOR_TYPE}>();");
        }

        lines.Add("for (long __loop_iteration = 0; (__loop_limit is null || __loop_iteration < __loop_limit.Value) && __loop_condition.item<bool>(); __loop_iteration++)");
        lines.Add("{");
        lines.Add($"    var __loop_iteration_tensor = torch.tensor(new long[] {{ __loop_iteration }}, Array.Empty<long>(), dtype: torch.ScalarType.Int64, device: __loop_condition.device);");
        var bindings = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            [body.Inputs[0].Name] = "__loop_iteration_tensor",
            [body.Inputs[1].Name] = "__loop_condition",
        };
        for (var index = 0; index < carriedCount; index++)
        {
            bindings[body.Inputs[index + 2].Name] = carriedNames[index];
        }

        lines.Add($"    var __loop_body_results = {CompilerOnnxGraphCSharpPrinter.PrintTensorArray(body, operation.Span, bindings)};");
        lines.Add("    __loop_condition = __loop_body_results[0];");
        for (var index = 0; index < carriedCount; index++)
        {
            lines.Add($"    {carriedNames[index]} = __loop_body_results[{index + 1}];");
        }

        for (var index = 0; index < scanCount; index++)
        {
            lines.Add($"    {scanNames[index]}.Add(__loop_body_results[{carriedCount + index + 1}]);");
        }
        lines.Add("}");

        var resultNames = carriedNames.ToList();
        for (var index = 0; index < scanCount; index++)
        {
            var scanType = body.Outputs[carriedCount + index + 1].Type as OnnxTensorType
                ?? throw Unsupported(operation.Span, "ONNX Loop scan outputs must have tensor types.");
            var scanResultName = CompilerCSharpNaming.Identifier($"__loop_scan_result_{index}");
            var dimensions = scanType.Shape?.Dimensions.Select(dimension => dimension.GetValue() is long fixedValue
                ? fixedValue.ToString(CultureInfo.InvariantCulture) + "L"
                : throw Unsupported(operation.Span, "Zero-trip Loop scan outputs require statically known body dimensions.")) ?? [];
            var dimensionsText = string.Join(", ", new[] { "0L" }.Concat(dimensions));
            var dtype = TorchScalarType(scanType.Type, operation.Span);
            lines.Add($"var {scanResultName} = {scanNames[index]}.Count == 0 ? torch.empty(new long[] {{ {dimensionsText} }}, dtype: torch.ScalarType.{dtype}, device: __loop_condition.device) : torch.stack({scanNames[index]}.ToArray(), 0);");
            resultNames.Add(scanResultName);
        }

        lines.Add($"return new {TENSOR_TYPE}[] {{ {string.Join(", ", resultNames)} }};");
        var bodyExpression = string.Join(" ", lines);
        var arrayExpression = $"((global::System.Func<{TENSOR_TYPE}[]>)(() => {{ {bodyExpression} }}))()";
        return node.Outputs.Count == 1 ? $"{arrayExpression}[0]" : arrayExpression;
    }

    private static string TorchScalarType(Type type, CompilerSourceSpan? span)
    {
        try
        {
            return CompilerElementTypeMap.ToTorchSharpScalarTypeName(CompilerElementTypeMap.FromSystemType(type));
        }
        catch (CompilerConversionException)
        {
            throw Unsupported(span, $"Loop scan output dtype '{type.Name}' is not supported by the zero-trip TorchSharp lowering.");
        }
    }
}
