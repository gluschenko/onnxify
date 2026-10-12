using System.Globalization;
using Onnxify;

namespace Onnxify.Compiler.Operators;

/// <summary>Prints captured ONNX branch graphs as lazily executed TorchSharp tensor-array expressions.</summary>
internal static class CompilerOnnxGraphCSharpPrinter
{
    private const string TENSOR_TYPE = "global::TorchSharp.torch.Tensor";

    public static string PrintTensorArray(
        OnnxGraph graph,
        CompilerSourceSpan? span,
        IReadOnlyDictionary<string, string>? inputBindings = null)
    {
        if (graph.Inputs.Count != 0 && inputBindings is null)
        {
            throw Unsupported(span, "ONNX branch graphs must use lexical captures instead of explicit graph inputs.");
        }

        var lines = new List<string>();
        foreach (var input in graph.Inputs)
        {
            if (inputBindings is null || !inputBindings.TryGetValue(input.Name, out var binding))
            {
                throw Unsupported(span, $"ONNX subgraph input '{input.Name}' has no runtime binding.");
            }

            lines.Add($"var {CompilerCSharpNaming.Identifier(input.Name)} = {binding};");
        }

        foreach (var initializer in graph.Initializers)
        {
            lines.Add($"var {CompilerCSharpNaming.Identifier(initializer.Name)} = {PrintTensor(initializer, span)};");
        }

        foreach (var node in graph.Nodes)
        {
            if (!CompilerOperatorRegistry.TryGetOnnx(node, out var mapping) || mapping is null)
            {
                throw Unsupported(span, $"ONNX If branch contains unsupported operator '{node.Domain}::{node.OpType}'.");
            }

            var operation = mapping.ScanOnnx(node, node.Name, span);
            if (!mapping.Accepts(operation)
                || mapping.Capability is not (CompilerOperationCapability.Bidirectional or CompilerOperationCapability.ImportOnly))
            {
                throw Unsupported(span, $"ONNX If branch operator '{node.Domain}::{node.OpType}' has no TorchSharp import mapping.");
            }

            var expression = mapping.PrintTorchSharp(operation);
            if (operation.Outputs.Count == 1)
            {
                lines.Add($"var {CompilerCSharpNaming.Identifier(operation.Outputs[0].Name)} = {expression};");
                continue;
            }

            var resultsName = CompilerCSharpNaming.Identifier($"__{operation.Outputs[0].Name}_outputs");
            lines.Add($"var {resultsName} = {expression};");
            for (var index = 0; index < operation.Outputs.Count; index++)
            {
                lines.Add($"var {CompilerCSharpNaming.Identifier(operation.Outputs[index].Name)} = {resultsName}[{index}];");
            }
        }

        var outputs = string.Join(", ", graph.Outputs.Select(output => CompilerCSharpNaming.Identifier(output.Name)));
        lines.Add($"return new {TENSOR_TYPE}[] {{ {outputs} }};");
        var body = string.Join(" ", lines);
        return $"((global::System.Func<{TENSOR_TYPE}[]>)(() => {{ {body} }}))()";
    }

    private static string PrintTensor(OnnxTensor tensor, CompilerSourceSpan? span)
    {
        var (elementType, scalarType) = tensor.DataType == typeof(float) ? ("float", "Float32")
            : tensor.DataType == typeof(double) ? ("double", "Float64")
            : tensor.DataType == typeof(long) ? ("long", "Int64")
            : tensor.DataType == typeof(int) ? ("int", "Int32")
            : tensor.DataType == typeof(bool) ? ("bool", "Bool")
            : tensor.DataType == typeof(byte) ? ("byte", "UInt8")
            : tensor.DataType == typeof(sbyte) ? ("sbyte", "Int8")
            : throw Unsupported(span, $"ONNX If branch initializer dtype '{tensor.DataType.Name}' is not supported by TorchSharp printing.");
        var values = string.Join(", ", tensor.Values.Select(value => PrintScalar(value, tensor.DataType, span)));
        var dimensions = string.Join(", ", tensor.Shape.Select(dimension => dimension.ToString(CultureInfo.InvariantCulture) + "L"));
        return $"torch.tensor(new {elementType}[] {{ {values} }}, new long[] {{ {dimensions} }}, dtype: torch.ScalarType.{scalarType})";
    }

    private static string PrintScalar(object? value, Type type, CompilerSourceSpan? span)
    {
        if (value is null)
        {
            throw Unsupported(span, "ONNX If branch tensor initializer contains a null scalar value.");
        }

        if (type == typeof(bool)) return (bool)value ? "true" : "false";
        if (type == typeof(float)) return ((float)value).ToString("R", CultureInfo.InvariantCulture) + "f";
        if (type == typeof(double)) return ((double)value).ToString("R", CultureInfo.InvariantCulture) + "d";
        if (type == typeof(byte)) return ((byte)value).ToString(CultureInfo.InvariantCulture);
        if (type == typeof(sbyte)) return ((sbyte)value).ToString(CultureInfo.InvariantCulture);
        if (type == typeof(int)) return ((int)value).ToString(CultureInfo.InvariantCulture);
        if (type == typeof(long)) return ((long)value).ToString(CultureInfo.InvariantCulture) + "L";
        throw Unsupported(span, $"ONNX If branch initializer scalar type '{type.Name}' is not supported.");
    }

    private static CSharpCompilerDiagnosticException Unsupported(CompilerSourceSpan? span, string message) =>
        new(CompilerDiagnosticCodes.Unsupported, message, CompilerDiagnosticStage.Emit, span);
}
