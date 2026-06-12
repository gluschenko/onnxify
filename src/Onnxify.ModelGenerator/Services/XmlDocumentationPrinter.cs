using System.Collections.Immutable;
using System.Text;
using Onnxify.ModelGenerator.Helpers;
using static Onnxify.ModelGenerator.Helpers.TextHelper;
using static Onnxify.ModelGenerator.OnnxModelGenerator;

namespace Onnxify.ModelGenerator.Services;

internal sealed class XmlDocumentationPrinter
{
    internal string InputProperty(ModelTensorContract input)
    {
        return TensorProperty(
            summary: input.IsRequired
                ? $"Gets or initializes the tensor supplied for model input '{input.OnnxName}'."
                : $"Gets or initializes the optional tensor supplied for model input '{input.OnnxName}'.",
            tensor: input
        );
    }

    internal string OutputProperty(ModelTensorContract output)
    {
        return TensorProperty(
            summary: $"Gets the tensor returned for model output '{output.OnnxName}'.",
            tensor: output
        );
    }

    internal string GetTensor()
    {
        return $$"""
        {{XmlSummary("Gets a typed tensor from the raw ONNX Runtime output collection by output name.")}}
        {{XmlParam("name", "The ONNX output name to resolve from the inference result.")}}
        {{XmlReturns("The tensor value for the requested output name.")}}
        """;
    }

    internal string TensorParameters(
        IEnumerable<ModelTensorContract> tensors
    )
    {
        var parameters = tensors.Select(static tensor =>
        {
            return XmlParamXml(
                tensor.MethodParameterName!,
                TensorMethodParameterDescription(tensor)
            );
        });

        return string.Join("\n", parameters);
    }

    internal string MetadataCollection(
        string kind,
        ImmutableArray<ModelTensorContract> tensors
    )
    {
        var summary = $"Describes the generated ONNX model {kind}s using Onnxify metadata objects.";

        return XmlDocumentation(
            summary,
            tensors.Select(MetadataParagraph)
        );
    }

    internal string TensorCollectionType(
        string summary,
        ImmutableArray<ModelTensorContract> tensors,
        string roleLabel
    )
    {
        return XmlDocumentation(
            summary,
            tensors.Select(tensor => TensorCollectionParagraph(tensor, roleLabel))
        );
    }

    internal string TorchModuleType(ModelGenerationSpecification specification)
    {
        var paragraphs = new List<string>
        {
            $"Load compatible initializer values with {FormatCode("LoadWeightsFromOnnx(string modelPath)")} or {FormatCode("LoadWeightsFromOnnx(OnnxModel model)")}."
        };

        if (specification.Inputs.Length == 1)
        {
            paragraphs.Add($"The generated module expects input tensor shape {FormatCode(FormatTensorShape(specification.Inputs[0].Shape))}.");
        }

        return XmlDocumentation(
            $"Reconstructs the ONNX graph from '{specification.FileName}' as a TorchSharp module.",
            paragraphs
        );
    }

    internal string TorchModuleProjectRelativePath()
    {
        return XmlSummary("Gets the source ONNX model path relative to the consuming project directory.");
    }

    internal string TorchModuleConstructor(string torchClassName)
    {
        return XmlDocumentation(
            $"Creates a {torchClassName} TorchSharp module with graph-compatible child modules and registered ONNX initializers.",
            [
                $"Call {FormatCode("LoadWeightsFromOnnx(string modelPath)")} or {FormatCode("LoadWeightsFromOnnx(OnnxModel model)")} before inference or training when weights should be initialized from an ONNX model."
            ]
        );
    }

    internal string TorchModuleLoadWeights()
    {
        return $$"""
        {{XmlSummary("Loads compatible initializer tensors from an ONNX model file into this TorchSharp module.")}}
        {{XmlParam("modelPath", "Path to an ONNX model with a graph-compatible initializer layout.")}}
        """;
    }

    internal string TorchModuleLoadWeightsFromModel()
    {
        return $$"""
        {{XmlSummary("Loads compatible initializer tensors from an ONNX model into this TorchSharp module.")}}
        {{XmlParam("model", "ONNX model with a graph-compatible initializer layout.")}}
        """;
    }

    internal string TorchModuleForward(
        ModelGenerationSpecification specification,
        TorchModuleGenerationSpecification torchModule
    )
    {
        var inputParagraphs = torchModule.Inputs.Select(input =>
        {
            var contract = specification.Inputs.FirstOrDefault(x => string.Equals(x.OnnxName, input.OnnxName, StringComparison.Ordinal));
            return contract is null
                ? $"Input {FormatCode(input.OnnxName)} shape is not described by the source ONNX model."
                : $"Input {FormatCode(input.OnnxName)} shape: {FormatCode(FormatTensorShape(contract.Shape))}.";
        });
        var outputParagraphs = torchModule.Outputs.Select(output =>
        {
            var contract = specification.Outputs.FirstOrDefault(x => string.Equals(x.OnnxName, output.OnnxName, StringComparison.Ordinal));
            return contract is null
                ? $"Output {FormatCode(output.OnnxName)} shape is not described by the source ONNX model."
                : $"Output {FormatCode(output.OnnxName)} shape: {FormatCode(FormatTensorShape(contract.Shape))}.";
        });

        return $$"""
        {{XmlDocumentation(
            "Runs the reconstructed TorchSharp graph.",
            inputParagraphs.Concat(outputParagraphs)
        )}}
        {{string.Join("\n", torchModule.Inputs.Select(input => XmlParamXml(input.ParameterName, $"Tensor value for ONNX input {FormatCode(input.OnnxName)}.")))}}
        {{XmlReturns(torchModule.Outputs.Length == 1
            ? $"The tensor value produced for ONNX output '{torchModule.Outputs[0].OnnxName}'."
            : "The tensor values produced for the ONNX graph outputs.")}}
        """;
    }

    private static string TensorProperty(
        string summary,
        ModelTensorContract tensor
    )
    {
        return XmlDocumentation(
            summary,
            TensorDocumentationParagraphs(tensor)
        );
    }

    private static IEnumerable<string> TensorDocumentationParagraphs(
        ModelTensorContract tensor
    )
    {
        yield return $"ONNX name: {FormatCode(tensor.OnnxName)}";
        yield return $"Tensor type: {FormatCode($"Tensor<{tensor.ElementClrTypeName}>")}";
        yield return $"Element type: {FormatCode(tensor.ElementClrTypeName)}";
        yield return $"Shape: {FormatCode(FormatTensorShape(tensor.Shape))}";

        if (!string.IsNullOrWhiteSpace(tensor.Denotation))
        {
            yield return $"Denotation: {FormatCode(tensor.Denotation!)}";
        }

        if (!tensor.IsRequired)
        {
            yield return tensor.HasDefaultInitializer
                ? "Optional input: pass null to omit this value and let the model use its initializer-backed default."
                : "Optional input: pass null to omit this ONNX input.";
        }
    }

    private static string MetadataParagraph(ModelTensorContract tensor)
    {
        var builder = new StringBuilder();
        builder.Append($"{FormatCode(tensor.OnnxName)}: {FormatCode($"Tensor<{tensor.ElementClrTypeName}>")}, shape {FormatCode(FormatTensorShape(tensor.Shape))}");

        if (!string.IsNullOrWhiteSpace(tensor.Denotation))
        {
            builder.Append($", denotation {FormatCode(tensor.Denotation!)}");
        }

        return builder.ToString();
    }

    private static string TensorCollectionParagraph(
        ModelTensorContract tensor,
        string roleLabel
    )
    {
        var builder = new StringBuilder();
        builder.Append($"{roleLabel} {FormatCode(tensor.PropertyName)} maps to ONNX name {FormatCode(tensor.OnnxName)}");
        builder.Append($"; tensor type {FormatCode($"Tensor<{tensor.ElementClrTypeName}>")}");
        builder.Append($"; shape {FormatCode(FormatTensorShape(tensor.Shape))}");

        if (!string.IsNullOrWhiteSpace(tensor.Denotation))
        {
            builder.Append($"; denotation {FormatCode(tensor.Denotation!)}");
        }

        return builder.ToString();
    }

    private static string TensorMethodParameterDescription(ModelTensorContract tensor)
    {
        var builder = new StringBuilder();
        builder.Append($"{(tensor.IsRequired ? "Tensor" : "Optional tensor")} value for model input {FormatCode(tensor.OnnxName)}");
        builder.Append($"; parameter type {FormatCode(InputTensorTypeName(tensor))}");
        builder.Append($"; shape {FormatCode(FormatTensorShape(tensor.Shape))}");

        if (!string.IsNullOrWhiteSpace(tensor.Denotation))
        {
            builder.Append($"; denotation {FormatCode(tensor.Denotation!)}");
        }

        if (!tensor.IsRequired)
        {
            builder.Append(tensor.HasDefaultInitializer
                ? "; pass null to omit this input and let the model use its initializer-backed default"
                : "; pass null to omit this optional ONNX input");
        }

        return builder.ToString();
    }

    private static string XmlDocumentation(
        string summary,
        IEnumerable<string> paragraphs
    )
    {
        var lines = new List<string>
        {
            "/// <summary>",
            $"/// {TextHelper.EscapeXml(summary)}",
        };

        foreach (var paragraph in paragraphs)
        {
            lines.Add($"/// <para>{paragraph}</para>");
        }

        lines.Add("/// </summary>");
        return string.Join("\n", lines);
    }

    private static string FormatTensorShape(ImmutableArray<ModelDimensionContract> shape)
    {
        if (shape.Length == 0)
        {
            return "[]";
        }

        var dimensions = shape.Select(static dimension =>
        {
            if (dimension.NumericValueLiteral is not null)
            {
                return dimension.NumericValueLiteral.EndsWith("L", StringComparison.Ordinal)
                    ? dimension.NumericValueLiteral.Substring(0, dimension.NumericValueLiteral.Length - 1)
                    : dimension.NumericValueLiteral;
            }

            if (dimension.SymbolicNameLiteral is not null)
            {
                return dimension.SymbolicNameLiteral.Trim('"');
            }

            return "?";
        });

        return $"[{string.Join(", ", dimensions)}]";
    }

    private static string InputTensorTypeName(ModelTensorContract tensor)
    {
        return tensor.IsRequired
            ? $"Tensor<{tensor.ElementClrTypeName}>"
            : $"Tensor<{tensor.ElementClrTypeName}>?";
    }
}
