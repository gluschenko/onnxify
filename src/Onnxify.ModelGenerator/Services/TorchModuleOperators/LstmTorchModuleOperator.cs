using System.Collections.Immutable;
using static Onnxify.ModelGenerator.Helpers.TextHelper;
using static Onnxify.ModelGenerator.OnnxModelGenerator;

namespace Onnxify.ModelGenerator.Services.TorchModuleOperators;

[TorchSharpOp("LSTM")]
internal sealed class LstmTorchModuleOperator : TorchModuleOperator
{
    internal override string OnnxOpType => "LSTM";

    internal override bool TryCreateModuleNode(
        TorchNodeSpecification node,
        IReadOnlyDictionary<string, TorchInitializerSpecification> initializers,
        HashSet<string> usedFieldNames,
        out TorchModuleNodeSpecification module,
        out ImmutableArray<string> consumedInitializers
    )
    {
        module = null!;
        consumedInitializers = [];
        if (node.Inputs.Length < 3
            || !initializers.TryGetValue(node.Inputs[1], out var w)
            || !initializers.TryGetValue(node.Inputs[2], out var r)
            || w.ClrTypeName != "float"
            || r.ClrTypeName != "float"
            || w.Shape.Length != 3
            || r.Shape.Length != 3
            || w.Shape[0] != r.Shape[0]
            || w.Shape[1] != r.Shape[1]
            || r.Shape[1] != 4L * r.Shape[2])
        {
            return false;
        }

        var numDirections = w.Shape[0];
        var hiddenSize = r.Shape[2];
        var inputSize = w.Shape[2];
        if (numDirections is not (1L or 2L))
        {
            return false;
        }

        var direction = GetStringAttribute(node, "direction", "forward");
        var bidirectional = string.Equals(direction, "bidirectional", StringComparison.Ordinal);
        if (numDirections == 2L != bidirectional)
        {
            return false;
        }

        TorchInitializerSpecification? b = null;
        if (node.Inputs.Length > 3
            && initializers.TryGetValue(node.Inputs[3], out var foundB))
        {
            if (foundB.ClrTypeName != "float"
                || foundB.Shape.Length != 2
                || foundB.Shape[0] != numDirections
                || foundB.Shape[1] != 8L * hiddenSize)
            {
                return false;
            }

            b = foundB;
        }

        if (node.Inputs.Length > 4)
        {
            return false;
        }

        var fieldName = MakeUniqueIdentifier("_" + ToCamelIdentifier(node.Name, "lstm"), usedFieldNames, "_module");
        if (b is null)
        {
            return false;
        }

        module = new TorchModuleNodeSpecification(
            node.Name,
            TorchModuleNodeKind.LSTM,
            fieldName,
            "TorchModules.LSTM",
            $"LSTM({inputSize}, {hiddenSize}, numLayers: 1, bidirectional: {FormatBool(bidirectional)}, batchFirst: false)",
            TransposeInput: false,
            [
                $"LoadOnnxLstmWeights(tensors, \"{Escape(w.OnnxName)}\", \"{Escape(r.OnnxName)}\", {(b is null ? "null" : $"\"{Escape(b.OnnxName)}\"")}, {fieldName}, {hiddenSize}L, {inputSize}L, {numDirections}L);",
            ],
            $"ToOnnxLstmY({fieldName}.forward({{0}}).Item1, {numDirections}L, {hiddenSize}L)"
        );
        consumedInitializers = b is null ? [w.OnnxName, r.OnnxName] : [w.OnnxName, r.OnnxName, b.OnnxName];
        return true;
    }
}
