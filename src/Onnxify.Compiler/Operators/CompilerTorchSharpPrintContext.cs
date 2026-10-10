namespace Onnxify.Compiler.Operators;

internal sealed class CompilerTorchSharpPrintContext(CompilerComputationTree tree)
{
    public bool TryGetTensorInitializer(string name, out CompilerTensorLiteral? tensor)
    {
        var initializer = tree.Initializers.FirstOrDefault(member =>
            string.Equals(member.Name, name, StringComparison.Ordinal));
        tensor = initializer?.Value as CompilerTensorLiteral;
        return tensor is not null;
    }
}
