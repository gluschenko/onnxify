namespace Onnxify.Compiler.Operators;

internal sealed class CompilerTorchSharpPrintContext(CompilerComputationTree tree)
{
    public bool IsValueUsed(string name) =>
        tree.Outputs.Any(output => string.Equals(output.Name, name, StringComparison.Ordinal))
        || tree.Operations.OfType<CompilerOnnxStep>().Any(operation => operation.Inputs.Any(input => string.Equals(input.Name, name, StringComparison.Ordinal)));

    public bool TryGetTensorInitializer(string name, out CompilerTensorLiteral? tensor)
    {
        var initializer = tree.Initializers.FirstOrDefault(member =>
            string.Equals(member.Name, name, StringComparison.Ordinal));
        tensor = initializer?.Value as CompilerTensorLiteral;
        return tensor is not null;
    }
}
