using Onnxify.Compiler.Operators;

namespace Onnxify.Compiler.Tests;

public sealed class CompilerOperatorRegistryTests
{
    [Fact]
    public void BuildOnnxIndexRejectsDuplicateIdentities()
    {
        var operators = new CompilerOperator[] { new RegistryTestOperator("Add", "torch.add"), new RegistryTestOperator("Add", "torch.add_alias") };

        var exception = Assert.Throws<InvalidOperationException>(() => CompilerOperatorRegistry.BuildOnnxIndex(operators));

        Assert.Contains("Duplicate compiler operator ONNX identity", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void BuildFormIndexRejectsDuplicateCallForms()
    {
        var operators = new CompilerOperator[] { new RegistryTestOperator("Add", "torch.add"), new RegistryTestOperator("Mul", "torch.add") };

        var exception = Assert.Throws<InvalidOperationException>(() =>
            CompilerOperatorRegistry.BuildFormIndex(operators, CompilerTorchSharpFormKind.Call));

        Assert.Contains("Duplicate compiler operator Call 'torch.add'", exception.Message, StringComparison.Ordinal);
    }

    private sealed class RegistryTestOperator(string name, string form) : CompilerOperator<Onnxify.Add>(CompilerOperatorIdentity.Onnx(name))
    {
        public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } = [CompilerTorchSharpForm.Call(form)];
        protected override string PrintTorchSharp(Onnxify.Add node, CompilerSourceSpan? span)
        {
            return node.Name;
        }
    }
}
