namespace Onnxify.Compiler.Operators;

[AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = false)]
internal sealed class CompilerTorchOperatorAttribute(string name) : Attribute
{
    public string Name { get; } = string.IsNullOrWhiteSpace(name)
        ? throw new ArgumentException("A Torch operator name is required.", nameof(name))
        : name;
}
