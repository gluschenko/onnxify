
namespace Onnxify.Compiler.Operators;

internal sealed class CompilerTorchSharpForm
{
    public CompilerTorchSharpForm(CompilerTorchSharpFormKind kind, string name)
    {
        Kind = kind;
        Name = name;
    }

    public CompilerTorchSharpFormKind Kind { get; }
    public string Name { get; }
    public static CompilerTorchSharpForm Call(string name)
    {
        return new CompilerTorchSharpForm(CompilerTorchSharpFormKind.Call, name);
    }

    public static CompilerTorchSharpForm Binary(string token)
    {
        return new CompilerTorchSharpForm(CompilerTorchSharpFormKind.Binary, token);
    }

    public static CompilerTorchSharpForm Unary(string token)
    {
        return new CompilerTorchSharpForm(CompilerTorchSharpFormKind.Unary, token);
    }
}

