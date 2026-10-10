
namespace Onnxify.Compiler.Operators;

internal readonly struct CompilerOperatorIdentity : IEquatable<CompilerOperatorIdentity>
{
    public CompilerOperatorIdentity(string domain, string name)
    {
        Domain = domain ?? string.Empty;
        Name = name ?? string.Empty;
    }

    public string Domain { get; }
    public string Name { get; }
    public static CompilerOperatorIdentity Onnx(string name, string domain = "")
    {
        return new CompilerOperatorIdentity(domain, name);
    }

    public bool Equals(CompilerOperatorIdentity other)
    {
        return string.Equals(Domain, other.Domain, StringComparison.Ordinal)
            && string.Equals(Name, other.Name, StringComparison.Ordinal);
    }

    public override bool Equals(object? obj)
    {
        return obj is CompilerOperatorIdentity other && Equals(other);
    }

    public override int GetHashCode()
    {
        return CompilerStructural.Combine(17, Domain, Name);
    }
}

