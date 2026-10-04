namespace Onnxify.Compiler;

/// <summary>Describes a compiler-visible C# method value without referencing TorchSharp.</summary>
public sealed class CompilerTorchSharpValueDescriptor : IEquatable<CompilerTorchSharpValueDescriptor>
{
    public CompilerTorchSharpValueDescriptor(
        string name,
        CompilerType type,
        string? csharpTypeName = null
    )
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Value name cannot be empty.", nameof(name));
        }

        CompilerStructural.RequireNotNull(type, nameof(type));
        Name = name;
        Type = type;
        CSharpTypeName = string.IsNullOrWhiteSpace(csharpTypeName)
            ? "global::TorchSharp.torch.Tensor"
            : csharpTypeName!;
    }

    public string Name { get; }

    public CompilerType Type { get; }

    public string CSharpTypeName { get; }

    public bool Equals(CompilerTorchSharpValueDescriptor? other)
    {
        return other is not null
            && string.Equals(Name, other.Name, StringComparison.Ordinal)
            && EqualityComparer<CompilerType>.Default.Equals(Type, other.Type)
            && string.Equals(CSharpTypeName, other.CSharpTypeName, StringComparison.Ordinal);
    }

    public override bool Equals(object? obj) => Equals(obj as CompilerTorchSharpValueDescriptor);

    public override int GetHashCode() => CompilerStructural.Combine(17, Name, Type, CSharpTypeName);
}

/// <summary>Describes a compiler-visible state member without a runtime object.</summary>
public sealed class CompilerTorchSharpStateMemberDescriptor : IEquatable<CompilerTorchSharpStateMemberDescriptor>
{
    public CompilerTorchSharpStateMemberDescriptor(
        string name,
        CompilerStateMemberKind kind,
        CompilerType type,
        CompilerLiteral? value = null,
        string? csharpTypeName = null
    )
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("State member name cannot be empty.", nameof(name));
        }

        CompilerStructural.RequireNotNull(type, nameof(type));
        if (kind == CompilerStateMemberKind.Initializer && value is null)
        {
            throw new ArgumentNullException(nameof(value), "Initializers require a literal value.");
        }

        Name = name;
        Kind = kind;
        Type = type;
        Value = value;
        CSharpTypeName = string.IsNullOrWhiteSpace(csharpTypeName)
            ? "global::TorchSharp.torch.Tensor"
            : csharpTypeName!;
    }

    public string Name { get; }

    public CompilerStateMemberKind Kind { get; }

    public CompilerType Type { get; }

    public CompilerLiteral? Value { get; }

    public string CSharpTypeName { get; }

    public bool Equals(CompilerTorchSharpStateMemberDescriptor? other)
    {
        return other is not null
            && string.Equals(Name, other.Name, StringComparison.Ordinal)
            && Kind == other.Kind
            && EqualityComparer<CompilerType>.Default.Equals(Type, other.Type)
            && EqualityComparer<CompilerLiteral?>.Default.Equals(Value, other.Value)
            && string.Equals(CSharpTypeName, other.CSharpTypeName, StringComparison.Ordinal);
    }

    public override bool Equals(object? obj) => Equals(obj as CompilerTorchSharpStateMemberDescriptor);

    public override int GetHashCode() => CompilerStructural.Combine(17, Name, Kind, Type, Value, CSharpTypeName);
}

/// <summary>Describes a child module or reusable block in a compiler-neutral way.</summary>
public sealed class CompilerTorchSharpChildModuleDescriptor : IEquatable<CompilerTorchSharpChildModuleDescriptor>
{
    public CompilerTorchSharpChildModuleDescriptor(
        string name,
        string typeName,
        string? blockName = null
    )
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Child module name cannot be empty.", nameof(name));
        }

        if (string.IsNullOrWhiteSpace(typeName))
        {
            throw new ArgumentException("Child module type name cannot be empty.", nameof(typeName));
        }

        Name = name;
        TypeName = typeName;
        BlockName = string.IsNullOrWhiteSpace(blockName) ? name : blockName!;
    }

    public string Name { get; }

    public string TypeName { get; }

    public string BlockName { get; }

    public bool Equals(CompilerTorchSharpChildModuleDescriptor? other)
    {
        return other is not null
            && string.Equals(Name, other.Name, StringComparison.Ordinal)
            && string.Equals(TypeName, other.TypeName, StringComparison.Ordinal)
            && string.Equals(BlockName, other.BlockName, StringComparison.Ordinal);
    }

    public override bool Equals(object? obj) => Equals(obj as CompilerTorchSharpChildModuleDescriptor);

    public override int GetHashCode() => CompilerStructural.Combine(17, Name, TypeName, BlockName);
}

/// <summary>Describes a helper method that may be emitted as a reusable compiler block.</summary>
public sealed class CompilerTorchSharpHelperMethodDescriptor : IEquatable<CompilerTorchSharpHelperMethodDescriptor>
{
    public CompilerTorchSharpHelperMethodDescriptor(
        string name,
        IEnumerable<CompilerTorchSharpValueDescriptor>? inputs = null,
        IEnumerable<CompilerTorchSharpValueDescriptor>? outputs = null
    )
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Helper method name cannot be empty.", nameof(name));
        }

        Name = name;
        Inputs = CompilerStructural.Copy(inputs ?? Array.Empty<CompilerTorchSharpValueDescriptor>(), nameof(inputs));
        Outputs = CompilerStructural.Copy(outputs ?? Array.Empty<CompilerTorchSharpValueDescriptor>(), nameof(outputs));
    }

    public string Name { get; }

    public IReadOnlyList<CompilerTorchSharpValueDescriptor> Inputs { get; }

    public IReadOnlyList<CompilerTorchSharpValueDescriptor> Outputs { get; }

    public bool Equals(CompilerTorchSharpHelperMethodDescriptor? other)
    {
        return other is not null
            && string.Equals(Name, other.Name, StringComparison.Ordinal)
            && CompilerStructural.SequenceEqual(Inputs, other.Inputs)
            && CompilerStructural.SequenceEqual(Outputs, other.Outputs);
    }

    public override bool Equals(object? obj) => Equals(obj as CompilerTorchSharpHelperMethodDescriptor);

    public override int GetHashCode() => CompilerStructural.Combine(
        17,
        Name,
        CompilerStructural.GetHashCode(Inputs),
        CompilerStructural.GetHashCode(Outputs));
}

/// <summary>
/// Neutral description of a compiled C# TorchSharp module. It contains only paths, names,
/// contracts, and compiler-owned metadata, so the compiler package does not depend on TorchSharp.
/// </summary>
public sealed class CompilerTorchSharpModuleDescriptor : IEquatable<CompilerTorchSharpModuleDescriptor>
{
    public CompilerTorchSharpModuleDescriptor(
        string assemblyPath,
        string typeName,
        string methodName = "forward",
        int? methodMetadataToken = null,
        string? document = null,
        IEnumerable<CompilerTorchSharpValueDescriptor>? inputs = null,
        IEnumerable<CompilerTorchSharpValueDescriptor>? outputs = null,
        IEnumerable<CompilerTorchSharpStateMemberDescriptor>? stateMembers = null,
        IEnumerable<CompilerTorchSharpChildModuleDescriptor>? childModules = null,
        IEnumerable<CompilerTorchSharpHelperMethodDescriptor>? helperMethods = null
    )
    {
        if (string.IsNullOrWhiteSpace(assemblyPath))
        {
            throw new ArgumentException("Assembly path cannot be empty.", nameof(assemblyPath));
        }

        if (string.IsNullOrWhiteSpace(typeName))
        {
            throw new ArgumentException("Type name cannot be empty.", nameof(typeName));
        }

        if (string.IsNullOrWhiteSpace(methodName))
        {
            throw new ArgumentException("Method name cannot be empty.", nameof(methodName));
        }

        AssemblyPath = Path.GetFullPath(assemblyPath);
        TypeName = typeName;
        MethodName = methodName;
        MethodMetadataToken = methodMetadataToken;
        Document = string.IsNullOrWhiteSpace(document) ? AssemblyPath : document!;
        Inputs = CompilerStructural.Copy(inputs ?? Array.Empty<CompilerTorchSharpValueDescriptor>(), nameof(inputs));
        Outputs = CompilerStructural.Copy(outputs ?? Array.Empty<CompilerTorchSharpValueDescriptor>(), nameof(outputs));
        StateMembers = CompilerStructural.Copy(stateMembers ?? Array.Empty<CompilerTorchSharpStateMemberDescriptor>(), nameof(stateMembers));
        ChildModules = CompilerStructural.Copy(childModules ?? Array.Empty<CompilerTorchSharpChildModuleDescriptor>(), nameof(childModules));
        HelperMethods = CompilerStructural.Copy(helperMethods ?? Array.Empty<CompilerTorchSharpHelperMethodDescriptor>(), nameof(helperMethods));
    }

    public string AssemblyPath { get; }

    public string TypeName { get; }

    public string MethodName { get; }

    public int? MethodMetadataToken { get; }

    public string Document { get; }

    public IReadOnlyList<CompilerTorchSharpValueDescriptor> Inputs { get; }

    public IReadOnlyList<CompilerTorchSharpValueDescriptor> Outputs { get; }

    public IReadOnlyList<CompilerTorchSharpStateMemberDescriptor> StateMembers { get; }

    public IReadOnlyList<CompilerTorchSharpChildModuleDescriptor> ChildModules { get; }

    public IReadOnlyList<CompilerTorchSharpHelperMethodDescriptor> HelperMethods { get; }

    public bool Equals(CompilerTorchSharpModuleDescriptor? other)
    {
        return other is not null
            && string.Equals(AssemblyPath, other.AssemblyPath, StringComparison.Ordinal)
            && string.Equals(TypeName, other.TypeName, StringComparison.Ordinal)
            && string.Equals(MethodName, other.MethodName, StringComparison.Ordinal)
            && MethodMetadataToken == other.MethodMetadataToken
            && string.Equals(Document, other.Document, StringComparison.Ordinal)
            && CompilerStructural.SequenceEqual(Inputs, other.Inputs)
            && CompilerStructural.SequenceEqual(Outputs, other.Outputs)
            && CompilerStructural.SequenceEqual(StateMembers, other.StateMembers)
            && CompilerStructural.SequenceEqual(ChildModules, other.ChildModules)
            && CompilerStructural.SequenceEqual(HelperMethods, other.HelperMethods);
    }

    public override bool Equals(object? obj) => Equals(obj as CompilerTorchSharpModuleDescriptor);

    public override int GetHashCode() => CompilerStructural.Combine(
        17,
        AssemblyPath,
        TypeName,
        MethodName,
        MethodMetadataToken,
        Document,
        CompilerStructural.GetHashCode(Inputs),
        CompilerStructural.GetHashCode(Outputs),
        CompilerStructural.GetHashCode(StateMembers),
        CompilerStructural.GetHashCode(ChildModules),
        CompilerStructural.GetHashCode(HelperMethods));
}

/// <summary>Options controlling generated C# TorchSharp source.</summary>
public sealed class CompilerCSharpGenerationOptions
{
    public string Namespace { get; set; } = "Onnxify.Generated";

    public string ClassName { get; set; } = "GeneratedTorchModule";

    public string ModuleName { get; set; } = "GeneratedTorchModule";

    public bool IncludeNullableContext { get; set; } = true;
}
