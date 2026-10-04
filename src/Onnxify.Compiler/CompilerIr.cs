namespace Onnxify.Compiler;

/// <summary>Identifies a compiler state member.</summary>
public enum CompilerStateMemberKind
{
    Parameter = 1,
    Buffer = 2,
    Initializer = 3,
}

/// <summary>Classifies the directions supported by an operator mapping.</summary>
public enum CompilerOperationCapability
{
    Bidirectional = 1,
    ExportOnly = 2,
    ImportOnly = 3,
    Unsupported = 0,
}

/// <summary>References a named compiler value, including an explicitly empty optional slot.</summary>
public sealed class CompilerValueReference : IEquatable<CompilerValueReference>
{
    public CompilerValueReference(string name, bool isEmptyOptional = false)
    {
        if (string.IsNullOrWhiteSpace(name) && !isEmptyOptional)
        {
            throw new ArgumentException("A value reference name is required unless the slot is explicitly empty optional.", nameof(name));
        }

        Name = name ?? string.Empty;
        IsEmptyOptional = isEmptyOptional;
    }

    public string Name { get; }

    public bool IsEmptyOptional { get; }

    public bool Equals(CompilerValueReference? other)
    {
        return other is not null
            && string.Equals(Name, other.Name, StringComparison.Ordinal)
            && IsEmptyOptional == other.IsEmptyOptional;
    }

    public override bool Equals(object? obj) => Equals(obj as CompilerValueReference);

    public override int GetHashCode() => CompilerStructural.Combine(17, Name, IsEmptyOptional);
}

/// <summary>Named graph value definition used by inputs, outputs, and intermediate values.</summary>
public sealed class CompilerValue : IEquatable<CompilerValue>
{
    public CompilerValue(
        string name,
        CompilerType type,
        CompilerSourceSpan? span = null
    )
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Value name cannot be empty.", nameof(name));
        }

        CompilerStructural.RequireNotNull(type, nameof(type));
        Name = name;
        Type = type;
        Span = span;
    }

    public string Name { get; }

    public CompilerType Type { get; }

    public CompilerSourceSpan? Span { get; }

    public bool Equals(CompilerValue? other)
    {
        return other is not null
            && string.Equals(Name, other.Name, StringComparison.Ordinal)
            && EqualityComparer<CompilerType>.Default.Equals(Type, other.Type)
            && EqualityComparer<CompilerSourceSpan?>.Default.Equals(Span, other.Span);
    }

    public override bool Equals(object? obj) => Equals(obj as CompilerValue);

    public override int GetHashCode() => CompilerStructural.Combine(17, Name, Type, Span);
}

/// <summary>Compiler-owned parameter, buffer, or initializer metadata.</summary>
public sealed class CompilerStateMember : IEquatable<CompilerStateMember>
{
    public CompilerStateMember(
        string name,
        CompilerStateMemberKind kind,
        CompilerType type,
        CompilerLiteral? value = null,
        CompilerSourceSpan? span = null
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
        Span = span;
    }

    public string Name { get; }

    public CompilerStateMemberKind Kind { get; }

    public CompilerType Type { get; }

    public CompilerLiteral? Value { get; }

    public CompilerSourceSpan? Span { get; }

    public bool Equals(CompilerStateMember? other)
    {
        return other is not null
            && string.Equals(Name, other.Name, StringComparison.Ordinal)
            && Kind == other.Kind
            && EqualityComparer<CompilerType>.Default.Equals(Type, other.Type)
            && EqualityComparer<CompilerLiteral?>.Default.Equals(Value, other.Value)
            && EqualityComparer<CompilerSourceSpan?>.Default.Equals(Span, other.Span);
    }

    public override bool Equals(object? obj) => Equals(obj as CompilerStateMember);

    public override int GetHashCode() => CompilerStructural.Combine(17, Name, Kind, Type, Value, Span);
}

/// <summary>Shared operator identity and normalized mapping metadata.</summary>
public sealed class CompilerOperatorDescriptor : IEquatable<CompilerOperatorDescriptor>
{
    public CompilerOperatorDescriptor(
        string name,
        string? domain = null,
        CompilerOperationCapability capability = CompilerOperationCapability.Unsupported,
        IEnumerable<string>? constraints = null
    )
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Operator name cannot be empty.", nameof(name));
        }

        Name = name;
        Domain = domain ?? string.Empty;
        Capability = capability;
        Constraints = CompilerStructural.Copy(constraints ?? Array.Empty<string>(), nameof(constraints));
    }

    public string Name { get; }

    public string Domain { get; }

    public CompilerOperationCapability Capability { get; }

    public IReadOnlyList<string> Constraints { get; }

    public bool Equals(CompilerOperatorDescriptor? other)
    {
        return other is not null
            && string.Equals(Name, other.Name, StringComparison.Ordinal)
            && string.Equals(Domain, other.Domain, StringComparison.Ordinal)
            && Capability == other.Capability
            && CompilerStructural.SequenceEqual(Constraints, other.Constraints);
    }

    public override bool Equals(object? obj) => Equals(obj as CompilerOperatorDescriptor);

    public override int GetHashCode() => CompilerStructural.Combine(17, Name, Domain, Capability, CompilerStructural.GetHashCode(Constraints));
}

/// <summary>Base class for ordered computation-tree steps.</summary>
public abstract class CompilerComputationStep : IEquatable<CompilerComputationStep>
{
    protected CompilerComputationStep(string name, CompilerSourceSpan? span = null)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Computation step name cannot be empty.", nameof(name));
        }

        Name = name;
        Span = span;
    }

    public string Name { get; }

    public CompilerSourceSpan? Span { get; }

    public bool Equals(CompilerComputationStep? other)
    {
        return other is not null
            && GetType() == other.GetType()
            && string.Equals(Name, other.Name, StringComparison.Ordinal)
            && EqualityComparer<CompilerSourceSpan?>.Default.Equals(Span, other.Span)
            && EqualsCore(other);
    }

    public override bool Equals(object? obj) => Equals(obj as CompilerComputationStep);

    public override int GetHashCode() => CompilerStructural.Combine(17, Name, Span, GetHashCodeCore());

    protected abstract bool EqualsCore(CompilerComputationStep other);

    protected abstract int GetHashCodeCore();
}

public sealed class CompilerOperation : CompilerComputationStep
{
    public CompilerOperation(
        string name,
        CompilerOperatorDescriptor descriptor,
        IEnumerable<CompilerValueReference> inputs,
        IEnumerable<CompilerValueReference> outputs,
        IEnumerable<CompilerAttribute>? attributes = null,
        CompilerSourceSpan? span = null
    ) : base(name, span)
    {
        CompilerStructural.RequireNotNull(descriptor, nameof(descriptor));
        Descriptor = descriptor;
        Inputs = CompilerStructural.Copy(inputs, nameof(inputs));
        Outputs = CompilerStructural.Copy(outputs, nameof(outputs));
        Attributes = CompilerStructural.Copy(attributes ?? Array.Empty<CompilerAttribute>(), nameof(attributes));
        EnsureUniqueAttributes(Attributes);
    }

    public CompilerOperatorDescriptor Descriptor { get; }

    public IReadOnlyList<CompilerValueReference> Inputs { get; }

    public IReadOnlyList<CompilerValueReference> Outputs { get; }

    public IReadOnlyList<CompilerAttribute> Attributes { get; }

    protected override bool EqualsCore(CompilerComputationStep other)
    {
        var operation = (CompilerOperation)other;
        return EqualityComparer<CompilerOperatorDescriptor>.Default.Equals(Descriptor, operation.Descriptor)
            && CompilerStructural.SequenceEqual(Inputs, operation.Inputs)
            && CompilerStructural.SequenceEqual(Outputs, operation.Outputs)
            && CompilerStructural.SequenceEqual(Attributes, operation.Attributes);
    }

    protected override int GetHashCodeCore()
    {
        return CompilerStructural.Combine(
            17,
            Descriptor,
            CompilerStructural.GetHashCode(Inputs),
            CompilerStructural.GetHashCode(Outputs),
            CompilerStructural.GetHashCode(Attributes));
    }

    private static void EnsureUniqueAttributes(IReadOnlyList<CompilerAttribute> attributes)
    {
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (var attribute in attributes)
        {
            if (!names.Add(attribute.Name))
            {
                throw new ArgumentException($"Duplicate operation attribute '{attribute.Name}'.", nameof(attributes));
            }
        }
    }
}

public sealed class CompilerModuleCall : CompilerComputationStep
{
    public CompilerModuleCall(
        string name,
        string targetBlock,
        IEnumerable<CompilerValueReference> inputs,
        IEnumerable<CompilerValueReference> outputs,
        CompilerSourceSpan? span = null
    ) : base(name, span)
    {
        if (string.IsNullOrWhiteSpace(targetBlock))
        {
            throw new ArgumentException("Target block name cannot be empty.", nameof(targetBlock));
        }

        TargetBlock = targetBlock;
        Inputs = CompilerStructural.Copy(inputs, nameof(inputs));
        Outputs = CompilerStructural.Copy(outputs, nameof(outputs));
    }

    public string TargetBlock { get; }

    public IReadOnlyList<CompilerValueReference> Inputs { get; }

    public IReadOnlyList<CompilerValueReference> Outputs { get; }

    protected override bool EqualsCore(CompilerComputationStep other)
    {
        var call = (CompilerModuleCall)other;
        return string.Equals(TargetBlock, call.TargetBlock, StringComparison.Ordinal)
            && CompilerStructural.SequenceEqual(Inputs, call.Inputs)
            && CompilerStructural.SequenceEqual(Outputs, call.Outputs);
    }

    protected override int GetHashCodeCore()
    {
        return CompilerStructural.Combine(
            17,
            TargetBlock,
            CompilerStructural.GetHashCode(Inputs),
            CompilerStructural.GetHashCode(Outputs));
    }
}

public sealed class CompilerComputationBlock : IEquatable<CompilerComputationBlock>
{
    public CompilerComputationBlock(
        string name,
        IEnumerable<CompilerValueReference> inputs,
        IEnumerable<CompilerValueReference> outputs,
        CompilerBlockStatement body,
        CompilerSourceSpan? span = null
    )
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Block name cannot be empty.", nameof(name));
        }

        CompilerStructural.RequireNotNull(body, nameof(body));
        Name = name;
        Inputs = CompilerStructural.Copy(inputs, nameof(inputs));
        Outputs = CompilerStructural.Copy(outputs, nameof(outputs));
        Body = body;
        Span = span;
        EnsureUniqueReferences(Inputs, nameof(inputs));
        EnsureUniqueReferences(Outputs, nameof(outputs));
    }

    public string Name { get; }

    public IReadOnlyList<CompilerValueReference> Inputs { get; }

    public IReadOnlyList<CompilerValueReference> Outputs { get; }

    public CompilerBlockStatement Body { get; }

    public CompilerSourceSpan? Span { get; }

    public bool Equals(CompilerComputationBlock? other)
    {
        return other is not null
            && string.Equals(Name, other.Name, StringComparison.Ordinal)
            && CompilerStructural.SequenceEqual(Inputs, other.Inputs)
            && CompilerStructural.SequenceEqual(Outputs, other.Outputs)
            && EqualityComparer<CompilerBlockStatement>.Default.Equals(Body, other.Body)
            && EqualityComparer<CompilerSourceSpan?>.Default.Equals(Span, other.Span);
    }

    public override bool Equals(object? obj) => Equals(obj as CompilerComputationBlock);

    public override int GetHashCode() => CompilerStructural.Combine(17, Name, CompilerStructural.GetHashCode(Inputs), CompilerStructural.GetHashCode(Outputs), Body, Span);

    private static void EnsureUniqueReferences(IReadOnlyList<CompilerValueReference> references, string parameterName)
    {
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (var reference in references)
        {
            var key = reference.IsEmptyOptional ? $"<empty:{names.Count}>" : reference.Name;
            if (!reference.IsEmptyOptional && !names.Add(key))
            {
                throw new ArgumentException($"Duplicate block value reference '{reference.Name}'.", parameterName);
            }
        }
    }
}

/// <summary>Immutable compiler-owned computation tree consumed by future frontends and backends.</summary>
public sealed class CompilerComputationTree : ICompilerTree, IEquatable<CompilerComputationTree>
{
    public CompilerComputationTree(
        string name,
        IEnumerable<CompilerValue> inputs,
        IEnumerable<CompilerValue> outputs,
        IEnumerable<CompilerValue> intermediateValues,
        IEnumerable<CompilerStateMember> parameters,
        IEnumerable<CompilerStateMember> buffers,
        IEnumerable<CompilerStateMember> initializers,
        IEnumerable<CompilerComputationStep> operations,
        IEnumerable<CompilerComputationBlock>? blocks = null,
        CompilerBlockStatement? syntaxBody = null,
        IEnumerable<KeyValuePair<string, string>>? metadata = null,
        IEnumerable<CompilerValue>? captures = null,
        CompilerModelEnvelope? modelEnvelope = null,
        IEnumerable<CompilerQuantizationAnnotation>? quantizationAnnotations = null,
        string? document = null
    )
    {
        Name = name ?? string.Empty;
        Inputs = CompilerStructural.Copy(inputs, nameof(inputs));
        Outputs = CompilerStructural.Copy(outputs, nameof(outputs));
        IntermediateValues = CompilerStructural.Copy(intermediateValues, nameof(intermediateValues));
        Parameters = CompilerStructural.Copy(parameters, nameof(parameters));
        Buffers = CompilerStructural.Copy(buffers, nameof(buffers));
        Initializers = CompilerStructural.Copy(initializers, nameof(initializers));
        Operations = CompilerStructural.Copy(operations, nameof(operations));
        Blocks = CompilerStructural.Copy(blocks ?? Array.Empty<CompilerComputationBlock>(), nameof(blocks));
        SyntaxBody = syntaxBody;
        Metadata = CompilerStructural.Copy(metadata ?? Array.Empty<KeyValuePair<string, string>>(), nameof(metadata));
        Captures = CompilerStructural.Copy(captures ?? Array.Empty<CompilerValue>(), nameof(captures));
        ModelEnvelope = modelEnvelope;
        QuantizationAnnotations = CompilerStructural.Copy(
            quantizationAnnotations ?? Array.Empty<CompilerQuantizationAnnotation>(),
            nameof(quantizationAnnotations));
        Document = document ?? string.Empty;

        ValidateDefinitions();
    }

    public string Name { get; }

    public IReadOnlyList<CompilerValue> Inputs { get; }

    public IReadOnlyList<CompilerValue> Outputs { get; }

    public IReadOnlyList<CompilerValue> IntermediateValues { get; }

    public IReadOnlyList<CompilerStateMember> Parameters { get; }

    public IReadOnlyList<CompilerStateMember> Buffers { get; }

    public IReadOnlyList<CompilerStateMember> Initializers { get; }

    public IReadOnlyList<CompilerComputationStep> Operations { get; }

    public IReadOnlyList<CompilerComputationBlock> Blocks { get; }

    public CompilerBlockStatement? SyntaxBody { get; }

    public IReadOnlyList<KeyValuePair<string, string>> Metadata { get; }

    /// <summary>Gets values captured from an enclosing graph scope.</summary>
    public IReadOnlyList<CompilerValue> Captures { get; }

    /// <summary>Gets optional model-level ONNX metadata.</summary>
    public CompilerModelEnvelope? ModelEnvelope { get; }

    /// <summary>Gets graph quantization annotations in source order.</summary>
    public IReadOnlyList<CompilerQuantizationAnnotation> QuantizationAnnotations { get; }

    /// <summary>Gets graph-level documentation independent from model-level documentation.</summary>
    public string Document { get; }

    public bool Equals(CompilerComputationTree? other)
    {
        return other is not null
            && string.Equals(Name, other.Name, StringComparison.Ordinal)
            && CompilerStructural.SequenceEqual(Inputs, other.Inputs)
            && CompilerStructural.SequenceEqual(Outputs, other.Outputs)
            && CompilerStructural.SequenceEqual(IntermediateValues, other.IntermediateValues)
            && CompilerStructural.SequenceEqual(Parameters, other.Parameters)
            && CompilerStructural.SequenceEqual(Buffers, other.Buffers)
            && CompilerStructural.SequenceEqual(Initializers, other.Initializers)
            && CompilerStructural.SequenceEqual(Operations, other.Operations)
            && CompilerStructural.SequenceEqual(Blocks, other.Blocks)
            && EqualityComparer<CompilerBlockStatement?>.Default.Equals(SyntaxBody, other.SyntaxBody)
            && CompilerStructural.SequenceEqual(Metadata, other.Metadata)
            && CompilerStructural.SequenceEqual(Captures, other.Captures)
            && EqualityComparer<CompilerModelEnvelope?>.Default.Equals(ModelEnvelope, other.ModelEnvelope)
            && CompilerStructural.SequenceEqual(QuantizationAnnotations, other.QuantizationAnnotations)
            && string.Equals(Document, other.Document, StringComparison.Ordinal);
    }

    public override bool Equals(object? obj) => Equals(obj as CompilerComputationTree);

    public override int GetHashCode()
    {
        return CompilerStructural.Combine(
            17,
            Name,
            CompilerStructural.GetHashCode(Inputs),
            CompilerStructural.GetHashCode(Outputs),
            CompilerStructural.GetHashCode(IntermediateValues),
            CompilerStructural.GetHashCode(Parameters),
            CompilerStructural.GetHashCode(Buffers),
            CompilerStructural.GetHashCode(Initializers),
            CompilerStructural.GetHashCode(Operations),
            CompilerStructural.GetHashCode(Blocks),
            SyntaxBody,
            CompilerStructural.GetHashCode(Metadata),
            CompilerStructural.GetHashCode(Captures),
            ModelEnvelope,
            CompilerStructural.GetHashCode(QuantizationAnnotations),
            Document);
    }

    private void ValidateDefinitions()
    {
        var values = new Dictionary<string, CompilerValue>(StringComparer.Ordinal);
        foreach (var value in Inputs.Concat(Outputs).Concat(IntermediateValues).Concat(Captures))
        {
            if (values.TryGetValue(value.Name, out var existing))
            {
                if (!EqualityComparer<CompilerType>.Default.Equals(existing.Type, value.Type))
                {
                    throw new ArgumentException($"Value '{value.Name}' has conflicting type definitions.");
                }

                continue;
            }

            values.Add(value.Name, value);
        }

        var stateMembers = new Dictionary<string, CompilerStateMember>(StringComparer.Ordinal);
        foreach (var member in Parameters.Concat(Buffers).Concat(Initializers))
        {
            if (stateMembers.ContainsKey(member.Name))
            {
                throw new ArgumentException($"Duplicate state member name '{member.Name}'.");
            }

            stateMembers.Add(member.Name, member);
        }

        foreach (var capture in Captures)
        {
            if (stateMembers.ContainsKey(capture.Name))
            {
                throw new ArgumentException($"Capture '{capture.Name}' conflicts with a state member.");
            }
        }

        EnsureUniqueNames(Operations.Select(x => x.Name), "operation");
        EnsureUniqueNames(Blocks.Select(x => x.Name), "block");

        var blockNames = new HashSet<string>(Blocks.Select(x => x.Name), StringComparer.Ordinal);
        foreach (var operation in Operations)
        {
            foreach (var reference in GetReferences(operation))
            {
                if (reference.IsEmptyOptional)
                {
                    continue;
                }

                if (!values.ContainsKey(reference.Name) && !stateMembers.ContainsKey(reference.Name))
                {
                    throw new ArgumentException($"Operation '{operation.Name}' references unknown value '{reference.Name}'.");
                }
            }

            if (operation is CompilerModuleCall call && !blockNames.Contains(call.TargetBlock))
            {
                throw new ArgumentException($"Module call '{call.Name}' references unknown block '{call.TargetBlock}'.");
            }
        }
    }

    private static IEnumerable<CompilerValueReference> GetReferences(CompilerComputationStep operation)
    {
        return operation switch
        {
            CompilerOperation node => node.Inputs.Concat(node.Outputs),
            CompilerModuleCall call => call.Inputs.Concat(call.Outputs),
            _ => Array.Empty<CompilerValueReference>(),
        };
    }

    private static void EnsureUniqueNames(IEnumerable<string> names, string kind)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var name in names)
        {
            if (!seen.Add(name))
            {
                throw new ArgumentException($"Duplicate {kind} name '{name}'.");
            }
        }
    }
}
