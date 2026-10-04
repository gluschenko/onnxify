namespace Onnxify.Compiler;

/// <summary>
/// Internal mutable construction helper used by future frontends and by compiler tests.
/// </summary>
internal sealed class CompilerComputationTreeBuilder
{
    private readonly List<CompilerValue> _inputs = [];
    private readonly List<CompilerValue> _outputs = [];
    private readonly List<CompilerValue> _intermediateValues = [];
    private readonly List<CompilerStateMember> _parameters = [];
    private readonly List<CompilerStateMember> _buffers = [];
    private readonly List<CompilerStateMember> _initializers = [];
    private readonly List<CompilerComputationStep> _operations = [];
    private readonly List<CompilerComputationBlock> _blocks = [];
    private readonly List<KeyValuePair<string, string>> _metadata = [];
    private readonly HashSet<string> _operationNames = new(StringComparer.Ordinal);
    private readonly HashSet<string> _blockNames = new(StringComparer.Ordinal);
    private CompilerBlockStatement? _syntaxBody;

    public CompilerComputationTreeBuilder(string? name = null)
    {
        Name = name ?? string.Empty;
    }

    public string Name { get; }

    public void AddInput(CompilerValue value) => AddUniqueValue(_inputs, value, "input");

    public void AddOutput(CompilerValue value) => AddUniqueValue(_outputs, value, "output");

    public void AddIntermediateValue(CompilerValue value) => AddUniqueValue(_intermediateValues, value, "intermediate value");

    public void AddStateMember(CompilerStateMember member)
    {
        CompilerStructural.RequireNotNull(member, nameof(member));

        var target = member.Kind switch
        {
            CompilerStateMemberKind.Parameter => _parameters,
            CompilerStateMemberKind.Buffer => _buffers,
            CompilerStateMemberKind.Initializer => _initializers,
            _ => throw new ArgumentOutOfRangeException(nameof(member)),
        };

        if (target.Any(x => string.Equals(x.Name, member.Name, StringComparison.Ordinal)))
        {
            throw new ArgumentException($"Duplicate {member.Kind} name '{member.Name}'.", nameof(member));
        }

        target.Add(member);
    }

    public void AddOperation(CompilerComputationStep operation)
    {
        CompilerStructural.RequireNotNull(operation, nameof(operation));
        if (!_operationNames.Add(operation.Name))
        {
            throw new ArgumentException($"Duplicate operation name '{operation.Name}'.", nameof(operation));
        }

        foreach (var reference in GetReferences(operation))
        {
            if (!reference.IsEmptyOptional && !IsKnownReference(reference.Name))
            {
                throw new ArgumentException($"Operation '{operation.Name}' references unknown value '{reference.Name}'.", nameof(operation));
            }
        }

        _operations.Add(operation);
    }

    public void AddBlock(CompilerComputationBlock block)
    {
        CompilerStructural.RequireNotNull(block, nameof(block));
        if (!_blockNames.Add(block.Name))
        {
            throw new ArgumentException($"Duplicate block name '{block.Name}'.", nameof(block));
        }

        _blocks.Add(block);
    }

    public void SetSyntaxBody(CompilerBlockStatement body)
    {
        CompilerStructural.RequireNotNull(body, nameof(body));
        _syntaxBody = body;
    }

    public void AddMetadata(string key, string value)
    {
        if (string.IsNullOrWhiteSpace(key))
        {
            throw new ArgumentException("Metadata key cannot be empty.", nameof(key));
        }

        if (_metadata.Any(x => string.Equals(x.Key, key, StringComparison.Ordinal)))
        {
            throw new ArgumentException($"Duplicate metadata key '{key}'.", nameof(key));
        }

        _metadata.Add(new KeyValuePair<string, string>(key, value ?? string.Empty));
    }

    public CompilerComputationTree Build()
    {
        return new CompilerComputationTree(
            Name,
            _inputs,
            _outputs,
            _intermediateValues,
            _parameters,
            _buffers,
            _initializers,
            _operations,
            _blocks,
            _syntaxBody,
            _metadata);
    }

    private bool IsKnownReference(string name)
    {
        return _inputs.Concat(_outputs).Concat(_intermediateValues).Any(x => string.Equals(x.Name, name, StringComparison.Ordinal))
            || _parameters.Concat(_buffers).Concat(_initializers).Any(x => string.Equals(x.Name, name, StringComparison.Ordinal));
    }

    private static void AddUniqueValue(List<CompilerValue> target, CompilerValue value, string kind)
    {
        CompilerStructural.RequireNotNull(value, nameof(value));
        if (target.Any(x => string.Equals(x.Name, value.Name, StringComparison.Ordinal)))
        {
            throw new ArgumentException($"Duplicate {kind} name '{value.Name}'.", nameof(value));
        }

        target.Add(value);
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
}
