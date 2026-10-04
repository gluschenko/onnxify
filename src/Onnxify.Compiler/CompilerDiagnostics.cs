namespace Onnxify.Compiler;

/// <summary>Identifies the phase that produced a compiler diagnostic.</summary>
public enum CompilerDiagnosticStage
{
    Parse = 1,
    Normalize = 2,
    Analyze = 3,
    Lower = 4,
    Emit = 5,
    Validate = 6,
}

/// <summary>Identifies the impact of a compiler diagnostic.</summary>
public enum CompilerDiagnosticSeverity
{
    Info = 0,
    Warning = 1,
    Error = 2,
}

/// <summary>Identifies the source representation associated with a source span.</summary>
public enum CompilerSourceSpanKind
{
    Unknown = 0,
    CSharp = 1,
    Onnx = 2,
}

/// <summary>Stable diagnostic codes used by compiler phases.</summary>
public static class CompilerDiagnosticCodes
{
    public const string Unsupported = "COMPILER_UNSUPPORTED";
    public const string Ambiguous = "COMPILER_AMBIGUOUS";
    public const string Lossy = "COMPILER_LOSSY";
    public const string DuplicateName = "COMPILER_DUPLICATE_NAME";
    public const string MissingReference = "COMPILER_MISSING_REFERENCE";
}

/// <summary>
/// Describes a location in either C# source or an ONNX graph without retaining a raw syntax or protobuf object.
/// </summary>
public sealed class CompilerSourceSpan : IEquatable<CompilerSourceSpan>
{
    public CompilerSourceSpan(
        CompilerSourceSpanKind kind,
        string document,
        int start,
        int length,
        int startLine,
        int startColumn,
        int endLine,
        int endColumn,
        string? nodeName = null,
        string? operatorName = null
    )
    {
        if (start < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(start));
        }

        if (length < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(length));
        }

        if (startLine < 0 || startColumn < 0 || endLine < 0 || endColumn < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(startLine), "Line and column values cannot be negative.");
        }

        Kind = kind;
        Document = document ?? string.Empty;
        Start = start;
        Length = length;
        StartLine = startLine;
        StartColumn = startColumn;
        EndLine = endLine;
        EndColumn = endColumn;
        NodeName = nodeName;
        OperatorName = operatorName;
    }

    public CompilerSourceSpanKind Kind { get; }

    public string Document { get; }

    public int Start { get; }

    public int Length { get; }

    public int StartLine { get; }

    public int StartColumn { get; }

    public int EndLine { get; }

    public int EndColumn { get; }

    public string? NodeName { get; }

    public string? OperatorName { get; }

    public bool Equals(CompilerSourceSpan? other)
    {
        return other is not null
            && Kind == other.Kind
            && string.Equals(Document, other.Document, StringComparison.Ordinal)
            && Start == other.Start
            && Length == other.Length
            && StartLine == other.StartLine
            && StartColumn == other.StartColumn
            && EndLine == other.EndLine
            && EndColumn == other.EndColumn
            && string.Equals(NodeName, other.NodeName, StringComparison.Ordinal)
            && string.Equals(OperatorName, other.OperatorName, StringComparison.Ordinal);
    }

    public override bool Equals(object? obj) => Equals(obj as CompilerSourceSpan);

    public override int GetHashCode()
    {
        return CompilerStructural.Combine(
            17,
            Kind,
            Document,
            Start,
            Length,
            StartLine,
            StartColumn,
            EndLine,
            EndColumn,
            NodeName,
            OperatorName);
    }
}

/// <summary>Optional caller/callee context attached to a compiler diagnostic.</summary>
public sealed class CompilerDiagnosticContext : IEquatable<CompilerDiagnosticContext>
{
    public CompilerDiagnosticContext(string? caller = null, string? callee = null)
    {
        Caller = caller;
        Callee = callee;
    }

    public string? Caller { get; }

    public string? Callee { get; }

    public bool Equals(CompilerDiagnosticContext? other)
    {
        return other is not null
            && string.Equals(Caller, other.Caller, StringComparison.Ordinal)
            && string.Equals(Callee, other.Callee, StringComparison.Ordinal);
    }

    public override bool Equals(object? obj) => Equals(obj as CompilerDiagnosticContext);

    public override int GetHashCode() => CompilerStructural.Combine(17, Caller, Callee);
}

/// <summary>One structured compiler diagnostic.</summary>
public sealed class CompilerDiagnostic : IEquatable<CompilerDiagnostic>
{
    public CompilerDiagnostic(
        string code,
        string message,
        CompilerDiagnosticStage stage,
        CompilerDiagnosticSeverity severity,
        CompilerSourceSpan? span = null,
        CompilerDiagnosticContext? context = null
    )
    {
        if (string.IsNullOrWhiteSpace(code))
        {
            throw new ArgumentException("Diagnostic code cannot be empty.", nameof(code));
        }

        if (string.IsNullOrWhiteSpace(message))
        {
            throw new ArgumentException("Diagnostic message cannot be empty.", nameof(message));
        }

        Code = code;
        Message = message;
        Stage = stage;
        Severity = severity;
        Span = span;
        Context = context;
    }

    public string Code { get; }

    public string Message { get; }

    public CompilerDiagnosticStage Stage { get; }

    public CompilerDiagnosticSeverity Severity { get; }

    public CompilerSourceSpan? Span { get; }

    public CompilerDiagnosticContext? Context { get; }

    public bool Equals(CompilerDiagnostic? other)
    {
        return other is not null
            && string.Equals(Code, other.Code, StringComparison.Ordinal)
            && string.Equals(Message, other.Message, StringComparison.Ordinal)
            && Stage == other.Stage
            && Severity == other.Severity
            && EqualityComparer<CompilerSourceSpan?>.Default.Equals(Span, other.Span)
            && EqualityComparer<CompilerDiagnosticContext?>.Default.Equals(Context, other.Context);
    }

    public override bool Equals(object? obj) => Equals(obj as CompilerDiagnostic);

    public override int GetHashCode() => CompilerStructural.Combine(17, Code, Message, Stage, Severity, Span, Context);
}

/// <summary>Immutable value-plus-diagnostics result returned by compiler operations.</summary>
public sealed class CompilerResult<T>
{
    public CompilerResult(T? value, IEnumerable<CompilerDiagnostic>? diagnostics = null)
    {
        Value = value;
        Diagnostics = CompilerStructural.Copy(
            diagnostics ?? Array.Empty<CompilerDiagnostic>(),
            nameof(diagnostics));
    }

    public T? Value { get; }

    public IReadOnlyList<CompilerDiagnostic> Diagnostics { get; }

    public bool HasErrors => Diagnostics.Any(x => x.Severity == CompilerDiagnosticSeverity.Error);

    public bool IsSuccess => !HasErrors;

    public static CompilerResult<T> Success(T value, IEnumerable<CompilerDiagnostic>? diagnostics = null)
    {
        CompilerStructural.RequireNotNull(value, nameof(value));
        return new CompilerResult<T>(value, diagnostics);
    }

    public static CompilerResult<T> Failure(
        IEnumerable<CompilerDiagnostic> diagnostics,
        T? value = default
    )
    {
        CompilerStructural.RequireNotNull(diagnostics, nameof(diagnostics));
        return new CompilerResult<T>(value, diagnostics);
    }
}
