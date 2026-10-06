namespace Onnxify.Compiler;

/// <summary>Controls how compiler extension failures are handled.</summary>
public enum CompilerErrorMode
{
    Permissive = 0,
    Strict = 1,
}

/// <summary>Common options and observation hooks for compiler operations.</summary>
public sealed class CompilerOptions
{
    public CompilerErrorMode ErrorMode { get; set; } = CompilerErrorMode.Permissive;

    public CompilerExtensionCollection Extensions { get; } = new();

    public Action<CompilerDiagnostic>? DiagnosticCallback { get; set; }

    public Action<CompilerComputationTree>? IntermediateRepresentationCallback { get; set; }

    public bool EnableRuntimeReflection { get; set; } = true;

    internal void Report(IEnumerable<CompilerDiagnostic> diagnostics)
    {
        if (DiagnosticCallback is null)
        {
            return;
        }

        foreach (var diagnostic in diagnostics)
        {
            DiagnosticCallback(diagnostic);
        }
    }

    internal CompilerResult<T> Observe<T>(CompilerResult<T> result)
    {
        Report(result.Diagnostics);
        if (result.Value is CompilerComputationTree tree)
        {
            IntermediateRepresentationCallback?.Invoke(tree);
        }

        return result;
    }
}

/// <summary>Common priority for compiler extension handlers.</summary>
public interface ICompilerExtension
{
    int Priority { get; }
}

/// <summary>Provides metadata for compiler sources and values.</summary>
public interface ICompilerMetadataProvider : ICompilerExtension
{
    bool TryProvideMetadata(ICompilerSource source, out IReadOnlyDictionary<string, string> metadata);
}

/// <summary>Exports a consumer module into a compiler-neutral source.</summary>
public interface ICompilerModuleExporter : ICompilerExtension
{
    bool TryExport(object module, out ICompilerSource? source);
}

/// <summary>Scans a compiler source into the shared computation tree.</summary>
public interface ICompilerSourceScanner : ICompilerExtension
{
    CompilerResult<CompilerComputationTree>? Scan(ICompilerSource source);
}

/// <summary>Prints a computation tree to a compiler target.</summary>
public interface ICompilerOutputPrinter : ICompilerExtension
{
    CompilerResult<object>? Print(CompilerComputationTree tree, CompilerTargetKind target);
}

/// <summary>Lowers a source method into a compiler computation tree.</summary>
public interface ICompilerMethodLowering : ICompilerExtension
{
    CompilerResult<CompilerComputationTree>? Lower(ICompilerSource source);
}

/// <summary>Mutable registration collection for compiler extension points.</summary>
public sealed class CompilerExtensionCollection
{
    private readonly List<ICompilerExtension> _extensions = new();

    public void Add(ICompilerExtension extension)
    {
        CompilerStructural.RequireNotNull(extension, nameof(extension));
        _extensions.Add(extension);
    }

    public bool Remove(ICompilerExtension extension) => _extensions.Remove(extension);

    public IReadOnlyList<T> Get<T>() where T : class, ICompilerExtension
    {
        return _extensions.OfType<T>().OrderByDescending(static extension => extension.Priority).ToArray();
    }
}
