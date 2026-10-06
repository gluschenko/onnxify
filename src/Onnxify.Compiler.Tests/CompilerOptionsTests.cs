using Onnxify.Compiler;

namespace Onnxify.Compiler.Tests;

public sealed class CompilerOptionsTests
{
    [Fact]
    public void SourceScannerIsSelectedByPriorityAndDiagnosticsAreObservedOnce()
    {
        var options = new CompilerOptions();
        var diagnostics = new List<CompilerDiagnostic>();
        var inspectedTrees = 0;
        options.DiagnosticCallback = diagnostics.Add;
        options.IntermediateRepresentationCallback = _ => inspectedTrees++;
        var scanner = new DecliningScanner(10);
        var selected = new ProducingScanner(5);
        options.Extensions.Add(scanner);
        options.Extensions.Add(selected);

        var result = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource("return input.relu();"), options);

        Assert.True(result.IsSuccess);
        Assert.Equal(1, scanner.Calls);
        Assert.Equal(1, selected.Calls);
        Assert.Single(diagnostics);
        Assert.Equal(CompilerDiagnosticSeverity.Warning, diagnostics[0].Severity);
        Assert.Equal(1, inspectedTrees);
    }

    [Fact]
    public void StrictModeTurnsExtensionExceptionIntoError()
    {
        var options = new CompilerOptions { ErrorMode = CompilerErrorMode.Strict };
        options.Extensions.Add(new ThrowingScanner(1));

        var result = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource("return input.relu();"), options);

        Assert.False(result.IsSuccess);
        Assert.Contains(result.Diagnostics, diagnostic => diagnostic.Severity == CompilerDiagnosticSeverity.Error);
    }

    [Fact]
    public void EqualHighestScannerPriorityReportsWarningAndUsesBuiltInScanner()
    {
        var options = new CompilerOptions();
        var diagnostics = new List<CompilerDiagnostic>();
        options.DiagnosticCallback = diagnostics.Add;
        var first = new ProducingScanner(4);
        var second = new ProducingScanner(4);
        options.Extensions.Add(first);
        options.Extensions.Add(second);

        var result = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource("return input.relu();"), options);

        Assert.True(result.IsSuccess);
        Assert.Equal(0, first.Calls);
        Assert.Equal(0, second.Calls);
        Assert.Contains(diagnostics, diagnostic => diagnostic.Code == CompilerDiagnosticCodes.Ambiguous
            && diagnostic.Severity == CompilerDiagnosticSeverity.Warning);
    }

    [Fact]
    public void ModuleExporterMetadataProviderAndOutputPrinterCanBeRegistered()
    {
        var options = new CompilerOptions();
        var metadataProvider = new MetadataProvider(3);
        var exporter = new ModuleExporter(2);
        var printer = new OutputPrinter(1);
        options.Extensions.Add(metadataProvider);
        options.Extensions.Add(exporter);
        options.Extensions.Add(printer);

        var source = new CSharpTorchSharpSource("return input.relu();");
        var metadata = Compiler.GetMetadata(source, options);
        var imported = Compiler.CreateTreeFromModule(new object(), options);
        Assert.True(imported.IsSuccess);
        var printed = Compiler.GenerateCSharp(imported.Value!, null, options);

        Assert.Equal(nameof(CompilerSourceKind.CSharpTorchSharp), metadata["source"]);
        Assert.True(printed.IsSuccess);
        Assert.Equal("extension printer", printed.Value);
        Assert.Equal(1, metadataProvider.Calls);
        Assert.Equal(1, exporter.Calls);
        Assert.Equal(1, printer.Calls);
    }

    [Fact]
    public void MethodLoweringCanProduceTheCompilerTree()
    {
        var source = new CSharpTorchSharpSource("return input.relu();");
        var tree = Compiler.CreateTreeFromTorchSharp(source);
        Assert.True(tree.IsSuccess);
        var lowering = new MethodLowering(1, tree.Value!);
        var options = new CompilerOptions();
        options.Extensions.Add(lowering);

        var lowered = Compiler.CreateTreeFromTorchSharp(source, options);

        Assert.True(lowered.IsSuccess);
        Assert.Same(tree.Value, lowered.Value);
        Assert.Equal(1, lowering.Calls);
    }

    private sealed class DecliningScanner(int priority) : ICompilerSourceScanner
    {
        public int Priority => priority;
        public int Calls { get; private set; }
        public CompilerResult<CompilerComputationTree>? Scan(ICompilerSource source)
        {
            Calls++;
            throw new InvalidOperationException("declined by exception");
        }
    }

    private sealed class ProducingScanner(int priority) : ICompilerSourceScanner
    {
        public int Priority => priority;
        public int Calls { get; private set; }
        public CompilerResult<CompilerComputationTree>? Scan(ICompilerSource source)
        {
            Calls++;
            return null;
        }
    }

    private sealed class ThrowingScanner(int priority) : ICompilerSourceScanner
    {
        public int Priority => priority;
        public CompilerResult<CompilerComputationTree>? Scan(ICompilerSource source) => throw new InvalidOperationException("strict failure");
    }

    private sealed class MetadataProvider(int priority) : ICompilerMetadataProvider
    {
        public int Priority => priority;
        public int Calls { get; private set; }
        public bool TryProvideMetadata(ICompilerSource source, out IReadOnlyDictionary<string, string> metadata)
        {
            Calls++;
            metadata = new Dictionary<string, string> { ["source"] = source.Kind.ToString() };
            return true;
        }
    }

    private sealed class ModuleExporter(int priority) : ICompilerModuleExporter
    {
        public int Priority => priority;
        public int Calls { get; private set; }
        public bool TryExport(object module, out ICompilerSource? source)
        {
            Calls++;
            source = new CSharpTorchSharpSource("return input.relu();");
            return true;
        }
    }

    private sealed class OutputPrinter(int priority) : ICompilerOutputPrinter
    {
        public int Priority => priority;
        public int Calls { get; private set; }
        public CompilerResult<object>? Print(CompilerComputationTree tree, CompilerTargetKind target)
        {
            Calls++;
            return CompilerResult<object>.Success("extension printer");
        }
    }

    private sealed class MethodLowering(int priority, CompilerComputationTree tree) : ICompilerMethodLowering
    {
        public int Priority => priority;
        public int Calls { get; private set; }
        public CompilerResult<CompilerComputationTree>? Lower(ICompilerSource source)
        {
            Calls++;
            return CompilerResult<CompilerComputationTree>.Success(tree);
        }
    }
}
