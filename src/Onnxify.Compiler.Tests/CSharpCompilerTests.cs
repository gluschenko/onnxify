using Onnxify.Compiler;
using Onnxify.TorchSharp;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using System.Runtime.Loader;

namespace Onnxify.Compiler.Tests;

public sealed class CSharpCompilerTests
{
    [Fact]
    public void Source_text_is_scanned_into_immutable_syntax_ir()
    {
        var source = new CSharpTorchSharpSource(
            """
            public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input)
            {
                var values = new[] { 1, 2 };
                var selected = values[0];
                if (true)
                {
                    selected = selected + 1;
                }
                foreach (var item in values)
                {
                    selected = selected + item;
                }
                return input;
            }
            """);

        var first = Compiler.CreateTreeFromTorchSharp(source);
        var second = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source.SourceText));

        Assert.True(first.IsSuccess, string.Join(Environment.NewLine, first.Diagnostics.Select(x => x.Message)));
        Assert.True(second.IsSuccess, string.Join(Environment.NewLine, second.Diagnostics.Select(x => x.Message)));
        Assert.Equal(first.Value, second.Value);
        Assert.NotNull(first.Value!.SyntaxBody);
        Assert.Equal(2, first.Value.IntermediateValues.Count);
        Assert.Contains(first.Value.SyntaxBody!.Statements, x => x is CompilerStaticIfStatement);
        Assert.Contains(first.Value.SyntaxBody.Statements, x => x is CompilerStaticForeachStatement);
    }

    [Fact]
    public void Backend_generates_torchsharp_source_from_syntax_tree()
    {
        var treeResult = Compiler.CreateTreeFromTorchSharp(
            new CSharpTorchSharpSource("return input + 1f;"));

        Assert.True(treeResult.IsSuccess);
        var generated = Compiler.GenerateCSharp(
            treeResult.Value!,
            new CompilerCSharpGenerationOptions
            {
                Namespace = "Generated.Tests",
                ClassName = "IdentityLikeModule",
                ModuleName = "IdentityLikeModule",
            });

        Assert.True(generated.IsSuccess, string.Join(Environment.NewLine, generated.Diagnostics.Select(x => x.Message)));
        Assert.Contains("torch.nn.Module", generated.Value);
        Assert.Contains("public override", generated.Value);
        Assert.Contains("return (input + 1d);", generated.Value);
    }

    [Fact]
    public void Generated_source_is_compilable_csharp()
    {
        var treeResult = Compiler.CreateTreeFromTorchSharp(
            new CSharpTorchSharpSource("return input;"));
        Assert.True(treeResult.IsSuccess);

        var generated = Compiler.GenerateCSharp(treeResult.Value!);
        Assert.True(generated.IsSuccess);

        var compilation = CSharpCompilation.Create(
            "Onnxify.Generated.CompilerTest",
            [CSharpSyntaxTree.ParseText(generated.Value!)],
            GetCompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var output = new MemoryStream();
        var emit = compilation.Emit(output);

        Assert.True(
            emit.Success,
            string.Join(Environment.NewLine, emit.Diagnostics.Where(x => x.Severity == DiagnosticSeverity.Error)));
    }

    [Fact]
    public void Generated_source_reconstructs_a_minimal_executable_module()
    {
        var treeResult = Compiler.CreateTreeFromTorchSharp(
            new CSharpTorchSharpSource("return input;"));
        Assert.True(treeResult.IsSuccess);
        var generated = Compiler.GenerateCSharp(treeResult.Value!);
        Assert.True(generated.IsSuccess);

        var compilation = CSharpCompilation.Create(
            "Onnxify.Generated.CompilerExecutionTest",
            [CSharpSyntaxTree.ParseText(generated.Value!)],
            GetCompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(
            emit.Success,
            string.Join(Environment.NewLine, emit.Diagnostics.Where(x => x.Severity == DiagnosticSeverity.Error)));

        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetType("Onnxify.Generated.GeneratedTorchModule");
        Assert.NotNull(moduleType);
        using var module = (IDisposable)Activator.CreateInstance(moduleType!, "generated")!;
        using var input = global::TorchSharp.torch.tensor(
            new float[] { 1f, 2f },
            new long[] { 2L });
        var output = (global::TorchSharp.torch.Tensor)moduleType!
            .GetMethod("forward")!
            .Invoke(module, [input])!;
        using (output)
        {
            Assert.Equal(input.shape.ToArray(), output.shape.ToArray());
        }
    }

    [Fact]
    public void Helper_methods_are_preserved_as_reusable_blocks()
    {
        var result = Compiler.CreateTreeFromTorchSharp(
            new CSharpTorchSharpSource(
                """
                public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input)
                {
                    return Project(input);
                }

                private global::TorchSharp.torch.Tensor Project(global::TorchSharp.torch.Tensor value)
                {
                    return value;
                }
                """));

        Assert.True(result.IsSuccess, string.Join(Environment.NewLine, result.Diagnostics.Select(x => x.Message)));
        Assert.Contains(result.Value!.Blocks, x => x.Name == "Project");

        var generated = Compiler.GenerateCSharp(result.Value!);

        Assert.True(generated.IsSuccess, string.Join(Environment.NewLine, generated.Diagnostics.Select(x => x.Message)));
        Assert.Contains("private global::TorchSharp.torch.Tensor Project", generated.Value);
        Assert.Contains("return Project(input);", generated.Value);
    }

    [Fact]
    public void Module_calls_are_emitted_through_reusable_blocks()
    {
        var builder = new CompilerComputationTreeBuilder("module-call");
        builder.AddInput(new CompilerValue("input", new CompilerTensorType(CompilerElementType.Float32, null)));
        builder.AddOutput(new CompilerValue("output", new CompilerTensorType(CompilerElementType.Float32, null)));
        builder.AddBlock(
            new CompilerComputationBlock(
                "Project",
                [new CompilerValueReference("value")],
                [new CompilerValueReference("result")],
                new CompilerBlockStatement(
                [
                    new CompilerReturnStatement(new CompilerReferenceExpression("value")),
                ])));
        builder.AddOperation(
            new CompilerModuleCall(
                "project-call",
                "Project",
                [new CompilerValueReference("input")],
                [new CompilerValueReference("output")]));

        var generated = Compiler.GenerateCSharp(builder.Build());

        Assert.True(generated.IsSuccess, string.Join(Environment.NewLine, generated.Diagnostics.Select(x => x.Message)));
        Assert.Contains("var output = Project(input);", generated.Value);
        Assert.Contains("return output;", generated.Value);
    }

    [Fact]
    public void Dynamic_if_is_reported_as_an_analyze_error_with_csharp_span()
    {
        var result = Compiler.CreateTreeFromTorchSharp(
            new CSharpTorchSharpSource(
                """
                public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input)
                {
                    if (input is not null)
                    {
                        return input;
                    }
                    return input;
                }
                """));

        Assert.False(result.IsSuccess);
        var diagnostic = Assert.Single(result.Diagnostics);
        Assert.Equal(CompilerDiagnosticCodes.Unsupported, diagnostic.Code);
        Assert.Equal(CompilerDiagnosticStage.Analyze, diagnostic.Stage);
        Assert.Equal(CompilerDiagnosticSeverity.Error, diagnostic.Severity);
        Assert.Equal(CompilerSourceSpanKind.CSharp, diagnostic.Span!.Kind);
        Assert.NotNull(diagnostic.Context);
    }

    [Fact]
    public void Compiler_descriptor_is_immutable_and_structurally_comparable()
    {
        var descriptor = new CompilerTorchSharpModuleDescriptor(
            typeof(CSharpCompilerTests).Assembly.Location,
            typeof(CSharpCompilerTests).FullName!,
            inputs:
            [
                new CompilerTorchSharpValueDescriptor(
                    "input",
                    new CompilerTensorType(
                        CompilerElementType.Float32,
                        [new CompilerSymbolicDimension("batch")])),
            ],
            outputs:
            [
                new CompilerTorchSharpValueDescriptor(
                    "output",
                    new CompilerTensorType(
                        CompilerElementType.Float32,
                        [new CompilerSymbolicDimension("batch")])),
            ],
            stateMembers:
            [
                new CompilerTorchSharpStateMemberDescriptor(
                    "threshold",
                    CompilerStateMemberKind.Initializer,
                    new CompilerScalarType(CompilerElementType.Float32),
                    new CompilerFloatingPointLiteral(CompilerElementType.Float32, 1.5)),
            ]);

        var equivalent = new CompilerTorchSharpModuleDescriptor(
            typeof(CSharpCompilerTests).Assembly.Location,
            typeof(CSharpCompilerTests).FullName!,
            inputs:
            [
                new CompilerTorchSharpValueDescriptor(
                    "input",
                    new CompilerTensorType(
                        CompilerElementType.Float32,
                        [new CompilerSymbolicDimension("batch")])),
            ],
            outputs:
            [
                new CompilerTorchSharpValueDescriptor(
                    "output",
                    new CompilerTensorType(
                        CompilerElementType.Float32,
                        [new CompilerSymbolicDimension("batch")])),
            ],
            stateMembers:
            [
                new CompilerTorchSharpStateMemberDescriptor(
                    "threshold",
                    CompilerStateMemberKind.Initializer,
                    new CompilerScalarType(CompilerElementType.Float32),
                    new CompilerFloatingPointLiteral(CompilerElementType.Float32, 1.5)),
            ]);

        Assert.Equal(descriptor, equivalent);
        Assert.Equal(descriptor.GetHashCode(), equivalent.GetHashCode());
    }

    [Fact]
    public void Compiled_module_descriptor_can_be_decompiled_without_torchsharp_in_compiler_api()
    {
        var descriptor = new CompilerTorchSharpModuleDescriptor(
            typeof(DescriptorFixture).Assembly.Location,
            typeof(DescriptorFixture).FullName!,
            methodName: nameof(DescriptorFixture.forward),
            methodMetadataToken: typeof(DescriptorFixture).GetMethod(nameof(DescriptorFixture.forward))!.MetadataToken,
            inputs:
            [
                new CompilerTorchSharpValueDescriptor(
                    "input",
                    new CompilerTensorType(CompilerElementType.Float32, null)),
            ],
            outputs:
            [
                new CompilerTorchSharpValueDescriptor(
                    "output",
                    new CompilerTensorType(CompilerElementType.Float32, null)),
            ]);

        var result = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(descriptor));

        Assert.True(result.IsSuccess, string.Join(Environment.NewLine, result.Diagnostics.Select(x => x.Message)));
        Assert.NotNull(result.Value!.SyntaxBody);
        Assert.Contains(result.Value.SyntaxBody!.Statements, x => x is CompilerReturnStatement);
    }

    [Fact]
    public void Torchsharp_adapter_exposes_a_compiler_neutral_descriptor()
    {
        using var module = new AdapterFixture();

        var source = TorchSharpCompilerAdapter.CreateSource(module);

        Assert.Equal(CompilerSourceKind.CSharpTorchSharp, source.Kind);
        Assert.NotNull(source.Module);
        Assert.Equal(typeof(AdapterFixture).Assembly.Location, source.Module!.AssemblyPath);
        Assert.Equal(nameof(AdapterFixture.forward), source.Module.MethodName);

        var result = Compiler.CreateTreeFromTorchSharp(source);

        Assert.True(result.IsSuccess, string.Join(Environment.NewLine, result.Diagnostics.Select(x => x.Message)));
        Assert.Single(result.Value!.Inputs);
    }

    private sealed class DescriptorFixture
    {
        public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input)
        {
            return input;
        }
    }

    private sealed class AdapterFixture
        : global::TorchSharp.torch.nn.Module<
            global::TorchSharp.torch.Tensor,
            global::TorchSharp.torch.Tensor>
    {
        public AdapterFixture()
            : base(nameof(AdapterFixture))
        {
        }

        public override global::TorchSharp.torch.Tensor forward(
            global::TorchSharp.torch.Tensor input)
        {
            return input;
        }
    }

    private static IEnumerable<MetadataReference> GetCompilationReferences()
    {
        var trustedPlatformAssemblies = (string?)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES");
        if (!string.IsNullOrWhiteSpace(trustedPlatformAssemblies))
        {
            foreach (var path in trustedPlatformAssemblies.Split(Path.PathSeparator))
            {
                yield return MetadataReference.CreateFromFile(path);
            }
        }

        yield return MetadataReference.CreateFromFile(typeof(global::TorchSharp.torch.Tensor).Assembly.Location);
    }
}
