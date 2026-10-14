using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Onnxify;
using Onnxify.Compiler;

namespace Onnxify.Compiler.Tests;

public sealed class RuntimeControlFlowCompilerTests
{
    [Fact]
    public void RuntimeCSharpIfLowersToOnnxIfWithCapturedValues()
    {
        var tree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(
            """
            public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor condition, global::TorchSharp.torch.Tensor data)
            {
                global::TorchSharp.torch.Tensor output;
                if (condition > 0f)
                {
                    if (condition > 1f)
                    {
                        output = data;
                    }
                    else
                    {
                        output = -data;
                    }
                }
                else
                {
                    output = -data;
                }
                return output;
            }
            """));
        Assert.True(tree.IsSuccess, string.Join(Environment.NewLine, tree.Diagnostics.Select(x => x.Message)));

        var emitted = Compiler.GenerateOnnx(tree.Value!);
        Assert.True(emitted.IsSuccess, string.Join(Environment.NewLine, emitted.Diagnostics.Select(x => x.Message)));
        Assert.Contains(emitted.Value!.Graph.Nodes, node => node.OpType == "If");
        var path = Path.Combine(Path.GetTempPath(), $"oxy042-csharp-if-{Guid.NewGuid():N}.onnx");
        try
        {
            emitted.Value.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            foreach (var (condition, expected) in new[]
            {
                (2f, new[] { 1f, -2f }),
                (0.5f, new[] { -1f, 2f }),
                (0f, new[] { -1f, 2f }),
            })
            {
                using var result = session.Run(
                [
                    NamedOnnxValue.CreateFromTensor("condition", new DenseTensor<float>(new[] { condition }, new int[0])),
                    NamedOnnxValue.CreateFromTensor("data", new DenseTensor<float>(new[] { 1f, -2f }, new[] { 2 })),
                ]);
                Assert.Equal(expected, result.Single().AsTensor<float>().ToArray());
            }
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
        ControlFlowCase[] cases =
        [
            new ControlFlowCase(null, [1f, -2f], [1f, -2f], FloatCondition: 2f),
            new ControlFlowCase(null, [1f, -2f], [-1f, 2f], FloatCondition: 0.5f),
            new ControlFlowCase(null, [1f, -2f], [-1f, 2f], FloatCondition: 0f),
        ];
        AssertRoundTripOutputs(emitted.Value, cases);
        AssertGeneratedTorchSharpMatchesOnnx(emitted.Value, cases);
    }

    [Fact]
    public void RuntimeCSharpForLowersToOnnxLoopWithCarriedValues()
    {
        var tree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(
            """
            public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input)
            {
                var index = 0;
                var result = input;
                for (; index < 3; index++)
                {
                    result = result + input;
                }
                return result;
            }
            """));
        Assert.True(tree.IsSuccess, string.Join(Environment.NewLine, tree.Diagnostics.Select(x => x.Message)));

        var emitted = Compiler.GenerateOnnx(tree.Value!);
        Assert.True(emitted.IsSuccess, string.Join(Environment.NewLine, emitted.Diagnostics.Select(x => x.Message)));
        Assert.Contains(emitted.Value!.Graph.Nodes, node => node.OpType == "Loop");
        AssertGeneratedTorchSharpMatchesOnnx(emitted.Value,
            [new ControlFlowCase(null, [1f, 2f], [4f, 8f])]);
        var path = Path.Combine(Path.GetTempPath(), $"oxy042-csharp-loop-{Guid.NewGuid():N}.onnx");
        try
        {
            emitted.Value.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            using var result = session.Run(
            [NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(new[] { 1f, 2f }, new[] { 2 }))]);
            Assert.Equal(new[] { 4f, 8f }, result.Single().AsTensor<float>().ToArray());
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void RuntimeCSharpWhileLowersToOnnxLoopForZeroAndSeveralIterations()
    {
        var tree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(
            """
            public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input)
            {
                var result = input;
                while (result < 4f)
                {
                    result = result + 1f;
                }
                return result;
            }
            """));
        Assert.True(tree.IsSuccess, string.Join(Environment.NewLine, tree.Diagnostics.Select(x => x.Message)));
        var emitted = Compiler.GenerateOnnx(tree.Value!);
        Assert.True(emitted.IsSuccess, string.Join(Environment.NewLine, emitted.Diagnostics.Select(x => x.Message)));
        AssertGeneratedTorchSharpMatchesOnnx(emitted.Value!,
            [new ControlFlowCase(null, [4f], [4f]), new ControlFlowCase(null, [0f], [4f])]);
        var path = Path.Combine(Path.GetTempPath(), $"oxy042-csharp-while-{Guid.NewGuid():N}.onnx");
        try
        {
            emitted.Value!.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            foreach (var input in new[] { 4f, 0f })
            {
                using var result = session.Run(
                    [NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(new[] { input }, new int[0]))]);
                Assert.Equal(4f, result.Single().AsTensor<float>().Single());
            }
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void RuntimeCSharpDirectBreakAndContinuePreserveLoopSemantics()
    {
        AssertRuntimeLoopOutput(
            """
            public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input)
            {
                var count = 0f;
                while (count < 5f)
                {
                    count++;
                    break;
                }
                return input + count;
            }
            """,
            [3f], [4f]);
        AssertRuntimeLoopOutput(
            """
            public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input)
            {
                var count = 0;
                var output = input;
                for (; count < 2; count++)
                {
                    continue;
                    output = output + 1f;
                }
                return output;
            }
            """,
            [3f], [3f]);
    }

    [Fact]
    public void ConditionalBreakAndContinueAreLoweredAsLoopControlState()
    {
        AssertRuntimeLoopOutput(
            """
            public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input)
            {
                var count = 0;
                var output = input;
                for (; count < 4; count++)
                {
                    if (count == 1) break;
                    output = output + input;
                }
                return output;
            }
            """,
            [3f, 4f], [6f, 8f]);
        AssertRuntimeLoopOutput(
            """
            public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input)
            {
                var count = 0;
                var output = input;
                for (; count < 3; count++)
                {
                    if (count < 2) continue;
                    output = output + input;
                }
                return output;
            }
            """,
            [3f, 4f], [6f, 8f]);
    }

    private static void AssertRuntimeLoopOutput(string source, float[] inputValues, float[] expected)
    {
        var tree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.True(tree.IsSuccess, string.Join(Environment.NewLine, tree.Diagnostics.Select(x => x.Message)));
        var emitted = Compiler.GenerateOnnx(tree.Value!);
        Assert.True(emitted.IsSuccess, string.Join(Environment.NewLine, emitted.Diagnostics.Select(x => x.Message)));
        AssertGeneratedTorchSharpMatchesOnnx(emitted.Value!,
            [new ControlFlowCase(null, inputValues, expected)]);
        var path = Path.Combine(Path.GetTempPath(), $"oxy042-csharp-break-{Guid.NewGuid():N}.onnx");
        try
        {
            emitted.Value!.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            using var result = session.Run(
                [NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(inputValues, new[] { inputValues.Length }))]);
            Assert.Equal(expected, result.Single().AsTensor<float>().ToArray());
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void IfBranchesRoundTripAndExecuteInOnnxRuntime()
    {
        // Runtime semantics: third_party/onnxruntime/onnxruntime/test/providers/cpu/controlflow/if_test.cc (IfOpTester).
        var thenGraph = OnnxModel.Create().Graph;
        thenGraph.Name = "then_branch";
        var thenOutput = thenGraph.AddOutput("then_output", OnnxTensorType.Create<float>([2]));
        var nestedThen = OnnxModel.Create().Graph;
        nestedThen.Name = "nested_then_branch";
        var nestedThenOutput = nestedThen.AddOutput("nested_then_output", OnnxTensorType.Create<float>([2]));
        nestedThen.AddNode("nested_identity", "Identity", string.Empty, string.Empty,
            [new OnnxEdge("data")], [nestedThenOutput], []);
        var nestedElse = OnnxModel.Create().Graph;
        nestedElse.Name = "nested_else_branch";
        var nestedElseOutput = nestedElse.AddOutput("nested_else_output", OnnxTensorType.Create<float>([2]));
        nestedElse.AddNode("nested_neg", "Neg", string.Empty, string.Empty,
            [new OnnxEdge("data")], [nestedElseOutput], []);
        thenGraph.AddNode("nested_choose", "If", string.Empty, string.Empty,
            [new OnnxEdge("condition")], [thenOutput],
            [
                new OnnxAttribute<OnnxGraph>("then_branch", nestedThen),
                new OnnxAttribute<OnnxGraph>("else_branch", nestedElse),
            ]);

        var elseGraph = OnnxModel.Create().Graph;
        elseGraph.Name = "else_branch";
        var elseOutput = elseGraph.AddOutput("else_output", OnnxTensorType.Create<float>([2]));
        elseGraph.AddNode("else_neg", "Neg", string.Empty, string.Empty,
            [new OnnxEdge("data")], [elseOutput], []);

        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 18 });
        var condition = model.Graph.AddInput("condition", OnnxTensorType.Create<bool>([]));
        model.Graph.AddInput("data", OnnxTensorType.Create<float>([2]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([2]));
        model.Graph.AddNode("choose", "If", string.Empty, string.Empty,
            [condition], [output],
            [
                new OnnxAttribute<OnnxGraph>("then_branch", thenGraph),
                new OnnxAttribute<OnnxGraph>("else_branch", elseGraph),
            ]);

        AssertRoundTripOutputs(model,
            [
                new ControlFlowCase(true, [1f, -2f], [1f, -2f]),
                new ControlFlowCase(false, [1f, -2f], [-1f, 2f]),
            ]);
        AssertGeneratedTorchSharpMatchesOnnx(model,
            [
                new ControlFlowCase(true, [1f, -2f], [1f, -2f]),
                new ControlFlowCase(false, [1f, -2f], [-1f, 2f]),
            ]);
    }

    [Fact]
    public void LoopCarriedValuesRoundTripAndExecuteInOnnxRuntime()
    {
        // Runtime semantics: third_party/onnxruntime/onnxruntime/test/providers/cpu/controlflow/loop_test.cc (LoopOpTester).
        var body = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 18 }).Graph;
        body.Name = "loop_body";
        var iteration = body.AddInput("iteration", OnnxTensorType.Create<long>([]));
        var condition = body.AddInput("condition_in", OnnxTensorType.Create<bool>([]));
        var state = body.AddInput("state_in", OnnxTensorType.Create<float>([1]));
        var nextCondition = body.AddOutput("condition_out", OnnxTensorType.Create<bool>([]));
        var nextState = body.AddOutput("state_out", OnnxTensorType.Create<float>([1]));
        body.AddTensor("one", [], [1f]);
        body.AddTensor("stop_after", [], [2L]);
        body.AddNode("continue_while_iteration_is_small", "Less", string.Empty, string.Empty,
            [iteration, new OnnxEdge("stop_after")], [nextCondition], []);
        body.AddNode("increment", "Add", string.Empty, string.Empty,
            [state, new OnnxEdge("one")], [nextState], []);

        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 18 });
        var tripCount = model.Graph.AddInput("trip_count", OnnxTensorType.Create<long>([]));
        var initialCondition = model.Graph.AddTensor("initial_condition", [], [true]);
        var initialState = model.Graph.AddInput("initial_state", OnnxTensorType.Create<float>([1]));
        var output = model.Graph.AddOutput("result", OnnxTensorType.Create<float>([1]));
        model.Graph.AddNode("counted_loop", "Loop", string.Empty, string.Empty,
            [tripCount, initialCondition, initialState], [output],
            [new OnnxAttribute<OnnxGraph>("body", body)]);

        AssertRoundTripOutputs(model,
            [
                new ControlFlowCase(null, [0f], [0f], TripCount: 0L),
                new ControlFlowCase(null, [0f], [3f], TripCount: 5L),
            ]);
        AssertGeneratedTorchSharpMatchesOnnx(model,
            [
                new ControlFlowCase(null, [0f], [0f], TripCount: 0L),
                new ControlFlowCase(null, [0f], [3f], TripCount: 5L),
            ]);
        _ = iteration;
    }

    [Fact]
    public void LoopScanOutputsRoundTripIncludingZeroIterations()
    {
        var body = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 18 }).Graph;
        body.Name = "scan_body";
        _ = body.AddInput("iteration", OnnxTensorType.Create<long>(Array.Empty<OnnxDimension>()));
        var condition = body.AddInput("condition_in", OnnxTensorType.Create<bool>(Array.Empty<OnnxDimension>()));
        var nextCondition = body.AddOutput(condition);
        body.AddTensor("scan_value", [], [2.5f]);
        var scanValue = body.AddOutput("scan_out", OnnxTensorType.Create<float>(Array.Empty<OnnxDimension>()));
        body.AddNode("emit_scan_value", "Identity", string.Empty, string.Empty,
            [new OnnxEdge("scan_value")], [scanValue], []);

        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 18 });
        var tripCount = model.Graph.AddInput("trip_count", OnnxTensorType.Create<long>(Array.Empty<OnnxDimension>()));
        var initialCondition = model.Graph.AddTensor("initial_condition", [], [true]);
        var scanOutput = model.Graph.AddOutput("scan_result", OnnxTensorType.Create<float>());
        model.Graph.AddNode("scan_loop", "Loop", string.Empty, string.Empty,
            [tripCount, initialCondition], [scanOutput],
            [new OnnxAttribute<OnnxGraph>("body", body)]);

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, string.Join(Environment.NewLine, imported.Diagnostics.Select(x => x.Message)));
        var emitted = Compiler.GenerateOnnx(imported.Value!);
        Assert.True(emitted.IsSuccess, string.Join(Environment.NewLine, emitted.Diagnostics.Select(x => x.Message)));
        var sourcePath = Path.Combine(Path.GetTempPath(), $"oxy042-scan-source-{Guid.NewGuid():N}.onnx");
        var emittedPath = Path.Combine(Path.GetTempPath(), $"oxy042-scan-emitted-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(sourcePath, overwrite: true);
            emitted.Value!.Save(emittedPath, overwrite: true);
            using var sourceSession = new InferenceSession(sourcePath);
            using var emittedSession = new InferenceSession(emittedPath);
            foreach (var count in new[] { 0L, 3L })
            {
                var inputs = new[]
                {
                    NamedOnnxValue.CreateFromTensor("trip_count", new DenseTensor<long>(new[] { count }, new int[0])),
                };
                using var sourceResult = sourceSession.Run(inputs);
                using var emittedResult = emittedSession.Run(inputs);
                var expected = Enumerable.Repeat(2.5f, (int)count).ToArray();
                Assert.Equal(expected, sourceResult.Single().AsTensor<float>().ToArray());
                Assert.Equal(expected, emittedResult.Single().AsTensor<float>().ToArray());
            }
        }
        finally
        {
            if (File.Exists(sourcePath)) File.Delete(sourcePath);
            if (File.Exists(emittedPath)) File.Delete(emittedPath);
        }
        _ = nextCondition;
    }

    private static void AssertRoundTripOutputs(OnnxModel source, IReadOnlyList<ControlFlowCase> cases)
    {
        var imported = Compiler.CreateTreeFromOnnx(source);
        Assert.True(imported.IsSuccess, string.Join(Environment.NewLine, imported.Diagnostics.Select(x => x.Message)));
        var emitted = Compiler.GenerateOnnx(imported.Value!);
        Assert.True(emitted.IsSuccess, string.Join(Environment.NewLine, emitted.Diagnostics.Select(x => x.Message)));

        var sourcePath = Path.Combine(Path.GetTempPath(), $"oxy042-source-{Guid.NewGuid():N}.onnx");
        var emittedPath = Path.Combine(Path.GetTempPath(), $"oxy042-emitted-{Guid.NewGuid():N}.onnx");
        try
        {
            source.Save(sourcePath, overwrite: true);
            emitted.Value!.Save(emittedPath, overwrite: true);
            using var sourceSession = new InferenceSession(sourcePath);
            using var emittedSession = new InferenceSession(emittedPath);
            foreach (var testCase in cases)
            {
                var inputValues = new List<NamedOnnxValue>();
                if (testCase.Condition is { } condition)
                {
                    inputValues.Add(NamedOnnxValue.CreateFromTensor("condition", new DenseTensor<bool>(new[] { condition }, new int[0])));
                }
                else if (testCase.FloatCondition is { } floatCondition)
                {
                    inputValues.Add(NamedOnnxValue.CreateFromTensor("condition", new DenseTensor<float>(new[] { floatCondition }, new int[0])));
                }

                if (testCase.TripCount is { } tripCount)
                {
                    inputValues.Add(NamedOnnxValue.CreateFromTensor("trip_count", new DenseTensor<long>(new[] { tripCount }, new int[0])));
                }

                inputValues.Add(NamedOnnxValue.CreateFromTensor(
                    testCase.Condition is null && testCase.FloatCondition is null ? "initial_state" : "data",
                    new DenseTensor<float>(testCase.Input, [testCase.Input.Length])));

                using var sourceResult = sourceSession.Run(inputValues);
                using var emittedResult = emittedSession.Run(inputValues);
                Assert.Equal(testCase.Expected, sourceResult.Single().AsTensor<float>().ToArray());
                Assert.Equal(testCase.Expected, emittedResult.Single().AsTensor<float>().ToArray());
            }
        }
        finally
        {
            if (File.Exists(sourcePath)) File.Delete(sourcePath);
            if (File.Exists(emittedPath)) File.Delete(emittedPath);
        }
    }

    private static void AssertGeneratedTorchSharpMatchesOnnx(
        OnnxModel source,
        IReadOnlyList<ControlFlowCase> cases)
    {
        var imported = Compiler.CreateTreeFromOnnx(source);
        Assert.True(imported.IsSuccess, string.Join(Environment.NewLine, imported.Diagnostics.Select(x => x.Message)));
        var generated = Compiler.GenerateSourceCode(imported.Value!, new CompilerCSharpGenerationOptions
        {
            Namespace = "Onnxify.GeneratedControlFlow",
            ClassName = "GeneratedIf",
            ModuleName = "generated-if",
        });
        Assert.True(generated.IsSuccess, string.Join(Environment.NewLine, generated.Diagnostics.Select(x => x.Message)));

        var references = ((string?)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES"))!
            .Split(Path.PathSeparator)
            .Select(static path => MetadataReference.CreateFromFile(path))
            .Append(MetadataReference.CreateFromFile(typeof(global::TorchSharp.torch.Tensor).Assembly.Location));
        var compilation = CSharpCompilation.Create(
            $"GeneratedIf{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(generated.SourceCode())],
            references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emitted = compilation.Emit(assemblyBytes);
        Assert.True(emitted.Success, string.Join(Environment.NewLine,
            emitted.Diagnostics.Where(x => x.Severity == DiagnosticSeverity.Error))
            + Environment.NewLine + generated.SourceCode());

        assemblyBytes.Position = 0;
        var assembly = System.Runtime.Loader.AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetTypes().Single(type => type.Name == "GeneratedIf");
        using var module = (IDisposable)Activator.CreateInstance(moduleType, "generated-if")!;
        foreach (var testCase in cases)
        {
            var inputTensors = new List<global::TorchSharp.torch.Tensor>();
            foreach (var inputDefinition in source.Graph.Inputs)
            {
                var inputTensor = inputDefinition.Name switch
                {
                    "condition" when testCase.Condition is { } conditionValue =>
                        global::TorchSharp.torch.tensor(new[] { conditionValue }, Array.Empty<long>(), dtype: global::TorchSharp.torch.ScalarType.Bool),
                    "condition" when testCase.FloatCondition is { } floatCondition =>
                        global::TorchSharp.torch.tensor(new[] { floatCondition }, Array.Empty<long>(), dtype: global::TorchSharp.torch.ScalarType.Float32),
                    "trip_count" when testCase.TripCount is { } tripCount =>
                        global::TorchSharp.torch.tensor(new[] { tripCount }, Array.Empty<long>(), dtype: global::TorchSharp.torch.ScalarType.Int64),
                    _ => global::TorchSharp.torch.tensor(testCase.Input, [testCase.Input.Length]),
                };
                inputTensors.Add(inputTensor);
            }
            object[] arguments = inputTensors.Cast<object>().ToArray();
            using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, arguments)!;
            Assert.Equal(testCase.Expected, output.data<float>().ToArray());
            foreach (var inputTensor in inputTensors) inputTensor.Dispose();
        }
    }

    private sealed record ControlFlowCase(bool? Condition, float[] Input, float[] Expected, long? TripCount = null, float? FloatCondition = null);
}
