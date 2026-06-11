# Porting ONNX Operators Into ModelGenerator TorchModules

Use this workflow when you want `Onnxify.ModelGenerator` to reconstruct a TorchSharp `torch.nn.Module` from an ONNX graph through `OnnxifyModelImportType=TorchModule`.

## Goal

Start from an ONNX operator or ONNX graph pattern that appears in real models, understand the equivalent TorchSharp behavior, and add a reverse ONNX-to-TorchSharp conversion that can emit compilable generated C#.

This is the reverse direction of `Onnxify.TorchSharp` exporter work:

- `Onnxify.TorchSharp`: TorchSharp API or module -> ONNX graph
- `Onnxify.ModelGenerator` TorchModule backend: ONNX graph -> generated TorchSharp module

## 1. Pick A Candidate From Evidence

Prefer candidates that unblock real ONNX models or common model families before adding niche operators.

Good signals:

- `Onnxify.ConsoleTest` or an example model fails with `OMG006` for an unsupported ONNX operator.
- `src/Onnxify.TorchSharp.Observer/torchsharp-operator-report.md` shows a missing `Onnxify.ModelGenerator coverage` entry for a common TorchSharp API or module.
- The operator is common in classification, CNN, transformer, quantized, or shape-manipulation graphs.
- The ONNX operator maps cleanly to a TorchSharp tensor method, `torch.nn.functional` call, or a TorchSharp module.

For large batches, do not choose only from intuition. Inspect:

- `src/Onnxify.TorchSharp.Observer/torchsharp-operator-report.md`
- existing generated failures in `src/Onnxify.ConsoleTest`
- real ONNX assets under `src/Onnxify.ConsoleTest/Assets`
- existing TorchSharp exporter behavior in `src/Onnxify.TorchSharp`

## 2. Use ONNXScript As Behavioral Context

The ONNXScript Torch registry lives under:

- `third_party/onnxscript/onnxscript/function_libs/torch_lib/ops`

For ModelGenerator work, read it in reverse.

The Python exporter tells you:

- which ONNX operator or pattern Torch/PyTorch semantics normally lower into
- which attributes, axes, dtype conversions, constants, and broadcasting rules matter
- whether one TorchSharp operation corresponds to one ONNX node or a small ONNX pattern
- which cases are intentionally unsupported or normalized by the exporter

Do not blindly invert code line by line. Use ONNXScript and its tests to understand the semantic contract, then express the reverse conversion idiomatically in C#.

Useful searches:

```powershell
rg -n "aten::where|aten::sum|aten::conv2d" third_party\onnxscript\onnxscript\function_libs\torch_lib\ops
rg -n "Where|ReduceSum|Conv" third_party\onnxscript
```

## 3. Decide Inline Operator Vs Module Operator

ModelGenerator has two TorchModule reconstruction shapes.

Use an inline operator when the generated `forward(...)` can express the operation directly as an expression:

- tensor math: `Add`, `Mul`, `Sin`, `Erf`
- shape and indexing helpers: `Reshape`, `Gather`, `Squeeze`
- functional calls: `AveragePool`, `BatchNormalization`, `Where`
- quantization helper calls: `QuantizeLinear`, `DequantizeLinear`

Inline operators live in:

- `src/Onnxify.ModelGenerator/Services/TorchModuleInlineOperators`

Each inline operator should be represented by a `TorchModuleInlineOperator` subclass with:

- `OnnxOpType`
- `Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values)`

Register it in:

- `TorchModuleInlineOperatorRegistry`

Use a module operator when reconstruction should create private module fields and load weights into `state_dict()`-backed parameters or buffers:

- `Conv`
- `Gemm`/`Linear`
- `BatchNormalization`
- `MaxPool`
- activation modules where a module-backed form is preferred

Module operators live in:

- `src/Onnxify.ModelGenerator/Services/TorchModuleOperators`

Each module operator should be represented by a `TorchModuleOperator` subclass and registered in:

- `TorchModuleOperatorRegistry`

Do not add a separate hard-coded `supportedOps` list for inline behavior. The supported set should be derived from the inline and module registries so support detection and code emission do not drift apart.

## 4. Implement The Conversion In The Owning Abstraction

For inline operators:

1. Add a focused `TorchModuleInlineOperator` subclass.
2. Put operator-specific attribute handling in that class, or in a protected helper on the base class if several operators share it.
3. Register the class in `TorchModuleInlineOperatorRegistry`.
4. Keep `TorchModulePrinter` as a router that asks the registry to emit the expression.

For module operators:

1. Add a focused `TorchModuleOperator` subclass.
2. Detect whether the ONNX node and initializers match the module-backed pattern.
3. Create a `TorchModuleNodeSpecification` with field type, constructor expression, and load statements.
4. Register the class in `TorchModuleOperatorRegistry`.

Prefer small converter classes over a growing switch-case. One ONNX-to-TorchSharp conversion should have one local home for:

- ONNX op key
- pattern constraints
- attribute normalization
- C# expression or module declaration
- generated load statements when weights are involved

## 5. Keep Weight And Initializer Handling General

The TorchModule backend should not special-case a single architecture such as MobileNet or AlexNet.

When initializers are needed:

- support dtype variants only when TorchSharp and `Onnxify.OnnxTensor<T>` can represent them safely
- load data through generated `LoadTensor<T>(...)` or module-specific load statements
- preserve the source ONNX initializer names through generated state names where possible
- avoid folding arbitrary constants into generated code when they should remain stateful buffers or parameters

Important nuance from the current implementation:

- float32 initializers usually become parameters when they are standalone tensor state
- non-float or quantization-related tensors usually become buffers
- QDQ models may require `uint8`, `int8`, `int16`, `int32`, `int64`, `float64`, and `bool` initializer support, not only `float32`

## 6. Add Focused Smoke Tests

Add tests in:

- `src/Onnxify.Tests/OnnxModelGeneratorTests.cs`

Good ModelGenerator TorchModule smoke tests should:

- build a tiny ONNX graph in memory with `ModelProto`
- set `build_metadata.additionalfiles.OnnxifyModelImportType = TorchModule`
- run the source generator through Roslyn
- assert no generator diagnostics with severity `Error`
- assert no updated compilation diagnostics with severity `Error`
- assert representative generated source snippets when useful

Prefer small synthetic graphs over large model fixtures for operator coverage. They are faster, isolate the operator, and catch generated C# signature issues.

Use `src/Onnxify.ConsoleTest` for manual repros against real assets, but do not rely on it as the only validation surface.

## 7. Declare Coverage For Reports

The Observer and generated skill reports discover ModelGenerator coverage through:

- `src/Onnxify.ModelGenerator/TorchSharpOpAttribute.cs`

The current tooling scans `Onnxify.ModelGenerator` for `[TorchSharpOp(...)]` attributes and uses those names for:

- `src/Onnxify.TorchSharp.Observer/torchsharp-operator-report.md`
- `.agents/skills/onnxify/references/operators`

For module-backed converters, decorate the converter class with the matching TorchSharp API/module name:

```csharp
[TorchSharpOp("Conv2d")]
internal sealed class Conv2dTorchModuleOperator : TorchModuleOperator
```

For inline operators, remember that adding a `TorchModuleInlineOperator` subclass can make generation work but may not automatically improve coverage reports unless the operator is also declared with `[TorchSharpOp(...)]` in a place the tooling scans.

Keep the declared name aligned with the TorchSharp API/module spelling used by the observer report, not necessarily the ONNX op name. If one ONNX operator corresponds to several TorchSharp API names, declare every safe mapping deliberately.

## 8. Refresh Generated Artifacts

After adding or changing report-visible ModelGenerator coverage, refresh the observer and skill artifacts:

```powershell
dotnet run --project src\Onnxify.TorchSharp.Observer
dotnet run --project src\Onnxify.AgentSkillGenerator
```

Expected outputs include:

- `src/Onnxify.TorchSharp.Observer/torchsharp-operator-report.md`
- `.agents/skills/onnxify/references/operators/**`
- `.agents/skills/onnxify/references/torchsharp-converters/**`

Do not hand-edit generated skill reference pages when the change should come from `src/Onnxify.AgentSkillGenerator`.

## 9. Validate The Full Path

Before finishing:

- run focused tests, usually `dotnet test src\Onnxify.Tests\Onnxify.Tests.csproj --no-restore --framework net10.0 --filter OnnxModelGeneratorTests`
- build `src\Onnxify.ConsoleTest\Onnxify.ConsoleTest.csproj` when real asset generation might be affected
- inspect `torchsharp-operator-report.md` when coverage reporting was expected to change
- inspect generated skill docs when AgentSkillGenerator was rerun

Warnings such as `OMG004` for external tensor data can be expected for large external-data assets; do not treat them as operator support failures unless the task is specifically about external tensor deployment.

## Heuristics

- Prefer registry-backed converter classes over central switch-cases or hard-coded support lists.
- Keep `TorchModulePrinter` responsible for source layout and routing, not for owning every operator's semantics.
- Keep `OnnxModelGenerator` responsible for graph analysis and diagnostics, not operator-specific code templates.
- Add helpers only when multiple operators share behavior; otherwise keep the logic close to the converter class.
- For shape, axes, and reduction ops, handle both attribute-style older ONNX forms and input-tensor newer forms when practical.
- For pooling and convolution, normalize pads, strides, dilations, groups, and ceil behavior explicitly.
- For comparison and selection ops, verify TorchSharp bool tensor methods through generated-code compilation, not just string assertions.
- For `Cast`, map ONNX `TensorProto.DataType` to TorchSharp `ScalarType` explicitly and fail clearly for unsupported dtypes.
- For quantized models, expect QDQ nodes and integer zero-point tensors.
- For real-model failures, fix the general operator path rather than adding model-specific architecture shortcuts.
