# OXY-1: Modular Deep Export Architecture

## Summary

Refactor the TorchSharp deep export feature from a monolithic decompiled-AST interpreter into a modular pipeline:

1. Read the live TorchSharp module through reflection and collect runtime metadata.
2. Read Onnxify.TorchSharp input/output contract attributes.
3. Decompile the TorchSharp module's `forward` method into C# syntax.
4. Scan supported syntax constructs into a typed intermediate representation.
5. Print the intermediate representation into an `OnnxGraph`.

The goal is to replace the current heuristic-heavy implementation in `src/Onnxify.TorchSharp/TorchModuleExportExtensions.cs` with a structure where each C# syntax construct, TorchSharp operation family, and graph-emission concern is owned by a small, testable component.

This task is architectural and should be implemented incrementally. Public deep export APIs must remain source-compatible unless a later roadmap item explicitly changes them.

The new architecture must also make deep export extensible by library consumers. Users should be able to teach Onnxify how to scan and lower their own module patterns, helper methods, TorchSharp wrappers, runtime metadata conventions, and custom operators without forking `Onnxify.TorchSharp`.

## Current State

Deep export currently lives mostly in `TorchModuleExportExtensions.cs`, which is approximately six thousand lines and combines several responsibilities:

- Public `ExportOnnxModel(...)` overloads and tensor contract extraction.
- ICSharpCode.Decompiler setup and `forward` method discovery.
- Statement walking for declarations, assignments, returns, `if`, `foreach`, `using`, deconstruction, and helper methods.
- Expression evaluation for identifiers, members, invocations, operators, casts, conditionals, indexers, arrays, collection expressions, inline arrays, and constants.
- Reflection over live TorchSharp modules and fields.
- Input/output metadata extraction from `ModuleInputAttribute` and `ModuleOutputAttribute`.
- Runtime shape and dtype discovery from TorchSharp tensors, module fields, module properties, and constructor-populated values.
- Graph-edge value tracking, rank tracking, dtype tracking, shape placeholders, tuple handling, and symbolic values.
- Direct ONNX graph emission for many TorchSharp tensor, functional, module, and helper calls.
- Constant folding, scalar conversion, shape resolution, dtype resolution, and runtime tensor materialization.

This makes local fixes fast at first, but increasingly fragile:

- Adding one syntax construct often requires touching distant helpers.
- Unsupported behavior is hard to diagnose because syntax recognition, value resolution, and graph emission fail in the same layer.
- Operator-specific behavior is mixed with general C# syntax interpretation.
- Tests mostly validate end behavior, but individual lowering decisions are difficult to unit test.
- The implementation cannot easily expose an inspectable representation between decompiled syntax and final ONNX graph.

## Goals

- Introduce a small, explicit intermediate representation for deep export.
- Split syntax recognition into scanner classes with a common abstraction.
- Split ONNX emission into printer classes with a common abstraction.
- Preserve existing deep export behavior while migrating construct by construct.
- Make unsupported syntax failures precise and actionable.
- Make each supported syntax construct independently testable.
- Treat decompiled syntax and reflected runtime metadata as separate, composable inputs to export.
- Use Onnxify.TorchSharp module contract attributes as first-class input/output metadata, not as a special case hidden inside the public facade.
- Resolve runtime-known dimensions and constants from the live module instance when they are not statically visible in decompiled syntax.
- Keep direct module exporters in `TorchModuleExtensions` usable by the new printer layer.
- Keep tensor/operator exporters in `TorchTensorOperatorExtensions` reusable rather than duplicating ONNX node-building logic.
- Provide a path for future support of loops, static branches, nested modules, helper methods, casts, shape expressions, and multi-output values without re-growing one giant dispatcher.
- Provide a stable public extension API for user-defined deep export scanners, printers, module exporters, operation lowerings, and diagnostics.

## Non-Goals

- Do not redesign the public `OnnxModel`, `OnnxGraph`, `OnnxNode`, `OnnxTensor`, or `IOnnxGraphEdge` APIs.
- Do not replace ICSharpCode.Decompiler in this task.
- Do not add runtime tracing as the primary export mechanism.
- Do not remove existing `ExportOnnxModel(...)` overloads.
- Do not change existing ONNX operator semantics unless tests reveal a current bug.
- Do not migrate every operator in a single commit. The implementation should allow gradual migration.
- Do not expose every internal scanner, printer, or IR type publicly by default. Public extension points should be deliberately shaped and versionable.

## Proposed Pipeline

```text
TorchSharp module instance
        |
        v
Runtime metadata collector
        |
        v
Module contract resolver
        |
        v
Forward method decompiler
        |
        v
ICSharpCode.Decompiler C# AST
        |
        v
Deep export scanners
        |
        v
Deep export IR
        |
        v
Deep export printers
        |
        v
OnnxGraph / OnnxModel
```

## Target Project Layout

Create a new internal folder tree under `src/Onnxify.TorchSharp`:

```text
DeepExport/
  Decompilation/
  Diagnostics/
  Ir/
  Metadata/
  Scanning/
    Statements/
    Expressions/
    Torch/
    Modules/
    Helpers/
  Printing/
    Statements/
    Expressions/
    Torch/
    Modules/
    Helpers/
  Runtime/
  Utilities/
```

The exact names may change during implementation, but the separation of responsibilities should remain.

Public extension-facing types should live in stable namespaces such as `Onnxify.TorchSharp.DeepExport` or `Onnxify.TorchSharp.Exporting`. Internal implementation details may remain under `DeepExport/Internal` or use `internal` types.

## Core Abstractions

### DeepExportContext

Owns immutable export inputs and shared services:

- Root TorchSharp module instance.
- Target `OnnxGraph`.
- Declared input and output contracts.
- Resolved module metadata.
- Export options.
- Reflection service.
- Runtime metadata service.
- Module contract service.
- Shape and dtype resolver.
- Diagnostic sink.
- Decompiler service.
- Scanner registry.
- Printer registry.
- User extension registry.

### DeepExportScope

Owns mutable lexical state while scanning or printing one method body:

- Local variables.
- Parameter bindings.
- Known graph values.
- Known constants.
- Known tuple values.
- Known arrays and inline-array builders.
- Known rank and dtype metadata.
- Return value.
- Parent scope reference for helper method calls.

The implementation should distinguish scan-time scope from print-time scope if a single shared type becomes confusing.

Public extensions must not receive unrestricted mutable access to internal scope dictionaries. Expose a narrow context API that allows reading known values, binding explicit outputs, reporting diagnostics, and requesting graph names without depending on private storage details.

### DeepExportModuleMetadata

Represents facts discovered from the live module and its declared contract before scanning begins.

Required metadata:

- Root module type and instance.
- Resolved inputs from explicit `ExportOnnxModel(...)` arguments or `ModuleInputAttribute`.
- Resolved outputs from explicit `ExportOnnxModel(...)` arguments or `ModuleOutputAttribute`.
- Tensor contract dimensions as fixed or symbolic `TensorDimension` values.
- Runtime-known module fields and properties that are safe to read.
- Runtime-known TorchSharp tensor shapes, dtypes, devices, and constant values where needed for graph construction.
- Module constructor arguments that are stored as fields or properties, such as hidden sizes, sequence lengths, vocabulary sizes, channel counts, and boolean configuration flags.
- Child module inventory and stable member names.

The metadata layer must not depend on decompiler syntax. It should be usable by scanners, printers, custom module exporters, and diagnostics.

### DeepExportMetadataProvider

Introduce a provider abstraction for metadata sources.

Suggested shape:

```csharp
public interface IDeepExportMetadataProvider
{
    void Collect(
        DeepExportMetadataContext context,
        object module,
        DeepExportModuleMetadataBuilder metadata
    );
}
```

Built-in providers should include:

- Attribute contract provider for `ModuleInputAttribute` and `ModuleOutputAttribute`.
- Reflection provider for fields and properties.
- Torch tensor provider for shape, dtype, device, and constant tensor values.
- Child module provider for named submodules and module collections.
- Explicit export argument provider for input/output metadata passed directly to `ExportOnnxModel(...)`.

Provider output should be merged deterministically. Explicit export arguments should win over attributes; attributes should win over best-effort reflection where they describe the same input/output contract.

### Shape And Type Resolution

Add a dedicated resolver for ranks, dimensions, and element types.

Responsibilities:

- Convert `ModuleInputAttribute` and `ModuleOutputAttribute` data into `OnnxTensorType`.
- Preserve symbolic dimensions such as `"batch_size"` and `"seq_len"`.
- Resolve fixed dimensions from runtime module fields and properties when syntax references them.
- Resolve dimensions from runtime TorchSharp tensor shapes.
- Resolve dtype references such as `x.dtype`, factory `dtype:` arguments, and module parameter dtypes.
- Track graph-edge rank and dtype metadata produced by printers.
- Report diagnostics when a dimension is ambiguous, unavailable, or conflicts with declared contract metadata.

This resolver should replace scattered helper logic such as ad hoc shape reads, rank dictionaries, dtype dictionaries, and direct conversion helpers.

### Scanner Abstraction

Each scanner handles one syntax pattern and converts it to IR.

Suggested shape:

```csharp
internal abstract class DeepExportScanner<TSyntax>
    where TSyntax : AstNode
{
    public abstract bool CanScan(DeepExportScanContext context, TSyntax syntax);

    public abstract DeepExportIrNode Scan(
        DeepExportScanContext context,
        TSyntax syntax
    );
}
```

Registries should allow ordered matching because some syntax forms are more specific than others. For example, a Torch tensor method call scanner must run before a generic invocation scanner.

Public scanners should implement a stable interface rather than inherit from internal generic base classes if that keeps binary compatibility easier.

### Printer Abstraction

Each printer handles one IR node kind and emits values into an `OnnxGraph`.

Suggested shape:

```csharp
internal abstract class DeepExportPrinter<TNode>
    where TNode : DeepExportIrNode
{
    public abstract bool CanPrint(DeepExportPrintContext context, TNode node);

    public abstract DeepExportValue Print(
        DeepExportPrintContext context,
        TNode node
    );
}
```

Printers should be deterministic, avoid reflection unless the IR explicitly requires it, and return a typed `DeepExportValue`.

Public printers should be able to emit graph fragments through a safe `DeepExportPrintContext` facade that exposes the target `OnnxGraph`, naming services, value conversion helpers, and diagnostics.

### DeepExportValue

Replace the current object-shaped `ExportValue` with a discriminated value model.

Required value categories:

- `GraphEdgeValue`: wraps `IOnnxGraphEdge`.
- `TensorInitializerValue`: wraps `OnnxTensor` where useful.
- `ScalarValue`: stores numeric, bool, string, dtype, or enum constants.
- `ArrayValue`: stores ordered `DeepExportValue` items.
- `TupleValue`: stores ordered and optionally named values.
- `ModuleValue`: wraps a live TorchSharp module or helper object.
- `MethodValue`: identifies a local helper method.
- `ShapeDimensionValue`: represents `x.shape[i]` before its consumer decides static vs dynamic lowering.
- `SymbolicTensorMemberValue`: represents `x.dtype`, `x.device`, and similar metadata.
- `NullValue`.

Avoid raw `object` where possible. If an escape hatch is needed, isolate it in a named `RuntimeObjectValue`.

Public extension code should only depend on public value categories. Internal-only values may exist, but they must not be required to implement common custom exporter scenarios.

## Public Extensibility API

Deep export must support user-defined extensions as a first-class design constraint.

### Configuration Surface

Add an options object that can be passed through the existing public export entry points without breaking current overloads.

Suggested shape:

```csharp
public sealed class TorchDeepExportOptions
{
    public DeepExportExtensionCollection Extensions { get; }

    public DeepExportDiagnosticLevel DiagnosticLevel { get; set; }

    public bool IncludeDebugIr { get; set; }

    public bool EnableRuntimeMetadataReflection { get; set; }
}
```

Existing `ExportOnnxModel(...)` overloads can either accept an optional `TorchDeepExportOptions` parameter in new overloads or allow `OnnxModelCreationOptions` to carry extension settings if that is judged cleaner. The public API should avoid forcing users to call internal registries directly.

### Extension Registration

Users should be able to register extensions fluently:

```csharp
var exportOptions = new TorchDeepExportOptions()
    .AddScanner(new MyShapeHelperScanner())
    .AddPrinter(new MyShapeHelperPrinter())
    .AddModuleExporter<MyCustomBlock>(new MyCustomBlockExporter())
    .AddTorchOperation("my_ops::swish", new SwishLowering());
```

Required registration capabilities:

- Register a syntax scanner.
- Register an IR printer.
- Register a module exporter for a TorchSharp module type.
- Register a method or helper lowering by reflection signature.
- Register a Torch operation lowering by symbolic name.
- Register a graph post-processor.
- Register a metadata provider.
- Register a shape or dtype resolver extension.
- Register a diagnostic listener.

### Extension Ordering

The extension collection must define deterministic ordering:

- Built-in exact-match scanners/printers should normally run before generic built-ins.
- User extensions must be able to run before built-ins when intentionally overriding behavior.
- User extensions must be able to run after built-ins when only filling gaps.
- Ambiguous matches should produce a clear diagnostic unless priority explicitly resolves them.

Suggested model:

```csharp
public enum DeepExportExtensionPriority
{
    BeforeBuiltIns,
    Normal,
    AfterBuiltIns,
}
```

### Custom Scanner API

A public scanner should be able to recognize a syntax node and return a public or extension IR node.

Suggested shape:

```csharp
public interface IDeepExportScanner
{
    bool CanScan(DeepExportScanContext context, DeepExportSyntax syntax);

    DeepExportNode Scan(
        DeepExportScanContext context,
        DeepExportSyntax syntax
    );
}
```

`DeepExportSyntax` may wrap ICSharpCode.Decompiler syntax instead of exposing raw AST types directly. If raw ICSharpCode types are exposed, the package versioning impact must be documented.

### Custom Printer API

A public printer should lower public or extension IR nodes into graph values.

Suggested shape:

```csharp
public interface IDeepExportPrinter
{
    bool CanPrint(DeepExportPrintContext context, DeepExportNode node);

    DeepExportValue Print(
        DeepExportPrintContext context,
        DeepExportNode node
    );
}
```

The print context should expose:

- `OnnxGraph Graph`.
- `string NextName(string prefix)`.
- Input, output, rank, dtype, and shape metadata readers.
- Helpers for scalar, tensor, array, tuple, and graph-edge conversion.
- Diagnostics.
- Access to registered built-in lowerings for delegation.

### Custom Module Exporters

Users should be able to export custom TorchSharp modules without relying on decompiled syntax.

Suggested shape:

```csharp
public interface IDeepExportModuleExporter
{
    bool CanExport(DeepExportModuleExportContext context, object module);

    DeepExportValue Export(
        DeepExportModuleExportContext context,
        object module,
        IReadOnlyList<DeepExportValue> inputs
    );
}
```

Generic convenience wrappers should be provided:

```csharp
public abstract class DeepExportModuleExporter<TModule> : IDeepExportModuleExporter
{
    public abstract DeepExportValue Export(
        DeepExportModuleExportContext context,
        TModule module,
        IReadOnlyList<DeepExportValue> inputs
    );
}
```

This is important for user-defined blocks where source decompilation is brittle or where the desired ONNX graph is known directly.

### Custom Operation Lowerings

Users should be able to map method calls or semantic operation names to ONNX fragments.

Required scenarios:

- Lower a static helper method such as `MyOps.Swish(x)`.
- Lower an instance method wrapper such as `x.MyNormalize(epsilon)`.
- Lower a user module's `forward`.
- Lower a generated or source-generator-created helper.
- Delegate part of the lowering to built-in tensor/operator helpers.

### Custom Metadata Providers

Users should be able to teach deep export how to discover runtime facts from their own module conventions.

Required scenarios:

- A module stores `HiddenSize`, `MaxSequenceLength`, or `VocabularySize` as public properties.
- A module stores shape information in private readonly fields populated by constructor arguments.
- A module wraps TorchSharp tensors in a custom parameter container.
- A module has non-standard input/output annotations that should map to Onnxify tensor contracts.
- A module has shape aliases where several symbolic dimensions refer to the same runtime value.

Metadata providers should run before scanning and should be available to scanners, printers, and custom module exporters through a read-only metadata view.

Suggested shape:

```csharp
public interface IDeepExportMetadataProvider
{
    void Collect(
        DeepExportMetadataContext context,
        object module,
        DeepExportModuleMetadataBuilder metadata
    );
}
```

### Attribute-Based Registration

Consider optional attributes for common extension cases:

```csharp
[DeepExportModuleExporter(typeof(MyCustomBlockExporter))]
public sealed class MyCustomBlock : torch.nn.Module<torch.Tensor, torch.Tensor>
{
}

[DeepExportMethodExporter(typeof(SwishExporter))]
public static torch.Tensor Swish(torch.Tensor input)
{
    return input * torch.sigmoid(input);
}

[DeepExportMetadataProvider(typeof(MyModelMetadataProvider))]
public sealed class MyModel : torch.nn.Module<torch.Tensor, torch.Tensor>
{
}
```

Attribute registration should be optional. Explicit options-based registration must remain available for users who cannot annotate third-party types.

### Diagnostics For Extensions

Diagnostics must distinguish built-in failures from extension failures:

- Extension type name.
- Extension priority.
- Syntax or IR node being handled.
- Whether the extension declined, failed, or produced an invalid value.
- Inner exception details when an extension throws.

The public API should allow strict and permissive modes:

- Strict mode: extension exceptions fail export.
- Permissive mode: extension exceptions become diagnostics and dispatch continues when safe.

### Versioning Rules

Public extensibility APIs must be designed as stable contracts:

- Prefer interfaces and small immutable DTOs over exposing internal mutable classes.
- Avoid leaking ICSharpCode.Decompiler types unless the dependency is accepted as part of the public compatibility surface.
- Avoid making runtime metadata APIs depend on decompiler syntax.
- Avoid requiring inheritance from internal classes.
- Document which extension points are stable and which are experimental.
- Include tests that compile a small external-style extension against the public API.

### Security And Safety

Extensions execute user code during export. The API should make this explicit:

- Do not sandbox extension execution.
- Do not catch and suppress extension exceptions by default in strict mode.
- Avoid invoking arbitrary reflection from extension code unless the user explicitly does so.
- Keep extension diagnostics clear enough to debug bad lowerings.

## Intermediate Representation

The IR should describe supported program semantics rather than raw syntax text.

### Base Node Requirements

Every IR node should carry:

- Source syntax text or source location for diagnostics.
- Stable node kind.
- Child nodes.
- Optional static result type or expected value category.

### Statement Nodes

Initial statement node types:

- `BlockNode`
- `VariableDeclarationNode`
- `AssignmentNode`
- `ReturnNode`
- `ExpressionStatementNode`
- `IfNode`
- `ForeachNode`
- `UsingScopeNode`
- `DeconstructionNode`
- `IgnoredValidationCallNode`
- `UnsupportedStatementNode`

### Expression Nodes

Initial expression node types:

- `ParameterReferenceNode`
- `LocalReferenceNode`
- `MemberReferenceNode`
- `StaticMemberReferenceNode`
- `InvocationNode`
- `TorchInvocationNode`
- `TensorMethodInvocationNode`
- `ModuleForwardInvocationNode`
- `LocalHelperInvocationNode`
- `GeneratedHelperInvocationNode`
- `BinaryExpressionNode`
- `UnaryExpressionNode`
- `ConditionalExpressionNode`
- `CastExpressionNode`
- `IndexerExpressionNode`
- `ArrayExpressionNode`
- `InlineArrayExpressionNode`
- `LiteralNode`
- `TupleExpressionNode`
- `NullNode`

### Torch Operation Nodes

Torch operation nodes should be more semantic than generic invocations when the operation is known:

- `TorchMatMulNode`
- `TorchConcatNode`
- `TorchWhereNode`
- `TorchSoftmaxNode`
- `TorchReluNode`
- `TorchGeluNode`
- `TorchConv2dNode`
- `TorchPadNode`
- `TorchBatchNormNode`
- `TorchAdaptiveAvgPool2dNode`
- `TorchArangeNode`
- `TorchFactoryNode`
- `TorchRandomFactoryNode`
- `TorchTensorConstructorNode`
- `TensorReshapeNode`
- `TensorFlattenNode`
- `TensorPermuteNode`
- `TensorTransposeNode`
- `TensorUnsqueezeNode`
- `TensorSliceNode`
- `TensorGatherNode`
- `TensorSumNode`
- `TensorClampNode`
- `TensorToTypeNode`

The first implementation may start with a smaller subset and keep a fallback to the current monolith.

## Scanner Responsibilities

Scanners must only parse syntax and produce IR. They should not emit ONNX nodes.

Required scanner families:

- Statement scanners: variable declarations, assignments, returns, using scopes, foreach loops over statically known module collections, static `if` branches, deconstruction, validation guards.
- Expression scanners: literals, identifiers, member references, invocations, binary/unary operators, casts, conditionals, indexers, arrays, collection expressions, inline-array compiler artifacts.
- Torch scanners: static `torch.*`, static-imported Torch calls, `torch.nn.functional.*`, tensor instance methods.
- Module scanners: known module `forward` calls, recursive user module calls, sequential module traversal.
- Helper scanners: local helper methods and generated helper methods from `Onnxify.ModelGenerator`.

Scanners may consult reflection and scope to classify syntax, but they should store the classification in IR rather than performing final graph emission.

## Printer Responsibilities

Printers consume IR and produce `DeepExportValue`.

Required printer families:

- Statement printers: execute data-flow semantics over `DeepExportScope`.
- Expression printers: resolve references, constants, tuples, arrays, shape placeholders, and scalar folding.
- Torch printers: emit ONNX graph fragments using existing `OnnxGraph`, `TorchModuleExtensions`, and `TorchTensorOperatorExtensions` helpers.
- Module printers: call existing concrete module exporters when available, or recursively print a child module's scanned IR.
- Helper printers: inline local helpers and generated graph helpers through explicit IR.

Printers must not inspect raw ICSharpCode syntax except for diagnostics already stored on IR nodes.

## Registry Design

Introduce registries for scanners and printers:

- `DeepExportScannerRegistry`
- `DeepExportPrinterRegistry`

Requirements:

- Ordered dispatch from most-specific to least-specific.
- Clear error messages when no scanner or printer matches.
- Unit tests proving dispatch order for ambiguous invocation syntax.
- No central thousand-line `switch` as a replacement for the current monolith.
- Public extension registration with deterministic priority and duplicate-resolution diagnostics.

## Diagnostics

Add structured diagnostics for unsupported syntax:

- Include the syntax snippet.
- Include the method/module being exported.
- Include whether the failure occurred during scanning or printing.
- Include the scanner/printer that rejected the node when applicable.
- Prefer `NotSupportedException` or `InvalidOperationException` at the public API boundary, but build messages from structured diagnostics.

Examples:

- Unsupported dynamic `foreach` source.
- Unsupported non-validation void helper.
- Unsupported tensor dtype.
- Unsupported runtime-only branch condition.
- Unsupported array expression shape.
- Unsupported module type with no concrete exporter and no scannable `forward`.
- Missing `ModuleInputAttribute` or `ModuleOutputAttribute` when attribute-based export is requested.
- Conflicting input/output contracts between explicit arguments and attributes.
- Runtime dimension referenced by syntax but unavailable through metadata providers.
- Runtime tensor shape or dtype needed for export but unavailable or ambiguous.
- User extension failed while scanning or printing a custom node.
- Multiple extensions matched the same syntax or IR node without an explicit priority winner.

## Migration Strategy

### Phase 1: Extract Infrastructure

- Move decompilation helpers into `DeepExport/Decompilation`.
- Move module contract extraction into `DeepExport/Metadata`.
- Introduce `DeepExportContext`, `DeepExportScope`, `DeepExportValue`, `DeepExportModuleMetadata`, diagnostics, and registries.
- Introduce runtime metadata providers for attributes, reflection, Torch tensors, child modules, and explicit export arguments.
- Introduce the public `TorchDeepExportOptions` and extension collection shape, even if only a small subset is wired initially.
- Keep `TorchModuleExportExtensions` as the public facade.
- Add tests for value categories, contract extraction, runtime metadata extraction, and diagnostic formatting.

### Phase 2: Build IR and Scanner Skeleton

- Add base IR nodes.
- Add statement and expression scanner base classes.
- Scan a minimal forward body containing parameters, local declarations, tensor method calls, and return statements.
- Add tests that scan existing tiny test modules and snapshot or assert the IR shape.

### Phase 3: Build Printer Skeleton

- Add printer base classes and registry.
- Print the minimal IR subset into the same ONNX graph shape as the current exporter.
- Validate with existing simple deep export tests.
- Add a minimal public extension test that registers a custom printer or module exporter.

### Phase 4: Migrate Syntax Constructs

Move support in focused batches:

- Variable declarations, assignments, returns.
- Binary and unary tensor/scalar operations.
- Member references and reflection.
- Runtime metadata-backed field/property references.
- Array and collection expressions.
- Inline-array compiler artifacts.
- Shape and dtype metadata reads.
- Static conditionals and validation guards.
- Using scopes.
- Static foreach over module collections.
- Deconstruction and tuple outputs.
- Local helper method inlining.
- Generated helper methods.

Each batch must include direct scanner/printer tests plus at least one existing behavior test.

### Phase 5: Migrate Torch Operation Families

Move operation families behind semantic IR nodes:

- Matrix multiplication: `matmul`, `mm`, `bmm`, tensor instance variants.
- Activations: ReLU, GELU, sigmoid, exp, log softmax.
- Shape ops: reshape/view, flatten, permute, transpose, unsqueeze, slice, gather.
- Reductions and comparisons.
- Factories: `arange`, `tensor`, `zeros`, `ones`, `full`, `empty`, like-factories.
- Random factories.
- Concatenation and where.
- Functional ops: conv2d, pad, batch norm, adaptive avg pool, interpolate.
- Module `forward` calls and recursive user module export.

### Phase 6: Remove Monolithic Fallback

After parity coverage is high enough:

- Delete migrated private helpers from `TorchModuleExportExtensions`.
- Keep only public overloads and orchestration.
- Ensure no new feature work is added to the old path.
- Update internal skill docs if the contributor workflow changes.

### Phase 7: Harden Public Extensibility

- Document the public extension API in the package README.
- Add examples for custom module exporters and custom helper-method lowerings.
- Add binary/source compatibility tests for an external-style extension project.
- Decide which extension APIs are stable and which remain experimental.
- Add release notes for the new extension surface before shipping.

## Testing Requirements

### Unit Tests

Add focused tests for:

- Scanner dispatch order.
- Printer dispatch order.
- IR shape for representative syntax constructs.
- `DeepExportValue` category conversions.
- Diagnostics for unsupported syntax.
- Diagnostics for failing or ambiguous user extensions.
- Metadata resolution from `ModuleInputAttribute` and `ModuleOutputAttribute`.
- Runtime field/property shape resolution for constructor-provided module sizes.
- Runtime TorchSharp tensor shape and dtype resolution.
- Conflict diagnostics for explicit contracts vs attribute contracts.
- Scope behavior for locals, nested helper calls, tuple deconstruction, and shadowing.
- Public extension registration and priority behavior.

### Existing Behavior Tests

Keep the existing deep export tests passing:

- `TorchModuleDeepExportTests`
- `TorchModuleDeepExportSmokeTests`
- Relevant parity tests in `DeepImportExportParityTests` when touched.

### Graph Validation

For each migrated operator family:

- Assert important emitted node types and attributes.
- Create an ONNX Runtime `InferenceSession` via `OnnxRuntimeCompatibilityAssert` where possible.
- Compare TorchSharp eager output with ONNX Runtime output for executable deterministic cases.

### Regression Fixtures

Add compact fixture modules for:

- Nested helper methods.
- Tuple-returning helpers.
- Static branch selection.
- Static foreach over modules.
- Inline array and collection expression shapes.
- Runtime dtype factory calls.
- Generated helper calls.
- User-defined custom module exporter.
- User-defined custom helper-method lowering.
- Attribute-declared input/output contracts.
- Runtime constructor argument dimensions reflected through fields or properties.
- User-defined metadata provider.

## Acceptance Criteria

OXY-1 is complete when:

- `TorchModuleExportExtensions` no longer owns scanning, semantic interpretation, and graph emission in one file.
- The public `ExportOnnxModel(...)` APIs still work for all currently covered tests.
- Attribute-based `ExportOnnxModel(options)` still reads `ModuleInputAttribute` and `ModuleOutputAttribute` through the new metadata layer.
- Runtime-known dimensions from module fields, properties, child modules, and TorchSharp tensors can be resolved without embedding that logic in syntax scanners.
- There is an explicit IR layer between decompiled syntax and `OnnxGraph`.
- At least the syntax constructs currently covered by `TorchModuleDeepExportTests` are represented through scanners and printers.
- Adding a new supported syntax construct requires adding or modifying a scanner/printer pair, not editing a central monolithic interpreter.
- Unsupported syntax failures identify the failing source snippet and pipeline stage.
- Existing deep export smoke tests pass.
- New scanner/printer unit tests cover the core dispatch and IR behavior.
- A user can register at least one custom module exporter and one custom method or operation lowering without modifying `Onnxify.TorchSharp` source.
- A user can register a custom metadata provider that contributes shape or contract information to export.
- Public extension failures produce diagnostics that identify the extension and the syntax or IR node involved.

## Implementation Notes

- Prefer internal types unless a public extension point is intentionally introduced; for extension points, prefer small stable public interfaces over exposing implementation classes.
- Keep naming aligned with existing repository style.
- Keep vertical argument formatting consistent with the repo.
- Do not hand-edit generated skill references as part of this refactor unless `TorchOpAttribute` coverage changes.
- Use existing graph helper methods in `TorchTensorOperatorExtensions` before adding new ONNX-emission code.
- Preserve deterministic graph naming where tests rely on names.
- When graph names change unavoidably, update tests to assert semantic structure rather than incidental names.
- Keep contract metadata resolution centralized. Do not reintroduce direct scattered reads of `ModuleInputAttribute`, `ModuleOutputAttribute`, or runtime shape fields in scanners/printers.
- If new public extension APIs are introduced, update package README and release notes before shipping.

## Risks

- The decompiler output is not a stable source language. Scanners must support compiler/decompiler artifacts such as inline arrays and generated helper syntax, while metadata that can be learned from reflection should not depend on that syntax.
- Splitting the pipeline can accidentally change eager-vs-ONNX parity. Keep runtime parity tests close to each migrated operator family.
- Too much generic IR can become another monolith. Prefer semantic nodes for Torch operations once recognized.
- Too much operator-specific IR can duplicate `TorchTensorOperatorExtensions`. Printers should delegate graph fragments to existing helpers.
- Reflection over live modules must remain controlled and testable because decompiled member names are mapped back to runtime fields and properties.
- Runtime metadata reflection can observe private implementation details. Keep reads deterministic, avoid mutating module state, and document what members are considered safe to inspect.
- Contract metadata conflicts can silently produce wrong ONNX signatures if not diagnosed. Explicit conflict checks are required.
- Public extension APIs can become hard to change after release. Keep the first stable surface small and mark broader hooks experimental if needed.
- Exposing raw decompiler AST types may tie Onnxify's public compatibility to ICSharpCode.Decompiler. Prefer wrappers unless direct access is clearly worth the cost.

## Open Questions

- Should the IR be printable/debuggable as JSON or markdown for troubleshooting failed exports?
- Should scanner/printer registries be hard-coded internal lists, attribute-discovered, or generated?
- Should direct module exporters in `TorchModuleExtensions` eventually be wrapped as printers, or remain as graph-emission helpers called by module printers?
- Should ONNX graph optimization or canonicalization happen after printing, or remain outside deep export?
- Should unsupported dynamic control flow produce a future `ControlFlowNode` IR, or continue to fail until ONNX control-flow export is explicitly designed?
- Should public extensions see raw ICSharpCode.Decompiler AST nodes or a stable Onnxify-owned syntax wrapper?
- Should the public extension API be marked experimental for one minor release before being treated as stable?
- Should extension registration support dependency injection, or is explicit options-based registration enough for the first version?
- Should symbolic dimensions from `TensorDimension` support alias constraints or runtime bindings beyond fixed/string values?
- Should runtime metadata reflection include private fields by default, or require opt-in through attributes/options?
- Should attribute-based input/output contracts be extensible with custom user attributes, or should that be handled only through metadata providers?
