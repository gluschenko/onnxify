# Finding High-Value TorchSharp Porting Candidates

Use this workflow when you want to choose the next `Onnxify.TorchSharp` operator to implement based on evidence from the repo rather than intuition alone.

## Goal

Start from the observer coverage report, confirm what `Onnxify.TorchSharp` already exports, and then rank missing operators by how much real model-export pain they remove.

## 1. Start With The Observer Report

Open:

- `src/Onnxify.TorchSharp.Observer/torchsharp-operator-report.md`

Treat the columns as follows:

- `Found` means the observer discovered a matching TorchSharp surface.
- `Coverage` means `Onnxify.TorchSharp` already declares converter coverage through `[TorchOp(...)]`.

Your main candidate pool is usually:

- `Found = yes`
- `Coverage = no`

That combination means TorchSharp already exposes the operator, but `Onnxify.TorchSharp` does not yet claim export support.

## 2. Separate Module Gaps From Tensor-Op Gaps

Do not stop at the report. Split missing candidates into two buckets by checking the current C# export surface:

- `src/Onnxify.TorchSharp/TorchModuleExtensions.cs`
- `src/Onnxify.TorchSharp/TorchTensorOperatorExtensions.cs`
- `.agents/skills/onnxify/references/torchsharp-converters/index.md`

Use this split deliberately:

- Module-backed gaps are usually operators like `Bilinear`, `EmbeddingBag`, or loss modules.
- Tensor-op gaps are usually shape, indexing, masking, reduction, and math operators like `unsqueeze`, `slice`, `cat`, `sum`, or `where`.

This distinction matters because the highest-value missing work is often in tensor-style operators, even when module coverage already looks decent.

## 3. Check Whether Core `Onnxify` Already Has The ONNX Building Blocks

Before ranking a candidate too highly, verify whether the core graph API already exposes the ONNX nodes you would need.

Search in:

- `src/Onnxify`
- especially `src/Onnxify/OnnxGraph.cs`

Good early candidates usually map onto ONNX operators the repo already knows how to emit, for example:

- `Concat`
- `Split`
- `Gather`
- `Slice`
- `Expand`
- `ReduceSum`
- `ReduceMean`
- `ArgMax`
- `TopK`
- `Where`

If the required ONNX primitive does not exist yet in core `Onnxify`, the task may still be worth doing, but it is no longer a quick export-layer win.

## 4. Look For Manual ONNX Workarounds In Examples

The best signal is not just that an operator is missing, but that the repo is already working around its absence manually.

Search in:

- `src/Onnxify.Examples`
- `src/Onnxify.Tests`

Especially inspect examples that already build custom graph fragments by hand, because they often reveal missing tensor operators that block a more automatic export path.

Example signals:

- a model uses TorchSharp ops like `unsqueeze`, `expand`, `slice`, `triu`, or masking in `forward(...)`
- the exported ONNX path recreates the same behavior manually with low-level `graph.*` calls
- a sample avoids calling `module.Export(...)` for a subgraph that would be natural to export automatically

When you see that pattern, the missing operator is usually a better candidate than a random uncovered row from the report.

## 5. Prefer Operators That Unlock Whole Model Families

Rank candidates by what they unblock, not by how easy they look in isolation.

High-value families tend to be:

- transformer shape and mask ops: `unsqueeze`, `slice`, `expand`, `where`, `masked_fill`, `triu`
- tensor composition ops: `cat`, `split`, `chunk`, `stack`
- common reductions: `sum`, `mean`, `amax`, `argmax`
- indexing ops: `gather`, `select`, `topk`

These operators often unlock many architectures at once:

- decoder attention blocks
- embedding-plus-position pipelines
- sequence aggregation heads
- detection or ranking heads
- classifier post-processing

By contrast, a niche module with no current examples in the repo may have lower priority even if it is easy to implement.

## 6. Use The Existing Examples As A Priority Lens

When the report is long, bias toward operators that help the examples or likely near-term model families already present in the repo.


Practical reading strategy:

- note which TorchSharp ops appear in `forward(...)`
- note which ones already have direct exporters
- note where export code drops down to explicit ONNX graph assembly

If a missing operator would remove custom export code from one of these examples, it is usually a strong candidate.

## 7. Confirm The Candidate Is Truly Uncovered

Before you declare an operator missing, verify all three views agree:

- the observer report still shows `Coverage = no`
- there is no matching `[TorchOp(...)]` in `TorchModuleExtensions.cs`
- there is no matching `[TorchOp(...)]` in `TorchTensorOperatorExtensions.cs`

Use `rg` for the exact Torch op spelling from the report, including overload suffixes when present.

Examples:

- `aten::sum.dim_IntList`
- `aten::slice.Tensor`
- `aten::where.self`

Do not rely on a nearby alias or a related overload unless the exact report spelling is already covered.

## 8. Build A Shortlist Before Implementing

A good shortlist usually includes:

- the exact Torch op name from the report
- whether it is a module gap or tensor-op gap
- the likely ONNX primitive(s) needed
- which example, test, or model family it would unblock
- whether the repo already has the necessary ONNX graph helpers

This is usually enough to choose between "quick win", "high impact", and "needs core work first".

## Heuristics

- Prefer `Found = yes` and `Coverage = no` before inventing brand-new TorchSharp surface area.
- Prefer candidates that remove manual ONNX graph construction already present in the repo.
- Prefer tensor composition, masking, indexing, and reduction ops over obscure math aliases when prioritizing model-export value.
- Prefer operators that unlock transformer, sequence, and detection workflows already visible in `src/Onnxify.Examples`.
- If the ONNX primitive is already available in `src/Onnxify`, the candidate is usually a better near-term port.
- If several report rows are aliases of the same conceptual operation, pick the implementation point that can cover the most exact `[TorchOp(...)]` names safely.
