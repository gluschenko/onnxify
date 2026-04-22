# Inference From ONNX

## Purpose

Use this workflow when the user gives you a path to an existing `.onnx` model and asks you to build C# inference code for it with `Microsoft.ML.OnnxRuntime`.

The goal is not to guess blindly from the filename. First inspect the model through `Onnxify.CLI`, then shape the inference code around the actual input/output contract and the likely semantics of the graph.

## Default Procedure

When asked to build inference code for an arbitrary ONNX model:

1. Start from the provided `.onnx` file path and confirm which project or code location should own the inference code.
2. Install `Onnxify.CLI` or verify that it is already available.
   Read [cli.md](cli.md) for the tool install and update workflow.
3. Run:

   ```powershell
   onnxify onnx io <path-to-model.onnx>
   ```

4. Inspect the reported model inputs and outputs.
   This tells you the names, tensor types, and shapes that the `OnnxRuntime` code must satisfy.
5. Run:

   ```powershell
   onnxify onnx show <path-to-model.onnx>
   ```

6. Inspect the broader graph structure.
   This is often important because the operator mix, tensor names, and graph layout can reveal whether the model expects images, token IDs, masks, embeddings, audio features, or some other data shape.
7. If the user did not provide real sample data or an existing interface, invent and mock representative input and output examples that match the discovered contract.
   Do not do this if the user explicitly asked to pass through existing data or adapt to an existing API shape.
8. Use `Microsoft.ML.OnnxRuntime` to implement a minimal but correct inference path for the model based on everything learned above.

## Tooling Checklist

Before writing inference code, prefer this exact inspection order:

```powershell
onnxify onnx io <path-to-model.onnx>
onnxify onnx show <path-to-model.onnx>
```

If the CLI is not installed yet, follow the install flow from [cli.md](cli.md), typically:

```powershell
dotnet tool install Onnxify.CLI --global
```

If it is already installed but may be outdated, follow the update flow from [cli.md](cli.md), typically:

```powershell
dotnet tool update Onnxify.CLI --global
```

## What To Look For In `onnx io`

Focus on:

- input names
- output names
- tensor element types
- rank and fixed dimensions
- dynamic dimensions such as batch size or sequence length

These details drive:

- the `DenseTensor<T>` element type
- the input array layout
- the `NamedOnnxValue` names
- any post-processing shape assumptions

## What To Look For In `onnx show`

Use the full graph dump to sanity-check the likely domain of the model.

Examples:

- `Conv`, `Resize`, `MaxPool`, `Transpose`, `Normalize`-like patterns often suggest image input.
- `Gather`, `MatMul`, `LayerNormalization`, `Attention`, `Softmax`, masks, and token-oriented names often suggest NLP or transformer-style input.
- spectral or framing operators can suggest audio features.
- unusual preprocessing nodes may mean the raw user input cannot be fed directly without adaptation.

You do not need a perfect semantic interpretation, but you should avoid writing inference code that contradicts the graph.

## Mocking Inputs And Outputs

When the user only wants a working inference example:

- invent minimal valid sample inputs that match the discovered tensor types and shapes
- keep batch sizes small, usually `1`
- use obvious placeholder values so the sample is easy to replace later
- document which parts are mocked assumptions

When the user already has real inputs or an existing interface:

- do not replace their interface with your own invented DTOs unless needed
- adapt the `OnnxRuntime` call site to the existing names, buffers, and shapes
- preserve any current abstractions if they already map cleanly onto the model contract

## Minimal OnnxRuntime Shape

The baseline implementation usually includes:

- a package reference to `Microsoft.ML.OnnxRuntime`
- session creation with `InferenceSession`
- one or more `DenseTensor<T>` inputs
- `NamedOnnxValue.CreateFromTensor(...)` for each model input
- `session.Run(inputs)` to execute inference
- output extraction by output name and expected tensor type

The exact tensor type and dimensions must come from the inspected model, not from generic examples.

## Expectations For The Produced Code

A good result should:

- use the real input and output names from the model
- use tensor element types consistent with the ONNX contract
- avoid hardcoding wrong shapes when dimensions are dynamic
- include enough mocked data to demonstrate end-to-end execution when real data is not available
- keep preprocessing and postprocessing explicit
- mention any assumptions that came from graph inspection rather than user-provided requirements

## Fallback Guidance

If the model structure is still ambiguous after inspection:

- state the uncertainty clearly
- keep the inference sample minimal and easy to adapt
- prefer a small working skeleton over a large speculative preprocessing pipeline
- annotate the assumptions that should be revisited once real sample inputs are available
