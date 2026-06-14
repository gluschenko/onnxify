# Onnxify Internal Glossary

Use this glossary for internal project terms, feature code names, and shorthand used in maintenance notes, planning, generated skill content, and architecture discussions.

## Deep Export

`deep export` is the `Onnxify.TorchSharp` feature path that exports ONNX models by recompiling TorchSharp modules.

In this workflow, Onnxify observes or reconstructs the TorchSharp module execution and lowers the discovered Torch operations into ONNX graph structure. Use this term when discussing exporter work that depends on understanding module internals, operator coverage, and TorchSharp-to-ONNX conversion behavior rather than only serializing an already assembled ONNX graph.

Related areas:

- `src/Onnxify.TorchSharp`
- `src/Onnxify.TorchSharp.Observer`
- `references/porting-onnxscript-converters.md`
- `references/finding-torchsharp-porting-candidates.md`

## Deep Import

`deep import` is the complementary `Onnxify.ModelGenerator` feature path that imports an ONNX model as a TorchSharp `TorchModule`.

In this workflow, the generator reads ONNX graph structure and emits TorchSharp module code that reconstructs the model behavior. Use this term when discussing ONNX-to-TorchSharp reconstruction, generated private module fields, inline TorchSharp expressions, and operator support needed for turning ONNX models back into TorchModule implementations.

Related areas:

- `src/Onnxify.ModelGenerator`
- `src/Onnxify.ModelGenerator/Services/TorchModuleInlineOperators`
- `src/Onnxify.ModelGenerator/Services/TorchModuleOperators`
- `references/porting-onnx-to-torchmodule-operators.md`
