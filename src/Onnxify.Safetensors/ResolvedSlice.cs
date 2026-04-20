namespace Onnxify.Safetensors;

internal readonly record struct ResolvedSlice(ulong Start, ulong Stop, bool IsSelect);
