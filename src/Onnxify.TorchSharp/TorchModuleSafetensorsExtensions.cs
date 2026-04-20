using TorchSharp;

namespace Onnxify.TorchSharp;

public static class TorchModuleSafetensorsExtensions
{
    public static void SaveStateAsSafetensors(
        this TorchModule module,
        string path,
        IReadOnlyDictionary<string, string>? metadata = null,
        bool forceContiguous = true)
    {
        TorchSafetensors.SaveModel(module, path, metadata, forceContiguous);
    }

    public static void LoadStateFromSafetensors(
        this TorchModule module,
        string path,
        bool strict = true,
        global::TorchSharp.torch.Device? device = null)
    {
        _ = TorchSafetensors.LoadModel(module, path, strict, device);
    }
}
