using Onnxify.Compiler;
using Onnxify;

namespace Onnxify.Compiler.Tests;

internal static class GeneratedModuleTestHelpers
{
    public static IDisposable CreateModule(
        Type moduleType,
        CompilerResult<IFile[]> generated,
        string name,
        OnnxModel? weightsModel = null
    )
    {
        var constructor = moduleType.GetConstructors().Single();
        var parameters = constructor.GetParameters();
        if (parameters.Length == 1)
        {
            return (IDisposable)Activator.CreateInstance(moduleType, name)!;
        }

        if (parameters.Length != 2 || parameters[0].ParameterType != typeof(string) || parameters[1].ParameterType != typeof(string))
        {
            throw new InvalidOperationException($"Unexpected generated module constructor: {constructor}.");
        }

        var artifact = generated.Value!.OfType<BinaryFile>().SingleOrDefault();
        if (artifact is null && weightsModel is null)
        {
            throw new InvalidOperationException("Generated module requires external weights; supply the source ONNX model or a generated binary artifact.");
        }

        var path = Path.Combine(Path.GetTempPath(), $"onnxify-test-{Guid.NewGuid():N}-{artifact?.Name ?? "weights.onnx"}");
        try
        {
            if (artifact is not null)
            {
                File.WriteAllBytes(path, artifact.Data.ToArray());
            }
            else
            {
                weightsModel!.Save(path, overwrite: true);
            }
            return (IDisposable)Activator.CreateInstance(moduleType, path, name)!;
        }
        finally
        {
            File.Delete(path);
        }
    }
}
