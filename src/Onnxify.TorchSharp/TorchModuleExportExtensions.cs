using ICSharpCode.Decompiler;
using ICSharpCode.Decompiler.CSharp;
using ICSharpCode.Decompiler.Metadata;

namespace Onnxify.TorchSharp;

public static class TorchModuleExportExtensions
{
    public static OnnxModel Export(
        this TorchModule module,
        OnnxTensorType input,
        OnnxTensorType output,
        OnnxModelCreationOptions options
    )
    {
        var onnxModel = OnnxModel.Create(options);

        var assemblyPath = module.GetType().Assembly.Location;

        var decompiler = new CSharpDecompiler(
            fileName: assemblyPath,
            settings: new DecompilerSettings
            {
                ThrowOnAssemblyResolveErrors = false,
            }
        );

        var method = module.GetType().GetMethod("forward")!;
        var metadataToken = MetadataTokenHelpers.EntityHandleOrNil(method.MetadataToken);

        var syntaxTree = decompiler.Decompile(metadataToken);


        return onnxModel;
    }
}
