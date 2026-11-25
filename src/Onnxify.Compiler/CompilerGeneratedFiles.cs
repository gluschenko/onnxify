namespace Onnxify.Compiler;

/// <summary>A file produced by compiler code generation.</summary>
public interface IFile
{
    string Name { get; }
}

/// <summary>A generated UTF-8 source artifact represented as text.</summary>
public sealed class SourceCodeFile : IFile
{
    public SourceCodeFile(string name, string content)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("File name cannot be empty.", nameof(name));
        }

        Name = name;
        Content = content ?? throw new System.ArgumentNullException(nameof(content));
    }

    public string Name { get; }

    public string Content { get; }
}

/// <summary>A generated binary artifact backed by an owned byte array.</summary>
public sealed class BinaryFile : IFile
{
    private readonly byte[] _data;

    public BinaryFile(string name, byte[] data)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("File name cannot be empty.", nameof(name));
        }
        if (data is null)
        {
            throw new System.ArgumentNullException(nameof(data));
        }
        Name = name;
        _data = (byte[])data.Clone();
    }

    public string Name { get; }

    public ReadOnlySpan<byte> Data => _data;
}

/// <summary>Selects where generated modules load their initializer tensors.</summary>
public enum CompilerWeightStorageKind
{
    Onnx = 1,
    Safetensors = 2,
}

/// <summary>Convenience accessors for the source artifact returned by C# generation.</summary>
public static class CompilerGeneratedFileExtensions
{
    public static string SourceCode(this CompilerResult<IFile[]> result)
    {
        if (result is null)
        {
            throw new System.ArgumentNullException(nameof(result));
        }
        var source = result.Value?.OfType<SourceCodeFile>().SingleOrDefault();
        return source?.Content ?? throw new InvalidOperationException("Compiler generation did not return exactly one source file.");
    }
}
