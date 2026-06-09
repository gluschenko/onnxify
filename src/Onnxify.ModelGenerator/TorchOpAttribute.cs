namespace Onnxify.ModelGenerator;

/// <summary>
/// Associates a TorchSharp export method with the Torch operator name it covers.
/// </summary>
/// <remarks>
/// The attribute is used by repository tooling to report converter coverage and to connect TorchSharp module exports back to ATen operator names. Multiple attributes can be applied when one exporter covers several Torch overload names.
/// </remarks>
[AttributeUsage(AttributeTargets.Method | AttributeTargets.Class, AllowMultiple = true)]
public class TorchOpAttribute : Attribute
{
    /// <summary>
    /// Gets the Torch operator name, for example <c>aten::conv2d</c>.
    /// </summary>
    public string Name { get; init; }

    /// <summary>
    /// Creates a coverage marker for a Torch operator handled by an export method.
    /// </summary>
    /// <param name="name">Torch operator name to associate with the method.</param>
    public TorchOpAttribute(string name)
    {
        Name = name;
    }
}
