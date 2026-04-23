namespace Onnxify;

/// <summary>
/// Represents a named graph wire that has no value-info metadata or initializer payload.
/// </summary>
/// <remarks>
/// Use this for intermediate node connections when shape/type metadata is unavailable or unnecessary. If you later need ONNX value-info metadata, add an <see cref="OnnxValue"/> with the same wire name instead of relying on a loose edge.
/// </remarks>
public class OnnxEdge : IOnnxGraphEdge
{
    /// <summary>
    /// Gets the exact wire name written into node input and output lists.
    /// </summary>
    public string Name { get; init; }

    /// <summary>
    /// Creates a loose graph edge.
    /// </summary>
    /// <param name="name">Wire name to use in node connections.</param>
    public OnnxEdge(string name)
    {
        Name = name;
    }

    /// <summary>
    /// Returns the wire name for compact graph diagnostics.
    /// </summary>
    /// <returns>The edge name.</returns>
    public override string ToString()
    {
        return Name;
    }
}
