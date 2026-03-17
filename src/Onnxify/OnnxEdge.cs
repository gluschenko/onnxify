namespace Onnxify;

public class OnnxEdge : IOnnxGraphEdge
{
    public string Name { get; init; }

    internal OnnxEdge(string name)
    {
        Name = name;
    }
}
