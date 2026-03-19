namespace Onnxify;

public class OnnxEdge : IOnnxGraphEdge
{
    public string Name { get; init; }

    public OnnxEdge(string name)
    {
        Name = name;
    }
}
