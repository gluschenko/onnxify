namespace Onnxify.TorchSharp;

[AttributeUsage(AttributeTargets.Method, AllowMultiple = true)]
public class TorchOpAttribute : Attribute
{
    public string Name { get; init; }

    public TorchOpAttribute(string name)
    {
        Name = name;
    }
}
