namespace Onnxify.Data;

[AttributeUsage(AttributeTargets.Property | AttributeTargets.Field, AllowMultiple = true)]
public class AcceptTypeAttribute<T> : Attribute
{
    public Type Type => typeof(T);
}
