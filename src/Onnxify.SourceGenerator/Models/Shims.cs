namespace Onnxify.SourceGenerator.Models
{
    [AttributeUsage(AttributeTargets.All, AllowMultiple = false)]
    internal sealed class RequiredMemberAttribute : Attribute
    {
        public RequiredMemberAttribute() { }
    }

    [AttributeUsage(AttributeTargets.All, AllowMultiple = true)]
    internal sealed class CompilerFeatureRequiredAttribute : Attribute
    {
        public CompilerFeatureRequiredAttribute(string featureName) { }
        public bool IsOptional { get; set; }
    }
}
