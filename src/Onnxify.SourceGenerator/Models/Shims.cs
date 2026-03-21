#pragma warning disable IDE0130 // Namespace does not match folder structure

namespace System.Runtime.CompilerServices
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

#pragma warning restore IDE0130 // Namespace does not match folder structure
