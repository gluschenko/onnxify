namespace Onnxify.Helpers;

internal static class TextHelper
{
    internal static string Indent(this string text, int tabs)
    {
        var indent = new string(' ', tabs * 4);
        return text.Trim().Replace("\n", $"\n{indent}").Trim();
    }
}
