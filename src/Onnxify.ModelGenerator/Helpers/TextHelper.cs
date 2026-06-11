using System.Text;

namespace Onnxify.ModelGenerator.Helpers;

internal static class TextHelper
{
    internal static string Escape(string value)
    {
        var builder = new StringBuilder(value.Length);
        foreach (var ch in value)
        {
            switch (ch)
            {
                case '\\':
                    builder.Append("\\\\");
                    break;
                case '\"':
                    builder.Append("\\\"");
                    break;
                case '\0':
                    builder.Append("\\0");
                    break;
                case '\a':
                    builder.Append("\\a");
                    break;
                case '\b':
                    builder.Append("\\b");
                    break;
                case '\f':
                    builder.Append("\\f");
                    break;
                case '\n':
                    builder.Append("\\n");
                    break;
                case '\r':
                    builder.Append("\\r");
                    break;
                case '\t':
                    builder.Append("\\t");
                    break;
                case '\v':
                    builder.Append("\\v");
                    break;
                default:
                    if (char.IsControl(ch))
                    {
                        builder.Append("\\u");
                        builder.Append(((int)ch).ToString("x4"));
                    }
                    else
                    {
                        builder.Append(ch);
                    }

                    break;
            }
        }

        return builder.ToString();
    }

    internal static string ToVerbatimStringLiteral(string value)
    {
        return "@\"" + value.Replace("\"", "\"\"") + "\"";
    }

    internal static string Indent(string text, int tabs)
    {
        var indent = new string(' ', tabs * 4);
        return text.Trim().Replace("\n", $"\n{indent}").Trim();
    }

    internal static string XmlSummary(string text)
    {
        return $$"""
        /// <summary>
        /// {{EscapeXml(text)}}
        /// </summary>
        """;
    }

    internal static string XmlParam(string name, string text)
    {
        return $"""/// <param name="{name}">{EscapeXml(text)}</param>""";
    }

    internal static string XmlParamXml(string name, string xml)
    {
        return $"""/// <param name="{name}">{xml}</param>""";
    }

    internal static string XmlReturns(string text)
    {
        return $"""/// <returns>{EscapeXml(text)}</returns>""";
    }

    internal static string FormatCode(string text)
    {
        return $"<c>{EscapeXml(text)}</c>";
    }

    internal static string EscapeXml(string text)
    {
        return text
            .Replace("&", "&amp;")
            .Replace("<", "&lt;")
            .Replace(">", "&gt;");
    }
}
