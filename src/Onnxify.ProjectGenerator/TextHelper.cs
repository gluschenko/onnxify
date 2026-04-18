using System.Text;

namespace Onnxify.ProjectGenerator;

internal static class TextHelper
{
    private static readonly HashSet<string> _keywords =
    [
        "class",
        "namespace",
        "public",
        "private",
        "protected",
        "internal",
        "static",
        "void",
        "string",
        "int",
        "long",
        "float",
        "double",
        "bool",
        "object",
        "return",
        "new",
        "base",
        "this",
        "params",
        "out",
        "ref",
        "in",
        "var",
    ];

    public static string SanitizeIdentifier(this string value, string fallback)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return fallback;
        }

        var builder = new StringBuilder(value.Length + 1);

        if (!IsIdentifierStart(value[0]))
        {
            builder.Append('_');
        }

        foreach (var ch in value)
        {
            builder.Append(IsIdentifierPart(ch) ? ch : '_');
        }

        var result = builder.ToString();
        return _keywords.Contains(result) ? "@" + result : result;
    }

    public static string SanitizeLocalIdentifier(this string value, string fallback)
    {
        var result = SanitizeIdentifier(value, fallback);
        var offset = result.StartsWith('@') ? 1 : 0;
        var chars = result.ToCharArray();
        var loweredFirstWord = false;

        for (var i = offset; i < chars.Length; i++)
        {
            if (chars[i] == '_')
            {
                continue;
            }

            if (!loweredFirstWord)
            {
                if (char.IsLetter(chars[i]))
                {
                    chars[i] = char.ToLowerInvariant(chars[i]);
                    loweredFirstWord = true;
                }

                continue;
            }

            if (char.IsDigit(chars[i]))
            {
                continue;
            }

            if (char.IsUpper(chars[i]))
            {
                if (i + 1 < chars.Length && char.IsLower(chars[i + 1]))
                {
                    break;
                }

                chars[i] = char.ToLowerInvariant(chars[i]);
                continue;
            }

            break;
        }

        return new string(chars);
    }

    public static string SanitizeFileName(this string value)
    {
        var invalidChars = Path.GetInvalidFileNameChars();
        var chars = value.Select(ch => invalidChars.Contains(ch) ? '_' : ch).ToArray();
        var result = new string(chars).Trim();
        return string.IsNullOrWhiteSpace(result) ? "tensor" : result;
    }

    public static string Indent(this string text, int tabs)
    {
        var indent = new string(' ', tabs * 4);
        var normalized = text.Trim().Replace("\r\n", "\n", StringComparison.Ordinal).TrimEnd();
        var lines = normalized.Split('\n');
        var result = string.Join(Environment.NewLine, lines.Select(line => string.IsNullOrWhiteSpace(line) ? string.Empty : indent + line));
        return result.Trim();
    }

    public static bool IsIdentifierStart(this char ch) => ch == '_' || char.IsLetter(ch);
    public static bool IsIdentifierPart(this char ch) => ch == '_' || char.IsLetterOrDigit(ch);
}
