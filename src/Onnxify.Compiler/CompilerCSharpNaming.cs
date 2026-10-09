using System.Globalization;
using System.Text;
using Microsoft.CodeAnalysis.CSharp;

namespace Onnxify.Compiler;

internal static class CompilerCSharpNaming
{
    public static string Identifier(string value)
    {
        var builder = new StringBuilder();
        for (var index = 0; index < value.Length; index++)
        {
            var character = value[index];
            var isEncodedSequence = character == '_'
                && index + 6 < value.Length
                && value[index + 1] == 'u'
                && value.AsSpan(index + 2, 4).ToString().All(Uri.IsHexDigit)
                && value[index + 6] == '_';

            if (isEncodedSequence)
            {
                AppendEscaped(builder, character);
            }
            else if ((index == 0 ? char.IsLetter(character) || character == '_' : char.IsLetterOrDigit(character) || character == '_'))
            {
                builder.Append(character);
            }
            else
            {
                AppendEscaped(builder, character);
            }
        }

        if (builder.Length == 0)
        {
            builder.Append('_');
        }

        var identifier = builder.ToString();
        return SyntaxFacts.GetKeywordKind(identifier) == SyntaxKind.None
            ? identifier
            : $"@{identifier}";
    }

    private static void AppendEscaped(StringBuilder builder, char character)
    {
        builder.Append("_u");
        builder.Append(((int)character).ToString("X4", CultureInfo.InvariantCulture));
        builder.Append('_');
    }
}
