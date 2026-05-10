using System.Text.RegularExpressions;
using Microsoft.CodeAnalysis;

public static class MarkdownHelper
{
    private static readonly Regex _codeBlockRegex = new(
        @"```(?:\w+)?\s*([\s\S]*?)```",
        RegexOptions.Compiled
    );

    private static readonly Regex _inlineCodeRegex = new(
        @"`([^`\n]+?)`",
        RegexOptions.Compiled
    );

    public static string ToXmlDoc(this string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return string.Empty;
        }

        var blocks = new List<string>();

        text = _codeBlockRegex.Replace(text, m =>
        {
            var content = Escape(m.Groups[1].Value.Trim())
                .Replace("\r\n", "\n")
                .Replace("\n", "&#xA;");

            blocks.Add($"<code>{content}</code>");
            return $"@@CODEBLOCK_{blocks.Count - 1}@@";
        });

        text = _inlineCodeRegex.Replace(text, m =>
        {
            var content = Escape(m.Groups[1].Value);
            blocks.Add($"<c>{content}</c>");
            return $"@@CODEBLOCK_{blocks.Count - 1}@@";
        });

        text = Escape(text);

        for (var i = 0; i < blocks.Count; i++)
        {
            text = text.Replace($"@@CODEBLOCK_{i}@@", blocks[i]);
        }

        var paragraphs = text
            .Split(["\r\n\r\n", "\n\n"], StringSplitOptions.RemoveEmptyEntries)
            .ToArray();

        return string.Join("\n", paragraphs.Length > 1 ? paragraphs.Select(x => $"<para>\n{x.Trim()}\n</para>") : paragraphs);
    }

    private static string Escape(string text)
    {
        return text
            .Replace("&", "&amp;")
            .Replace("<", "&lt;")
            .Replace(">", "&gt;");
    }
}
