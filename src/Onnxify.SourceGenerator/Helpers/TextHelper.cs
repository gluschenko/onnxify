using System.Text;

public static class TextHelper
{
    public static string PascalCase(this string s)
    {
        if (string.IsNullOrEmpty(s))
        {
            return s;
        }

        var words = s.Split(['_'], StringSplitOptions.RemoveEmptyEntries);

        var result = new StringBuilder();

        foreach (var word in words)
        {
            var newWord = char.ToUpperInvariant(word[0]) + word.Substring(1);
            result.Append(newWord);
        }

        return result.ToString();
    }

    public static string Comment(this string text)
    {
        return text.Trim().ToXmlDoc().Replace("\n", $"\n/// ").Trim();
    }

    public static string Indent(this string text, int tabs)
    {
        var indent = new string(' ', tabs * 4);
        return text.Trim().Replace("\n", $"\n{indent}").Trim();
    }
}
