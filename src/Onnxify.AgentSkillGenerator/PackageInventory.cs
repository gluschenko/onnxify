using System.Text;
using System.Xml.Linq;

namespace Onnxify.AgentSkillGenerator;

internal sealed class PackageInventory
{
    private PackageInventory(IReadOnlyList<PackageDoc> packages)
    {
        Packages = packages;
    }

    public IReadOnlyList<PackageDoc> Packages { get; }

    public static PackageInventory Load(string repoRoot)
    {
        string srcRoot = Path.Combine(repoRoot, "src");
        var packages = Directory
            .EnumerateFiles(srcRoot, "Onnxify*.csproj", SearchOption.AllDirectories)
            .Select(path => LoadProject(srcRoot, path))
            .Where(project => project.IsPackable && (project.Name.StartsWith("Onnxify.", StringComparison.Ordinal) || project.Name == "Onnxify"))
            .OrderBy(project => project.Name, StringComparer.Ordinal)
            .ToArray();

        var packageNames = packages
            .Select(project => project.Name)
            .ToHashSet(StringComparer.Ordinal);

        var packageDocs = packages
            .Select(project =>
            {
                var onnxifyDependencies = project.ProjectReferences
                    .Select(reference => ResolveProjectReference(srcRoot, project.ProjectDirectory, reference))
                    .Where(reference => reference is not null && packageNames.Contains(reference))
                    .Select(reference => reference!)
                    .Distinct(StringComparer.Ordinal)
                    .OrderBy(reference => reference, StringComparer.Ordinal)
                    .ToArray();

                var thirdPartyDependencies = project.PackageReferences
                    .Where(reference => !reference.Name.StartsWith("Onnxify.", StringComparison.Ordinal) && reference.Name != "Onnxify")
                    .Distinct()
                    .OrderBy(reference => reference.Name, StringComparer.Ordinal)
                    .ThenBy(reference => reference.Version, StringComparer.Ordinal)
                    .ToArray();

                return new PackageDoc(
                    project.Name,
                    project.Version,
                    onnxifyDependencies,
                    thirdPartyDependencies
                );
            })
            .ToArray();

        return new PackageInventory(packageDocs);
    }

    public string BuildOverviewMarkdown()
    {
        var builder = new StringBuilder();
        builder.AppendLine("## NuGet Package Versions");
        builder.AppendLine();
        builder.AppendLine("| Package | Version | Onnxify dependencies | Third-party NuGet dependencies |");
        builder.AppendLine("| --- | --- | --- | --- |");

        foreach (PackageDoc package in Packages)
        {
            builder.Append("| `")
                .Append(EscapeMarkdownCell(package.Name))
                .Append("` | `")
                .Append(EscapeMarkdownCell(package.Version))
                .Append("` | ")
                .Append(FormatOnnxifyDependencies(package.OnnxifyDependencies))
                .Append(" | ")
                .Append(FormatThirdPartyDependencies(package.ThirdPartyDependencies))
                .AppendLine(" |");
        }

        return builder.ToString().TrimEnd();
    }

    public static string BuildPackageContextMarkdown(IReadOnlyList<PackageDoc> packages)
    {
        var builder = new StringBuilder();
        builder.AppendLine("## Package Context");
        builder.AppendLine();
        builder.AppendLine("| Package | Version | Onnxify dependencies | Third-party NuGet dependencies |");
        builder.AppendLine("| --- | --- | --- | --- |");

        foreach (PackageDoc package in packages)
        {
            builder.Append("| `")
                .Append(EscapeMarkdownCell(package.Name))
                .Append("` | `")
                .Append(EscapeMarkdownCell(package.Version))
                .Append("` | ")
                .Append(FormatOnnxifyDependencies(package.OnnxifyDependencies))
                .Append(" | ")
                .Append(FormatThirdPartyDependencies(package.ThirdPartyDependencies))
                .AppendLine(" |");
        }

        return builder.ToString().TrimEnd();
    }

    private static ProjectDoc LoadProject(string srcRoot, string projectPath)
    {
        XDocument document = XDocument.Load(projectPath);
        XElement project = document.Root ?? throw new InvalidOperationException($"Project file '{projectPath}' has no root element.");
        string name = Path.GetFileNameWithoutExtension(projectPath);
        string version = ReadProperty(project, "Version") ?? "[not specified]";
        bool isPackable = !string.Equals(ReadProperty(project, "IsPackable"), "false", StringComparison.OrdinalIgnoreCase);

        var packageReferences = project
            .Descendants("PackageReference")
            .Select(element => new PackageReferenceDoc(
                ReadAttributeOrChild(element, "Include"),
                ReadAttributeOrChild(element, "Version"),
                ReadAttributeOrChild(element, "PrivateAssets"),
                ReadAttributeOrChild(element, "Condition")))
            .Where(reference => !string.IsNullOrWhiteSpace(reference.Name))
            .ToArray();

        var projectReferences = project
            .Descendants("ProjectReference")
            .Select(element => ReadAttributeOrChild(element, "Include"))
            .Where(value => !string.IsNullOrWhiteSpace(value))
            .Cast<string>()
            .ToArray();

        return new ProjectDoc(
            name,
            version,
            isPackable,
            Path.GetDirectoryName(projectPath) ?? srcRoot,
            packageReferences,
            projectReferences
        );
    }

    private static string? ResolveProjectReference(string srcRoot, string projectDirectory, string reference)
    {
        string fullPath = Path.GetFullPath(Path.Combine(projectDirectory, reference));
        if (!fullPath.StartsWith(srcRoot, StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        return Path.GetFileNameWithoutExtension(fullPath);
    }

    private static string? ReadProperty(XElement project, string name)
    {
        return project
            .Descendants(name)
            .Select(element => element.Value.Trim())
            .FirstOrDefault(value => !string.IsNullOrWhiteSpace(value));
    }

    private static string ReadAttributeOrChild(XElement element, string name)
    {
        string? value = element.Attribute(name)?.Value;
        if (!string.IsNullOrWhiteSpace(value))
        {
            return value.Trim();
        }

        return element.Element(name)?.Value.Trim() ?? string.Empty;
    }

    private static string FormatOnnxifyDependencies(IReadOnlyList<string> dependencies)
    {
        if (dependencies.Count == 0)
        {
            return "[none]";
        }

        return string.Join("<br>", dependencies.Select(dependency => $"`{EscapeMarkdownCell(dependency)}`"));
    }

    private static string FormatThirdPartyDependencies(IReadOnlyList<PackageReferenceDoc> dependencies)
    {
        if (dependencies.Count == 0)
        {
            return "[none]";
        }

        return string.Join("<br>", dependencies.Select(FormatThirdPartyDependency));
    }

    private static string FormatThirdPartyDependency(PackageReferenceDoc dependency)
    {
        var suffixes = new List<string>();
        if (!string.IsNullOrWhiteSpace(dependency.PrivateAssets))
        {
            suffixes.Add($"PrivateAssets={dependency.PrivateAssets}");
        }

        if (!string.IsNullOrWhiteSpace(dependency.Condition))
        {
            suffixes.Add($"Condition={dependency.Condition}");
        }

        string suffix = suffixes.Count == 0
            ? string.Empty
            : " (" + string.Join("; ", suffixes) + ")";

        return FormattableString.Invariant(
            $"`{EscapeMarkdownCell(dependency.Name)}` `{EscapeMarkdownCell(dependency.Version)}`{EscapeMarkdownCell(suffix)}"
        );
    }

    private static string EscapeMarkdownCell(string value)
    {
        return value
            .Replace("|", "\\|", StringComparison.Ordinal)
            .Replace("\r", " ", StringComparison.Ordinal)
            .Replace("\n", " ", StringComparison.Ordinal);
    }

    private sealed record ProjectDoc(
        string Name,
        string Version,
        bool IsPackable,
        string ProjectDirectory,
        IReadOnlyList<PackageReferenceDoc> PackageReferences,
        IReadOnlyList<string> ProjectReferences
    );

    public sealed record PackageDoc(
        string Name,
        string Version,
        IReadOnlyList<string> OnnxifyDependencies,
        IReadOnlyList<PackageReferenceDoc> ThirdPartyDependencies
    );

    public sealed record PackageReferenceDoc(
        string Name,
        string Version,
        string PrivateAssets,
        string Condition
    );
}
