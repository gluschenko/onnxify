namespace Onnxify.AgentSkillGenerator;

internal static class SkillGeneratorPaths
{
    public static string ResolveSkillRoot(string repoRoot)
    {
        string[] relativeCandidates =
        [
            Path.Combine(".skills", "agents", "onnxify"),
            Path.Combine(".agents", "skills", "onnxify"),
        ];

        foreach (string relativeCandidate in relativeCandidates)
        {
            string candidate = Path.Combine(repoRoot, relativeCandidate);
            if (Directory.Exists(candidate))
            {
                return candidate;
            }
        }

        throw new DirectoryNotFoundException(
            "Onnxify skill root was not found. Expected either '.skills/agents/onnxify' or '.agents/skills/onnxify'."
        );
    }

    public static string? FindRepositoryRoot(string? currentDirectory)
    {
        DirectoryInfo? directory = currentDirectory is null ? null : new DirectoryInfo(currentDirectory);

        while (directory is not null)
        {
            bool hasGit = Directory.Exists(Path.Combine(directory.FullName, ".git"));
            bool hasSrc = Directory.Exists(Path.Combine(directory.FullName, "src"));

            if (hasGit && hasSrc)
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        return null;
    }

    public static string NormalizeLineEndings(string value)
    {
        return value.Replace("\r\n", "\n", StringComparison.Ordinal).Replace("\r", "\n", StringComparison.Ordinal);
    }
}
