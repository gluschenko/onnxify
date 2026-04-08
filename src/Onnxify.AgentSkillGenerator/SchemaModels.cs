using System.Text.Json.Serialization;

namespace Onnxify.AgentSkillGenerator;

internal sealed class OperatorSchemaRoot
{
    [JsonPropertyName("operators")]
    public required List<OperatorSchema> Operators { get; init; }
}

internal sealed class OperatorSchema
{
    [JsonPropertyName("name")]
    public required string Name { get; init; }

    [JsonPropertyName("domain")]
    public required string Domain { get; init; }

    [JsonPropertyName("sinceVersion")]
    public required int SinceVersion { get; init; }

    [JsonPropertyName("doc")]
    public string? Doc { get; init; }

    [JsonPropertyName("attributes")]
    public required List<OperatorAttributeSchema> Attributes { get; init; }

    [JsonPropertyName("inputs")]
    public required List<OperatorParameterSchema> Inputs { get; init; }

    [JsonPropertyName("outputs")]
    public required List<OperatorParameterSchema> Outputs { get; init; }
}

internal sealed class OperatorAttributeSchema
{
    [JsonPropertyName("name")]
    public required string Name { get; init; }

    [JsonPropertyName("description")]
    public string? Description { get; init; }

    [JsonPropertyName("required")]
    public required bool Required { get; init; }

    [JsonPropertyName("type")]
    public required int Type { get; init; }

    [JsonPropertyName("default")]
    public object? Default { get; init; }
}

internal sealed class OperatorParameterSchema
{
    [JsonPropertyName("name")]
    public required string Name { get; init; }

    [JsonPropertyName("description")]
    public string? Description { get; init; }

    [JsonPropertyName("option")]
    public required FormalParameterOption Option { get; init; }

    [JsonPropertyName("minArity")]
    public required int MinArity { get; init; }

    [JsonPropertyName("type")]
    public required string Type { get; init; }

    [JsonPropertyName("types")]
    public required string[] Types { get; init; }
}

internal enum FormalParameterOption : byte
{
    Single = 0,
    Optional = 1,
    Variadic = 2,
}
