using System.Text.Json.Serialization;
using Onnxify.Legacy;

public sealed class OperatorSchemaRoot
{
    [JsonPropertyName("operators")]
    public required List<OperatorSchema> Operators { get; set; }
}

public sealed class OperatorSchema
{
    [JsonPropertyName("name")]
    public required string Name { get; set; }

    [JsonPropertyName("domain")]
    public required string Domain { get; set; }

    [JsonPropertyName("sinceVersion")]
    public required int SinceVersion { get; set; }

    [JsonPropertyName("doc")]
    public string? Doc { get; set; }

    [JsonPropertyName("attributes")]
    public required List<OperatorAttribute> Attributes { get; set; }

    [JsonPropertyName("inputs")]
    public required List<OperatorParameter> Inputs { get; set; }

    [JsonPropertyName("outputs")]
    public required List<OperatorParameter> Outputs { get; set; }
}

public sealed class OperatorAttribute
{
    [JsonPropertyName("name")]
    public required string Name { get; set; }

    [JsonPropertyName("description")]
    public string? Description { get; set; }

    [JsonPropertyName("required")]
    public required bool Required { get; set; }

    [JsonPropertyName("type")]
    public required int Type { get; set; }
}

public sealed class OperatorParameter
{
    [JsonPropertyName("name")]
    public required string Name { get; set; }

    [JsonPropertyName("option")]
    public required FormalParameterOption Option { get; set; }

    [JsonPropertyName("type")]
    public required string Type { get; set; }
}

