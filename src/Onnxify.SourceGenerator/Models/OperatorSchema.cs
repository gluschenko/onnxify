using System.Text.Json.Serialization;

namespace Onnxify.SourceGenerator.Models
{
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
        public required string? Doc { get; set; }

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

        [JsonPropertyName("default")]
        public object? Default { get; set; }
    }

    public sealed class OperatorParameter
    {
        [JsonPropertyName("name")]
        public required string Name { get; set; }

        [JsonPropertyName("description")]
        public string? Description { get; set; }

        [JsonPropertyName("option")]
        public required FormalParameterOption Option { get; set; }

        [JsonPropertyName("minArity")]
        public int MinArity { get; set; } = 1;

        [JsonPropertyName("type")]
        public required string Type { get; set; }

        [JsonPropertyName("types")]
        public required string[] Types { get; set; }
    }

    /// <summary>
    /// From schema.h
    /// </summary>
    public enum FormalParameterOption : byte
    {
        /// <summary>
        /// The formal parameter is single and not optional.
        /// Number of supplied actual parameters must be 1.
        /// </summary>
        Single = 0,
        /// <summary>
        /// The formal parameter is single and optional.
        /// Number of supplied actual parameters may be 0 or 1.
        /// </summary>
        Optional = 1,
        /// <summary>
        /// The formal parameter is variadic.
        /// Number of supplied actual parameters must be N or more, where
        /// the minimum value N is indicated separately (default value 1).
        /// </summary>
        Variadic = 2,
    };

    /// <summary>
    /// From schema.h
    /// </summary>
    public enum DifferentiationCategory : byte
    {
        /// <summary>
        /// Whether this formal parameter is differentiable or not cannot
        /// be statically determined. It also covers variadic formal
        /// parameters which contain both of differentiable and
        /// non-differentiable variables.
        /// </summary>
        Unknown = 0,
        /// <summary>
        /// This formal parameter is differentiable. That is, this formal
        /// parameter can be differentiable input of Gradient operator.
        /// </summary>
        Differentiable = 1,
        /// <summary>
        /// This formal parameter is not differentiable. That is, this formal
        /// parameter can not be differentiable input of Gradient operator.
        /// </summary>
        NonDifferentiable = 2
    };
}

