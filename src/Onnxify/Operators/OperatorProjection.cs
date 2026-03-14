using Onnxify.Abstractions;

namespace Onnxify.Operators
{
    public class OperatorProjection
    {
        public required string Name { get; set; }
        public required string Domain { get; set; }
        public required int SinceVersion { get; set; }
        public required IEnumerable<IFormalParameter> Inputs { get; set; }
        public required IEnumerable<IFormalParameter> Outputs { get; set; }
        public required IEnumerable<IOperatorAttribute> Attributes { get; set; }
    }
}
