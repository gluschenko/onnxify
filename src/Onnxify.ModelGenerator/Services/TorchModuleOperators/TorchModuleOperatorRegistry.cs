using System.Collections.Immutable;

namespace Onnxify.ModelGenerator.Services.TorchModuleOperators;

internal static class TorchModuleOperatorRegistry
{
    internal static ImmutableDictionary<string, TorchModuleOperator> Create()
    {
        var builder = ImmutableDictionary.CreateBuilder<string, TorchModuleOperator>(StringComparer.Ordinal);
        Add(new Conv2dTorchModuleOperator());
        Add(new BatchNorm2dTorchModuleOperator());
        Add(new LinearTorchModuleOperator());
        Add(new AdaptiveAvgPool2dTorchModuleOperator());
        Add(new MaxPool2dTorchModuleOperator());
        Add(new ReluTorchModuleOperator());
        Add(new Relu6TorchModuleOperator());
        return builder.ToImmutable();

        void Add(TorchModuleOperator @operator)
        {
            builder[@operator.OnnxOpType] = @operator;
        }
    }
}
