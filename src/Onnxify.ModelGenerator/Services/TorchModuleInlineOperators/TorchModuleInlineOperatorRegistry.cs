using System.Collections.Immutable;

namespace Onnxify.ModelGenerator.Services.TorchModuleInlineOperators;

internal static class TorchModuleInlineOperatorRegistry
{
    internal static ImmutableDictionary<string, TorchModuleInlineOperator> Create()
    {
        var builder = ImmutableDictionary.CreateBuilder<string, TorchModuleInlineOperator>(StringComparer.Ordinal);
        Add(new AddTorchModuleInlineOperator());
        Add(new SubTorchModuleInlineOperator());
        Add(new MulTorchModuleInlineOperator());
        Add(new DivTorchModuleInlineOperator());
        Add(new PowTorchModuleInlineOperator());
        Add(new AbsTorchModuleInlineOperator());
        Add(new NegTorchModuleInlineOperator());
        Add(new ExpTorchModuleInlineOperator());
        Add(new LogTorchModuleInlineOperator());
        Add(new SqrtTorchModuleInlineOperator());
        Add(new FloorTorchModuleInlineOperator());
        Add(new CeilTorchModuleInlineOperator());
        Add(new RoundTorchModuleInlineOperator());
        Add(new SignTorchModuleInlineOperator());
        Add(new SinTorchModuleInlineOperator());
        Add(new CosTorchModuleInlineOperator());
        Add(new TanTorchModuleInlineOperator());
        Add(new AcosTorchModuleInlineOperator());
        Add(new AcoshTorchModuleInlineOperator());
        Add(new AsinTorchModuleInlineOperator());
        Add(new AsinhTorchModuleInlineOperator());
        Add(new AtanTorchModuleInlineOperator());
        Add(new AtanhTorchModuleInlineOperator());
        Add(new ErfTorchModuleInlineOperator());
        Add(new ReciprocalTorchModuleInlineOperator());
        Add(new SigmoidTorchModuleInlineOperator());
        Add(new TanhTorchModuleInlineOperator());
        Add(new EluTorchModuleInlineOperator());
        Add(new HardSigmoidTorchModuleInlineOperator());
        Add(new LeakyReluTorchModuleInlineOperator());
        Add(new SoftmaxTorchModuleInlineOperator());
        Add(new IdentityTorchModuleInlineOperator());
        Add(new CastTorchModuleInlineOperator());
        Add(new MatMulTorchModuleInlineOperator());
        Add(new MaxTorchModuleInlineOperator());
        Add(new MinTorchModuleInlineOperator());
        Add(new GemmTorchModuleInlineOperator());
        Add(new ReshapeTorchModuleInlineOperator());
        Add(new FlattenTorchModuleInlineOperator());
        Add(new LrnTorchModuleInlineOperator());
        Add(new AveragePoolTorchModuleInlineOperator());
        Add(new ArgMaxTorchModuleInlineOperator());
        Add(new ArgMinTorchModuleInlineOperator());
        Add(new CeluTorchModuleInlineOperator());
        Add(new CumSumTorchModuleInlineOperator());
        Add(new DepthToSpaceTorchModuleInlineOperator());
        Add(new DropoutTorchModuleInlineOperator());
        Add(new ExpandTorchModuleInlineOperator());
        Add(new GatherElementsTorchModuleInlineOperator());
        Add(new GeluTorchModuleInlineOperator());
        Add(new GroupNormalizationTorchModuleInlineOperator());
        Add(new HardSwishTorchModuleInlineOperator());
        Add(new InstanceNormalizationTorchModuleInlineOperator());
        Add(new LayerNormalizationTorchModuleInlineOperator());
        Add(new LogSoftmaxTorchModuleInlineOperator());
        Add(new MishTorchModuleInlineOperator());
        Add(new PReluTorchModuleInlineOperator());
        Add(new PadTorchModuleInlineOperator());
        Add(new ReduceMaxTorchModuleInlineOperator());
        Add(new ReduceMinTorchModuleInlineOperator());
        Add(new ReduceProdTorchModuleInlineOperator());
        Add(new ResizeTorchModuleInlineOperator());
        Add(new SeluTorchModuleInlineOperator());
        Add(new SliceTorchModuleInlineOperator());
        Add(new SoftplusTorchModuleInlineOperator());
        Add(new SpaceToDepthTorchModuleInlineOperator());
        Add(new SplitTorchModuleInlineOperator());
        Add(new TileTorchModuleInlineOperator());
        Add(new TopKTorchModuleInlineOperator());
        Add(new TriluTorchModuleInlineOperator());
        Add(new TransposeTorchModuleInlineOperator());
        Add(new ClipTorchModuleInlineOperator());
        Add(new ConvTorchModuleInlineOperator());
        Add(new MaxPoolTorchModuleInlineOperator());
        Add(new BatchNormalizationTorchModuleInlineOperator());
        Add(new GlobalAveragePoolTorchModuleInlineOperator());
        Add(new ShapeTorchModuleInlineOperator());
        Add(new GatherTorchModuleInlineOperator());
        Add(new SqueezeTorchModuleInlineOperator());
        Add(new UnsqueezeTorchModuleInlineOperator());
        Add(new ConcatTorchModuleInlineOperator());
        Add(new ReduceMeanTorchModuleInlineOperator());
        Add(new ReduceSumTorchModuleInlineOperator());
        Add(new GreaterTorchModuleInlineOperator());
        Add(new LessTorchModuleInlineOperator());
        Add(new GreaterOrEqualTorchModuleInlineOperator());
        Add(new LessOrEqualTorchModuleInlineOperator());
        Add(new EqualTorchModuleInlineOperator());
        Add(new WhereTorchModuleInlineOperator());
        Add(new NotTorchModuleInlineOperator());
        Add(new ConstantTorchModuleInlineOperator());
        Add(new QuantizeLinearTorchModuleInlineOperator());
        Add(new DequantizeLinearTorchModuleInlineOperator());
        return builder.ToImmutable();

        void Add(TorchModuleInlineOperator @operator)
        {
            builder[@operator.OnnxOpType] = @operator;
        }
    }
}
