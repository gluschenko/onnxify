using System;
using Onnxify.Safetensors;
using Onnxify.TorchSharp;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace Onnxify.Tests;

public sealed class TorchModuleSafetensorsExtensionsTests
{
    [Theory]
    [InlineData(ScalarType.Byte)]
    [InlineData(ScalarType.Int8)]
    [InlineData(ScalarType.Int16)]
    [InlineData(ScalarType.Int32)]
    [InlineData(ScalarType.Int64)]
    [InlineData(ScalarType.Float16)]
    [InlineData(ScalarType.Float32)]
    [InlineData(ScalarType.Float64)]
    [InlineData(ScalarType.ComplexFloat32)]
    [InlineData(ScalarType.ComplexFloat64)]
    [InlineData(ScalarType.Bool)]
    [InlineData(ScalarType.BFloat16)]
    public void CreateTensorView_AndCopyTensorData_RoundTripSupportedScalarTypes(ScalarType dtype)
    {
        using var source = CreateTensor(dtype);
        var sourceView = TorchModuleSafetensorsExtensions.CreateTensorView(source);

        using var target = torch.empty(source.shape, dtype: source.dtype);
        TorchModuleSafetensorsExtensions.CopyTensorData(target, sourceView);

        var roundTripView = TorchModuleSafetensorsExtensions.CreateTensorView(target);

        Assert.Equal(sourceView.DataType, roundTripView.DataType);
        Assert.Equal(sourceView.Shape, roundTripView.Shape);
        Assert.Equal(sourceView.Data.ToArray(), roundTripView.Data.ToArray());
    }

    [Fact]
    public void CreateTensorView_ForComplexFloat64_UsesExpandedF64Shape()
    {
        using var tensor = CreateTensor(ScalarType.ComplexFloat64);

        var view = TorchModuleSafetensorsExtensions.CreateTensorView(tensor);

        Assert.Equal(DataType.F64, view.DataType);
        Assert.Equal([2UL, 2UL], view.Shape);
    }

    private static Tensor CreateTensor(ScalarType dtype)
    {
        return dtype switch
        {
            ScalarType.Byte => torch.tensor(new byte[] { 1, 2, 3, 4 }, new long[] { 2, 2 }, dtype: dtype),
            ScalarType.Int8 => torch.tensor(new sbyte[] { -1, 2, -3, 4 }, new long[] { 2, 2 }, dtype: dtype),
            ScalarType.Int16 => torch.tensor(new short[] { -1, 2, -3, 4 }, new long[] { 2, 2 }, dtype: dtype),
            ScalarType.Int32 => torch.tensor(new int[] { -1, 2, -3, 4 }, new long[] { 2, 2 }, dtype: dtype),
            ScalarType.Int64 => torch.tensor(new long[] { -1, 2, -3, 4 }, new long[] { 2, 2 }, dtype: dtype),
            ScalarType.Float16 => torch.tensor(new float[] { 1.5f, -2.25f, 0.5f, 4.5f }, new long[] { 2, 2 }, dtype: dtype),
            ScalarType.Float32 => torch.tensor(new float[] { 1.5f, -2.25f, 0.5f, 4.5f }, new long[] { 2, 2 }, dtype: dtype),
            ScalarType.Float64 => torch.tensor(new double[] { 1.5, -2.25, 0.5, 4.5 }, new long[] { 2, 2 }, dtype: dtype),
            ScalarType.ComplexFloat32 => torch.tensor(
                    new float[] { 1.5f, -2.25f, 0.5f, 4.5f },
                    new long[] { 2, 2 },
                    dtype: ScalarType.Float32)
                .view_as_complex(),
            ScalarType.ComplexFloat64 => torch.tensor(
                    new double[] { 1.5, -2.25, 0.5, 4.5 },
                    new long[] { 2, 2 },
                    dtype: ScalarType.Float64)
                .view_as_complex(),
            ScalarType.Bool => torch.tensor(new bool[] { true, false, false, true }, new long[] { 2, 2 }, dtype: dtype),
            ScalarType.BFloat16 => torch.tensor(new float[] { 1.5f, -2.25f, 0.5f, 4.5f }, new long[] { 2, 2 }, dtype: dtype),
            _ => throw new ArgumentOutOfRangeException(nameof(dtype), dtype, null),
        };
    }
}
