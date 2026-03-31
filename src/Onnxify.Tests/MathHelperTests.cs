using Onnxify.Helpers;

namespace Onnxify.Tests;

public class MathHelperTests
{
    [Fact]
    public void Frexp()
    {
        Assert.Equal(0.8F, MathHelper.Frexp(12.8F, out var exponent));
        Assert.Equal(4, exponent);

        Assert.Equal(0.5F, MathHelper.Frexp(0.25F, out exponent));
        Assert.Equal(-1, exponent);

        Assert.Equal(0.5F, MathHelper.Frexp(MathF.Pow(2F, 127F), out exponent));
        Assert.Equal(128, exponent);

        Assert.Equal(-0.5F, MathHelper.Frexp(-MathF.Pow(2F, -149F), out exponent));
        Assert.Equal(-148, exponent);
    }
}
