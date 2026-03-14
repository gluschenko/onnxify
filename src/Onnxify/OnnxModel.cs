namespace Onnxify
{
    public class OnnxModel
    {
        
    }

    public abstract class OnnxNodeX
    {
        public string Name { get; private set; }

        protected OnnxNodeX(string name)
        {
            Name = name;
        }
    }

    public class OnnxTensor
    {

    }
}
