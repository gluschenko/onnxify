namespace Onnxify
{
    public class OnnxModel
    {
        
    }

    public abstract class OnnxNode
    {
        public string Name { get; private set; }

        protected OnnxNode(string name)
        {
            Name = name;
        }
    }

    public class OnnxTensor
    {

    }
}
