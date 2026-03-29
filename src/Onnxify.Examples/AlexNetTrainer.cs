using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Onnxify.Examples
{
    internal class AlexNetTrainer
    {
        private readonly AlexNet _model;
        private readonly DataReader _reader;

        public AlexNetTrainer(AlexNet model, DataReader reader)
        {
            _model = model;
            _reader = reader;
        }

        public void Train(
            int epochs = 5,
            int batchSize = 32,
            float learningRate = 1e-3f,
            Device? device = null
        )
        {
            device ??= torch.cuda.is_available() ? CUDA : CPU;

            _model.to(device);
            _model.train();

            var optimizer = optim.Adam(_model.parameters(), learningRate);
            var criterion = CrossEntropyLoss();

            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                Console.WriteLine($"Epoch {epoch}/{epochs}");

                var batchData = new List<Tensor>();
                var batchLabels = new List<Tensor>();

                int batchIndex = 0;

                foreach (var (data, label) in _reader.Data())
                {
                    batchData.Add(data);
                    batchLabels.Add(label);

                    if (batchData.Count == batchSize)
                    {
                        TrainBatch(
                            batchData,
                            batchLabels,
                            optimizer,
                            criterion,
                            device,
                            ref batchIndex
                        );

                        batchData.Clear();
                        batchLabels.Clear();
                    }
                }

                // хвост
                if (batchData.Count > 0)
                {
                    TrainBatch(
                        batchData,
                        batchLabels,
                        optimizer,
                        criterion,
                        device,
                        ref batchIndex
                    );
                }
            }
        }

        private void TrainBatch(
            List<Tensor> batchData,
            List<Tensor> batchLabels,
            optim.Optimizer optimizer,
            Loss<torch.Tensor, torch.Tensor, torch.Tensor> criterion,
            Device device,
            ref int batchIndex
        )
        {
            using var x = torch.stack(batchData).to(device);     // [N, C, H, W]
            using var y = torch.stack(batchLabels).to(device).view(-1); // [N]

            optimizer.zero_grad();

            using var output = _model.forward(x);

            using var loss = criterion.call(output, y);

            loss.backward();
            optimizer.step();

            if (batchIndex % 10 == 0)
            {
                Console.WriteLine($"Batch {batchIndex} | Loss: {loss.ToSingle():F4}");
            }

            batchIndex++;
        }
    }
}
