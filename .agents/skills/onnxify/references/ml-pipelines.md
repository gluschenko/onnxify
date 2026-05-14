# Onnxify ML Pipelines

Use this reference when the task is not "export a TorchSharp model to ONNX", but "build a reusable inference or training pipeline around data preparation, batching, model execution, metrics, and orchestration".

## Why These Packages Exist

`Onnxify.ML` and `Onnxify.ML.TorchSharp` solve a different problem than the core `Onnxify` graph API:

- `Onnxify` is for building, loading, editing, and saving ONNX graphs.
- `Onnxify.TorchSharp` is for exporting TorchSharp modules into ONNX graphs.
- `Onnxify.ML` is for composing runtime ML workflows as strongly typed pipelines.
- `Onnxify.ML.TorchSharp` is for plugging TorchSharp tensors, forward passes, and optimization steps into those pipelines.

Use these packages when the user wants:

- a clean training loop without a giant handwritten `for` loop
- reusable preprocessing, batching, metrics, or routing stages
- custom batching strategies such as token budgets or shape buckets
- a shared pipeline abstraction for both inference and training
- pipeline-local state, progress reporting, or branching/forking

Do not reach for `Onnxify.ML` when the task is only about ONNX serialization, graph editing, or TorchSharp-to-ONNX conversion.

## Mental Model

The core abstraction is:

- `PipelineStage<TInput, TOutput>` transforms an async stream of `TInput` into an async stream of `TOutput`
- `Pipeline.Begin<TInput>().Then(...).Then(...).Build()` creates a reusable typed pipeline
- `PipelineContext` carries execution-local state, services, and stage progress callbacks

The most important stage families are:

- item-wise stages: `MapStage`, `FilterStage`, `TapStage`, `DelegateStage`
- batching stages: `MiniBatchStage`, `BatchingStage<TInput, TBatch>`, `TorchTensorBatchStage<TSample>`
- orchestration stages: `EpochStage`, `BranchStage`, `ForkStage`
- concurrency stages: `ParallelMapStage`
- TorchSharp execution stages: `TorchInferenceStage`, `TorchTrainingStage`

## How To Think About Stage Boundaries

A healthy pipeline usually separates concerns like this:

- source and preparation: read samples, normalize, tokenize, shape metadata
- batching and collation: decide which samples belong together and produce tensors
- model execution: forward pass for inference or forward/loss/backward/optimizer step for training
- metrics and post-processing: decode logits, aggregate running metrics, emit user-facing results
- orchestration: epochs, fan-out, branching, progress, and checkpoints

If one stage starts doing data loading, device placement, loss computation, logging, and aggregation all at once, it probably wants to be split.

## Minimal Inference Pipeline

The current repo does not yet have a dedicated inference-pipeline sample under `src/Onnxify.Examples`, but the intended usage is already covered by `TorchInferenceStage` in `src/Onnxify.Tests/OnnxifyPipelineTests.cs`.

This is the compact shape to use:

```csharp
using Onnxify.ML;
using Onnxify.ML.TorchSharp;
using Onnxify.ML.TorchSharp.Stages;
using TorchSharp;
using static TorchSharp.torch;
using Tensor = TorchSharp.torch.Tensor;

var pipeline = Pipeline.Begin<float[]>()
    .Then(
        new TorchTensorBatchStage<float[]>(
            batchSize: 2,
            collate: (samples, token) =>
            {
                var flat = tensor(
                    samples.SelectMany(static sample => sample).ToArray(),
                    dtype: ScalarType.Float32
                );

                return ValueTask.FromResult(
                    new TorchBatchTensors(
                        flat.reshape(samples.Count, samples[0].Length)
                    )
                );
            }
        )
    )
    .Then(
        new TorchInferenceStage<TorchMiniBatch<float[]>, Tensor, long[]>(
            forward: (batch, _, _) => ValueTask.FromResult(batch.Inputs.argmax(1)),
            resultSelector: (batch, output, _, _) =>
                ValueTask.FromResult(output.cpu().data<long>().ToArray())
        )
    )
    .Build();

var results = await pipeline.RunToListAsync(
    [
        [1f, 3f],
        [5f, 1f],
        [2f, 9f]
    ]
);

try
{
    var labels = results.Select(static x => x.Result).ToArray();
}
finally
{
    foreach (var item in results)
    {
        item.Dispose();
    }
}
```

What this gives the user:

- batching stays explicit and customizable
- the collate function is the one place where raw samples become tensors
- model execution stays isolated in one stage
- the result projection stage can return labels, probabilities, DTOs, or richer inference payloads
- `TorchInferenceStage` returns disposable `TorchInferenceResult<...>` objects, so callers that materialize results should dispose them after use

## Training Pipeline In Practice

The best current end-to-end example is `src/Onnxify.Examples/Models/AlexNetTrainer.cs`.

The high-level composition looks like this:

```csharp
var pipeline = Pipeline.Begin<AlexNetTrainingRun>()
    .Then(
        new EpochStage<AlexNetTrainingRun>(epochs)
    )
    .Then(new AlexNetBatchSourceStage())
    .Then(new AlexNetBatchTensorStage())
    .Then(
        new TorchTrainingStage<AlexNetTorchBatch, Tensor, AlexNetBatchMetrics>(
            optimizer,
            forward: (batch, _, _) => ValueTask.FromResult(batch.Run.Model.forward(batch.Inputs)),
            lossSelector: (batch, output, _, _) => ValueTask.FromResult(batch.Run.Criterion.call(output, batch.Targets)),
            resultSelector: (batch, output, loss, _, _) =>
            {
                using var predicted = output.argmax(1);
                using var correct = predicted.eq(batch.Targets);

                return ValueTask.FromResult(
                    new AlexNetBatchMetrics
                    {
                        BatchSize = batch.Source.Size,
                        CorrectPredictions = correct.sum().ToInt32()
                    }
                );
            }
        )
    )
    .Then(new AlexNetBatchProgressStage())
    .Build();
```

This is a good reference because it shows the intended split:

- `EpochStage` handles replay and orchestration
- `AlexNetBatchSourceStage` turns an epoch item into dataset batches
- `AlexNetBatchTensorStage` materializes tensors and performs device placement
- `TorchTrainingStage` owns forward, loss, backward, and optimizer step
- `AlexNetBatchProgressStage` turns raw step results into running metrics for logs or UI

## Example Of A Custom Stage

`AlexNetTrainer` is also useful because it shows what a custom stage should look like when the built-in primitives are not enough.

This source stage emits one work item per batch inside the current epoch:

```csharp
private sealed class AlexNetBatchSourceStage
    : PipelineStage<EpochItem<AlexNetTrainingRun>, AlexNetBatchWorkItem>
{
    public override async IAsyncEnumerable<AlexNetBatchWorkItem> ExecuteAsync(
        IAsyncEnumerable<EpochItem<AlexNetTrainingRun>> input,
        PipelineContext context,
        [EnumeratorCancellation] CancellationToken token)
    {
        await foreach (var epoch in input.WithCancellation(token))
        {
            var run = epoch.Value;
            var batchIndex = 0;

            await foreach (var batch in run.Reader.BatchAsync(
                run.BatchSize,
                shuffle: true,
                shuffleSeed: run.ShuffleSeed + epoch.EpochIndex,
                cancellationToken: token))
            {
                batchIndex++;

                yield return new AlexNetBatchWorkItem
                {
                    Run = run,
                    EpochIndex = epoch.EpochIndex,
                    EpochNumber = epoch.EpochNumber,
                    BatchNumber = batchIndex,
                    Source = batch,
                    // other metadata omitted for brevity
                };
            }
        }
    }
}
```

Use this pattern when a stage must:

- emit zero, one, or many outputs per input item
- integrate with an existing async dataset reader
- attach orchestration metadata such as epoch, batch, learning rate, or shard info

## When To Use TorchTensorBatchStage vs MiniBatchStage

Use `MiniBatchStage<T>` when:

- you only need grouped raw samples
- tensor materialization happens later
- you are building a framework-agnostic pipeline

Use `TorchTensorBatchStage<T>` when:

- batching and collation naturally belong together
- the next stage expects ready-to-run Torch tensors
- you want a single place that decides tensor shapes, dtypes, and optional targets

## Progress And Execution Context

Use `PipelineContext` for execution-local state such as:

- metric accumulators
- lookup tables or service-provider access
- per-run settings
- counters or user-visible progress callbacks

For user-facing progress:

- report operational progress from stages, not just "epoch finished"
- prefer emitting per-batch metrics after each optimization step for training UIs
- keep stage names meaningful because they surface in diagnostics and progress reporting

## Extension Guidance

When adding new reusable stages:

- prefer one narrow responsibility per stage
- keep inputs and outputs strongly typed
- make batching rules explicit instead of hiding them in model code
- put TorchSharp-specific execution in `Onnxify.ML.TorchSharp`, not `Onnxify.ML`
- if a stage is domain-specific to one example or model family, keep it in `src/Onnxify.Examples` until a broader abstraction is obvious

Good candidates for future reusable stages:

- bucketing by token count or sequence length
- padded sequence collation
- device-transfer helpers
- checkpoint and metric observer stages
- eval-only post-processing stages such as thresholding or decoding

## Where To Start In This Repo

- training composition example: `src/Onnxify.Examples/Models/AlexNetTrainer.cs`
- pipeline behavior tests: `src/Onnxify.Tests/OnnxifyPipelineTests.cs`
- core pipeline API: `src/Onnxify.ML`
- TorchSharp pipeline integration: `src/Onnxify.ML.TorchSharp`

If the user asks for a new training or inference workflow, start by deciding whether it is:

- mostly graph or export work: use `Onnxify` or `Onnxify.TorchSharp`
- mostly runtime pipeline work: use `Onnxify.ML` or `Onnxify.ML.TorchSharp`
