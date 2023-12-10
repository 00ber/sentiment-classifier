from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig

# create ORTOptimizer and define optimization configuration
task = "text-classification"
optimized_path = "models/imbd-roberta-base-sentiment-onxx-optimized-latest"
optimizer = ORTOptimizer.from_pretrained(onxx_model_path = "models/imbd-roberta-base-sentiment-onxx-latest", feature=task)
optimization_config = OptimizationConfig(optimization_level=99) # enable all optimizations

# apply the optimization configuration to the model
optimizer.export(
    onnx_model_path=optimized_path / "model.onnx",
    onnx_optimized_model_output_path=optimized_path / "model-optimized.onnx",
    optimization_config=optimization_config,
)
