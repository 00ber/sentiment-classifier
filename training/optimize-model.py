from transformers import AutoTokenizer
from optimum.onnxruntime import (
    AutoOptimizationConfig,
    ORTModelForSequenceClassification,
    ORTOptimizer
)
from optimum.onnxruntime.configuration import OptimizationConfig
from optimum.pipelines import pipeline
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# Load the tokenizer and export the model to the ONNX format
model_id = "models/imbd-roberta-base-sentiment-merged-latest"
save_dir = "models/onxx-"

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)

# Load the optimization configuration detailing the optimization we wish to apply
optimization_config = AutoOptimizationConfig.O3()
optimizer = ORTOptimizer.from_pretrained(model)

optimized_ver = optimizer.optimize(save_dir=save_dir + "-optimized", optimization_config=optimization_config)

qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
quantizer = ORTQuantizer.from_pretrained(optimized_ver)
quantizer.quantize(save_dir=save_dir + "-quantized", quantization_config=qconfig)

tokenizer.save_pretrained("models/onxx-optimized-quantized-tokenizer")
model.save_pretrained("models/onxx-optimized-quantized")

model = ORTModelForSequenceClassification.from_pretrained(save_dir)
onnx_clx = pipeline("text-classification", model=model, tokenizer=tokenizer, accelerator="ort")
text = "I like the new ORT pipeline"
pred = onnx_clx(text)
print(pred)
