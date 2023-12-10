from optimum.onnxruntime import ORTModelForSequenceClassification
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer


base_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
peft_model = "00BER/imbd-roberta-base-sentiment-latest"
config = PeftConfig.from_pretrained(peft_model)
model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2, ignore_mismatched_sizes=True)
model = PeftModel.from_pretrained(model, peft_model)
tokenizer = AutoTokenizer.from_pretrained(base_model, add_prefix_space=True)

# ----------------------------------------------------------------------------------
# Merge LoRA weights into the base model.
merged_model = model.merge_and_unload()
merged_model_path = "models/imbd-roberta-base-sentiment-merged-latest"
merged_model.save_pretrained(merged_model_path, save_adapter=True, save_config=True)

# ----------------------------------------------------------------------------------
# ONNX conversion

ort_model = ORTModelForSequenceClassification.from_pretrained(
    merged_model_path,
    use_io_binding=True,
    export=True,
    # use_cache=True,
    from_transformers=True,
    provider="CPUExecutionProvider",  # Change this to "CPUExecutionProvider" using CPU for inference
)

onxx_model_path = "models/imbd-roberta-base-sentiment-onxx-latest"
ort_model.save_pretrained(onxx_model_path)
