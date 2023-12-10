import torch
from datasets import load_dataset
from train import LABEL_COLUMN, TEXT_COLUMN, SENT2ID, ID2SENT, NUM_LABELS
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from time import perf_counter
import numpy as np

# Prepare dataset
CUTOFF_LENGTH = 256
dataset = load_dataset("csv", data_files="./data/imdb-dataset.csv", split="train")
dataset = dataset.class_encode_column("sentiment")
dataset = dataset.align_labels_with_mapping(SENT2ID, "sentiment")
dataset = (dataset
    .rename_column("review", TEXT_COLUMN)
    .rename_column("sentiment", LABEL_COLUMN)
)
split_ds = dataset.train_test_split(test_size=0.001)
print(len(split_ds["test"]))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

base_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
peft_model = "00BER/imbd-roberta-base-sentiment-latest"
config = PeftConfig.from_pretrained(peft_model)
model_pre_improvement = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=NUM_LABELS, ignore_mismatched_sizes=True)
model_pre_improvement = PeftModel.from_pretrained(model_pre_improvement, peft_model)
tokenizer = AutoTokenizer.from_pretrained(base_model, add_prefix_space=True)
model_pre_improvement.eval()

def get_prediction(model, tokenizer, text):
    tokens = tokenizer(text, truncation=True, max_length=CUTOFF_LENGTH, return_tensors="pt")
    output = model(**tokens)
    result = output.logits.argmax(-1)
    return ID2SENT[result.item()]

def measure_latency(model, tokenizer):
    latencies = []
    # warm up
    for _ in range(10):
        _ = get_prediction(model, tokenizer, "It's such a lovely day today!")
    # Timed run
    for i in range(len(split_ds["test"])):
        start_time = perf_counter()
        _ =  get_prediction(model, tokenizer, split_ds["test"][i][TEXT_COLUMN])
        latency = perf_counter() - start_time
        latencies.append(latency)
    # Compute run statistics
    time_avg_ms = 1000 * np.mean(latencies)
    time_std_ms = 1000 * np.std(latencies)
    return f"Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f}"

print(f"Vanilla model {measure_latency(model_pre_improvement, tokenizer)}")
# print(f"Optimized & Quantized model {measure_latency(quantized_optimum_qa)}")

# Vanilla model Average latency (ms) - 117.61 +\- 8.48
# Optimized & Quantized model Average latency (ms) - 64.94 +\- 3.65
