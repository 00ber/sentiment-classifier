import torch
from datasets import load_dataset
from train import LABEL_COLUMN, TEXT_COLUMN, SENT2ID, ID2SENT, NUM_LABELS
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSequenceClassification
from time import perf_counter
import numpy as np
from tqdm import tqdm

# Prepare dataset
CUTOFF_LENGTH = 256
dataset = load_dataset("csv", data_files="./data/imdb-dataset.csv", split="train")
dataset = dataset.class_encode_column("sentiment")
dataset = dataset.align_labels_with_mapping(SENT2ID, "sentiment")
dataset = (dataset
    .rename_column("review", TEXT_COLUMN)
    .rename_column("sentiment", LABEL_COLUMN)
)
print("===================== BENCHMARK ============================")
print("Number of samples: ", len(dataset))
print("------------------------------------------------------------")
print("\nLoading models")

base_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
finetuned_model_id = "00BER/imbd-roberta-base-sentiment-merged-latest"
finetuned_model = AutoModelForSequenceClassification.from_pretrained(finetuned_model_id, num_labels=NUM_LABELS, ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained(base_model, add_prefix_space=True)
finetuned_model.eval()

optimized_model_id = "00BER/imbd-roberta-base-sentiment-onxx-latest"
optimized_model = ORTModelForSequenceClassification.from_pretrained(optimized_model_id)

finetuned_pipeline = pipeline("text-classification", model=finetuned_model, tokenizer=tokenizer)
optimized_pipeline = pipeline("text-classification", model=optimized_model, tokenizer=tokenizer)

tokenizer_kwargs = { 'truncation':True, 'max_length':512 }

def measure_latency(pipe):
    latencies = []
    # warm up
    for _ in range(10):
        _ = pipe("It's such a lovely day today!", **tokenizer_kwargs)
    # Timed run
    for i in tqdm(range(len(dataset))):
        start_time = perf_counter()
        _ =  pipe(dataset[i][TEXT_COLUMN], **tokenizer_kwargs)
        latency = perf_counter() - start_time
        latencies.append(latency)
    # Compute run statistics
    time_avg_ms = 1000 * np.mean(latencies)
    time_std_ms = 1000 * np.std(latencies)
    return f"Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f}"

print("\nStarting benchmark:")
print(f"Vanilla model {measure_latency(finetuned_pipeline)}")
print(f"Optimized & Quantized model {measure_latency(optimized_pipeline)}")
