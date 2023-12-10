import os
import argparse
import torch
import evaluate
import numpy as np
from datasets import load_dataset
from distutils.util import strtobool
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from peft import get_peft_model, LoraConfig, TaskType
from transformers import TrainingArguments, Trainer

# https://colab.research.google.com/drive/1zVsxiFLxdvuGL-UMJukE7akZyeM-VNoU#scrollTo=0sM7hedHOIYb
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="cardiffnlp/twitter-roberta-base-sentiment-latest",
        help="the base BERT based model to use for training")
    parser.add_argument("--csv-path", type=str, default="./data/imdb-dataset.csv",
        help="the filepath of the IMDB Dataset")
    parser.add_argument("--out-dir", type=str,
        help="the directory to save model checkpoints in")
    parser.add_argument("--push-to-hub", type=lambda x: bool(strtobool(x)), default=True,
        help="whether or not to push model to hub")
    parser.add_argument("--hf-repository", type=str, default=None,
        help="if --push-to-hub is True, specifies which repository to push to")
    parser.add_argument("--save-train-test-splits", type=lambda x: bool(strtobool(x)), default=True,
        help="whether to save the train-test splits to disk")
    parser.add_argument("--train-test-save-dir", type=str, default=None,
        help="directory to save train-test splits to")
    
    # Hyperparameters
    parser.add_argument("--learning-rate", type=float, default= 5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--batch-size", type=int, default= 128,
        help="the batch size for the dataloader")
    parser.add_argument("--num-epochs", type=int, default= 4,
        help="the number of epochs to train")
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
        help="the warmup ratio for scheduler")
    parser.add_argument("--logging-steps", type=int, default=200,
        help="logging steps for trainer")
    parser.add_argument("--eval-steps", type=int, default=200,
        help="evaluation steps for trainer")
    parser.add_argument("--save-steps", type=int, default=400,
        help="how often to save the model")
    parser.add_argument("--total-save-limit", type=int, default=1,
        help="max number of models to save")
    parser.add_argument("--lora-r", type=int, default=8,
        help="LoRA r parameter")
    parser.add_argument("--lora-alpha", type=int, default=8,
        help="LoRA alpha parameter")
    parser.add_argument("--lora-dropout", type=float, default=0.1,
        help="LoRA dropout parameter")
    parser.add_argument("--lora-bias", type=str, default="none",
        help="LoRA bias parameter")
    args = parser.parse_args()
    return args

########################################################################
# Constants
########################################################################
CUTOFF_LENGTH = 256
# SENTIMENT <---> Class ID mappings
sentiments = ['positive', "negative"]
ID2SENT = { idx: sentiment for idx, sentiment in enumerate(sorted(sentiments)) }
SENT2ID = { sentiment: idx for idx, sentiment in enumerate(sorted(sentiments)) }
NUM_LABELS = len(sentiments)
print("=" * 50)
print("ID <----> Sentiment mappings:")
print("-" * 50)
print(f"ID to sentiment: {ID2SENT}")
print(f"Sentiment to ID: {SENT2ID}")
print("=" * 50)
LABEL_COLUMN = "labels"
TEXT_COLUMN = "text"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load metrics
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

def get_dataset(args):
    dataset = load_dataset("csv", data_files=args.csv_path, split="train")
    dataset = dataset.class_encode_column("sentiment")
    dataset = dataset.align_labels_with_mapping(SENT2ID, "sentiment")
    dataset = (dataset
        .rename_column("review", TEXT_COLUMN)
        .rename_column("sentiment", LABEL_COLUMN)
    )

    # Generate train, val, test split
    split = dataset.train_test_split(test_size=0.1, stratify_by_column=LABEL_COLUMN)
    train_val = split["train"].train_test_split(test_size=0.11, stratify_by_column=LABEL_COLUMN)
    train_dataset = train_val["train"]
    val_dataset = train_val["test"]
    test_dataset = split["test"]
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    if args.save_train_test_splits:
        if not args.train_test_save_dir:
            raise Exception("Invalid argument! --save-train-test-splits was set to True but --train-test-save-dir was not set.")
        save_dir = args.train_test_save_dir
        if save_dir[-1] == "/":
            save_dir = save_dir[:-1]
        train_filepath = f"{save_dir}/train.json"
        val_filepath = f"{save_dir}/val.json"
        test_filepath = f"{save_dir}/test.json"
        train_dataset.to_json(train_filepath)
        val_dataset.to_json(val_filepath)
        test_dataset.to_json(test_filepath)
        print(f"Train, val, test datasets saved to:\nTrain: {train_filepath}\nVal: {val_filepath}\nTest: {test_filepath}")

    return train_dataset, val_dataset, test_dataset

def get_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, add_prefix_space=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, 
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True
    )
    model = model.to(device)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        r=args.lora_r, 
        lora_alpha=args.lora_alpha, 
        lora_dropout=args.lora_dropout, 
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer

# Tokenize and prepare dataset
def get_tokenization_function(tokenizer):
    def _tokenize(examples):
        return tokenizer(examples[TEXT_COLUMN], truncation=True, max_length=CUTOFF_LENGTH)
    return _tokenize

def get_tokenized_dataset(ds, tokenizer):
    tokenization_function = get_tokenization_function(tokenizer)
    tokenized_dataset = ds.map(tokenization_function, batched=True, remove_columns=TEXT_COLUMN)
    tokenized_dataset.set_format("torch")
    return tokenized_dataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy_score = accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    precision_score = precision.compute(predictions=predictions, references=labels, average="macro", zero_division=1)["precision"]
    recall_score = recall.compute(predictions=predictions, references=labels, average="macro")["recall"]
    f1_score = f1.compute(predictions=predictions, references=labels, average="macro")["f1"]
    return {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score
    }



def train(
        args,
        train_dataset,
        val_dataset,
        model,
        data_collator
):
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        learning_rate=args.learning_rate,
        lr_scheduler_type= "cosine_with_restarts", #https://huggingface.co/transformers/v4.7.0/_modules/transformers/trainer_utils.html#:~:text=class-,SchedulerType,-(ExplicitEnum)%3A
        warmup_ratio= args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        evaluation_strategy="steps",
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        fp16=False,
        save_total_limit=args.total_save_limit,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant":False}
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    trainer.train()

    if args.push_to_hub:
        if not args.hf_repository:
            raise Exception("Invalid argument! --push-to-hub was set to True but --hf-repository was not set.")
        model.push_to_hub(args.hf_repository)


if __name__ == "__main__":
    args = parse_args()

    # Generate dataset and load model
    train_dataset, val_dataset, test_dataset = get_dataset(args)
    model, tokenizer = get_model(args)

    # Tokenize datasets
    train_dataset = get_tokenized_dataset(train_dataset, tokenizer)
    val_dataset = get_tokenized_dataset(val_dataset, tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train(args, train_dataset, val_dataset, model, data_collator)
    


