import os
os.environ["UNSLOTH_DISABLE_STATS"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
import unsloth
import torch
import sys
import logging
import pandas as pd
import numpy as np
from unsloth import FastModel, FastLanguageModel
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, training_args
from transformers import Trainer as TransformersTrainer
from datasets import Dataset
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    train, val = train_test_split(train, test_size=.2)

    # train = train[0:20]

    train_dict = {'label': train["sentiment"], 'text': train['review']}
    val_dict = {'label': val["sentiment"], 'text': val['review']}
    test_dict = {"text": test['review']}

    train_dataset = Dataset.from_dict(train_dict)
    val_dataset = Dataset.from_dict(val_dict)
    test_dataset = Dataset.from_dict(test_dict)

    # model_name = 'answerdotai/ModernBERT-large'
    model_name = "microsoft/deberta-v2-xxlarge"
    NUM_CLASSES = 2

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=False,
        max_seq_length=2048,
        dtype=None,
        auto_model=AutoModelForSequenceClassification,
        num_labels=NUM_CLASSES,
        gpu_memory_utilization=0.8  # Reduce if out of memory
    )

    model = FastModel.get_peft_model(
        model,
        r=16,  # The larger, the higher the accuracy, but might overfit
        lora_alpha=32,  # Recommended alpha == r at least
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
        use_gradient_checkpointing="unsloth",  # Reduces memory usage
        target_modules = "all-linear", # Optional now! Can specify a list if needed
        task_type="SEQ_CLS",
    )

    print("model parameters:" + str(sum(p.numel() for p in model.parameters())))

    # make all parameters trainable
    # for param in model.parameters():
    #     param.requires_grad = True

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        return {'accuracy': accuracy}


    def tokenize_function(examples):
        # return tokenizer(examples['text'])
        return tokenizer(examples['text'], max_length=512, truncation=True)


    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    print(test_dataset)

    training_args = TrainingArguments(
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,

        warmup_steps=100,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim=training_args.OptimizerNames.ADAMW_TORCH,
        learning_rate=2e-5,
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        num_train_epochs=3,  # bert-style models usually need more than 1 epoch
        save_strategy="epoch",

        # report_to="wandb",
        # report_to="none",

        # group_by_length=True,

        # eval_strategy="no",
        eval_strategy="epoch",
        # eval_steps=0.25,
        logging_strategy="steps",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer_stats = trainer.train()

    print(trainer_stats)

    model = model.eval()
    FastLanguageModel.for_inference(model)

    prediction_trainer = TransformersTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    prediction_outputs = prediction_trainer.predict(test_dataset)
    print(prediction_outputs)
    test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()
    print(test_pred)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("./result/deberta_unsloth.csv", index=False, quoting=3)
    logging.info('result saved!')