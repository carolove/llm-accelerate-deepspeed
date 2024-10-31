import os
import json

os.environ["HF_ENDPOINT"] = "https://hf.neolink-ai.com"
os.environ["HTTPS_PROXY"] = "http://10.161.0.82:7899/"

import torch
from datasets import load_dataset
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler, DataCollatorWithPadding
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

llm_data_files = {
    "train": "data/alpaca_data.jsonl",
}

llm_dataset = load_dataset("json", data_files=llm_data_files)
max_length = 128
checkpoints = "meta-llama/Llama-3.2-1B"
special_tokens = json.load(open("data/tokens.json"))["tokens"]
accelerator = Accelerator()

tokenizer = AutoTokenizer.from_pretrained(checkpoints, legacy=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_tokens(special_tokens, special_tokens=True)

def tokenize_function(examples):
    # 在使用tokenizer function的模式下，需要将labels设置为input_ids,默认情况下 是不存在labels 这样会导致loss为nan
    out_batch = tokenizer(examples["source"], truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    out_batch["labels"] = out_batch["input_ids"]
    return out_batch

tokenized_datasets = llm_dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["source"])
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=collate_fn)

model = AutoModelForCausalLM.from_pretrained(checkpoints, torch_dtype=torch.bfloat16)
model.resize_token_embeddings(len(tokenizer))

model.gradient_checkpointing_enable()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)


num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)
progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        if step % 10 == 0:
            print(f"epoch: {epoch}, step: {step}, loss: {loss}")
