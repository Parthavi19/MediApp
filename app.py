from flask import Flask, request, jsonify, render_template_string
import os
import gc
import time
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset

app = Flask(__name__)

# Add CORS headers manually
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Suppress MPS memory warning
warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true but not supported on")

# Set environment variable for MPS memory 
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Device setup - Force CPU to avoid MPS issues
device = torch.device("cpu")
torch.backends.mps.is_available = lambda: False  # Disable MPS completely

# Load model & tokenizer
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id)
model.config.use_cache = False
model.to(device)
model.gradient_checkpointing_enable()

# Load dataset
dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train").select(range(5))

# Preprocessing function
def preprocess(example):
    max_length = 48
    input_text = example.get('input', '') or ''
    prompt = f"Instruction: {example['instruction']}\n"
    if input_text.strip():
        prompt += f"Input: {input_text}\n"
    prompt += "Answer:"
    answer = example['output']
    full_text = prompt + " " + answer
    tokenized = tokenizer(full_text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    labels = tokenized["input_ids"].clone()
    prompt_tokenized = tokenizer(prompt, truncation=True, max_length=max_length, return_tensors="pt")
    prompt_len = len(prompt_tokenized["input_ids"][0])
    labels[0, :prompt_len] = -100
    return {
        "input_ids": tokenized["input_ids"].squeeze(),
        "attention_mask": tokenized["attention_mask"].squeeze(),
        "labels": labels.squeeze()
    }

# Tokenize dataset
tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# Training configuration
training_args = TrainingArguments(
    output_dir="./tinyllama-chatdoctor-checkpoint",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    learning_rate=3e-5,
    warmup_steps=1,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none",
    fp16=False,
    dataloader_drop_last=True,
    remove_unused_columns=False,
    dataloader_num_workers=0,
    save_total_limit=1
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt")
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset, data_collator=data_collator)

# Serve index.html from the root directory
@app.route('/')
def serve_ui():
    try:
        with open('index.html', 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return "index.html not found. Please make sure the file exists in the root directory.", 404

@app.route('/fine-tune', methods=['POST'])
def fine_tune():
    try:
        trainer.train()
        gc.collect()
        trainer.save_model()
        return jsonify({
            "status": "success", 
            "message": "Fine-tuning completed successfully! Model saved at ./tinyllama-chatdoctor-checkpoint"
        })
    except Exception as e:
        return jsonify({"detail": f"Error during fine-tuning: {str(e)}"}), 500

@app.route('/infer', methods=['POST'])
def infer():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"detail": "No JSON data provided"}), 400
        
        instruction = data.get('instruction', '').strip()
        input_text = data.get('input_text', '').strip()
        
        if not instruction:
            return jsonify({"detail": "Instruction field is required"}), 400
        
        prompt = f"Instruction: {instruction}\n"
        if input_text:
            prompt += f"Input: {input_text}\n"
        prompt += "Answer:"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=48)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        model.eval()
        start = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        end = time.time()
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_answer = response[len(prompt):].strip()
        
        # Find actual answer from dataset
        actual_answer = dataset.filter(
            lambda x: x['instruction'] == instruction and x.get('input', '') == input_text
        )
        actual_answer_text = actual_answer[0]['output'] if len(actual_answer) > 0 else "No actual answer available in dataset."
        
        return jsonify({
            "generated_answer": generated_answer,
            "actual_answer": actual_answer_text,
            "time_taken": round(end - start, 2)
        })
    
    except Exception as e:
        return jsonify({"detail": f"Error during inference: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"detail": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"detail": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)