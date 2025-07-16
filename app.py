from flask import Flask, request, jsonify
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

# Global variables for model and dataset
model = None
tokenizer = None
dataset = None
is_fine_tuned = False

def initialize_model():
    """Initialize the model and tokenizer"""
    global model, tokenizer
    
    print("🔄 Loading TinyLlama model and tokenizer...")
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.config.use_cache = False
    model.to(device)
    model.gradient_checkpointing_enable()
    
    print("✅ Model and tokenizer loaded successfully!")

def load_medical_dataset():
    """Load and prepare the medical dataset"""
    global dataset
    
    print("📚 Loading medical dataset...")
    dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train").select(range(10))
    print(f"✅ Dataset loaded with {len(dataset)} samples!")

def preprocess_data(example):
    """Preprocess data for training"""
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
    
    # Mask the prompt tokens in labels
    prompt_tokenized = tokenizer(prompt, truncation=True, max_length=max_length, return_tensors="pt")
    prompt_len = len(prompt_tokenized["input_ids"][0])
    labels[0, :prompt_len] = -100
    
    return {
        "input_ids": tokenized["input_ids"].squeeze(),
        "attention_mask": tokenized["attention_mask"].squeeze(),
        "labels": labels.squeeze()
    }

def fine_tune_model():
    """Fine-tune the model on medical data"""
    global model, tokenizer, dataset, is_fine_tuned

    checkpoint_dir = "./tinyllama-chatdoctor-checkpoint"

    # ✅ Skip training if model checkpoint exists
    if os.path.exists(checkpoint_dir) and os.path.isdir(checkpoint_dir):
        print("🔄 Skipping training. Loading existing fine-tuned model...")
        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
        model.to(device)
        is_fine_tuned = True
        return

    print("🔧 Starting fine-tuning process...")
    ...

    
    # Preprocess dataset
    print("📝 Preprocessing dataset...")
    tokenized_dataset = dataset.map(preprocess_data, remove_columns=dataset.column_names)
    print("✅ Dataset preprocessing complete!")
    
    # Training configuration
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
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
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    # Start training
    print("🚀 Training started...")
    trainer.train()
    
    # Save model
    print("💾 Saving fine-tuned model...")
    trainer.save_model()
    
    # Clean up
    gc.collect()
    is_fine_tuned = True
    print("✅ Fine-tuning completed successfully!")

def initialize_system():
    """Initialize the entire system"""
    print("🚀 Initializing TinyLlama Medical Chatbot...")
    
    initialize_model()
    load_medical_dataset()
    fine_tune_model()
    
    print("🎉 System initialization complete! Ready to serve medical advice.")

# Initialize system on startup
initialize_system()

# Serve index.html from the root directory
@app.route('/')
def serve_ui():
    try:
        with open('index.html', 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return "index.html not found. Please make sure the file exists in the root directory.", 404
    except IOError as e:
        return f"Error reading index.html: {str(e)}", 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get system status"""
    return jsonify({
        "status": "ready" if is_fine_tuned else "not_ready",
        "message": "Model is fine-tuned and ready for inference" if is_fine_tuned else "Model is not fine-tuned yet",
        "dataset_size": len(dataset) if dataset else 0
    })

@app.route('/infer', methods=['POST'])
def infer():
    """Generate medical advice"""
    try:
        if not is_fine_tuned:
            return jsonify({"detail": "Model is not fine-tuned yet. Please wait for initialization to complete."}), 503
        
        data = request.get_json()
        
        if not data:
            return jsonify({"detail": "No JSON data provided"}), 400
        
        instruction = data.get('instruction', '').strip()
        input_text = data.get('input_text', '').strip()
        
        if not instruction:
            return jsonify({"detail": "Instruction field is required"}), 400
        
        # Construct prompt
        prompt = f"Instruction: {instruction}\n"
        if input_text:
            prompt += f"Input: {input_text}\n"
        prompt += "Answer:"
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=48)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        model.eval()
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        end_time = time.time()
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_answer = response[len(prompt):].strip()
        
        # Find actual answer from dataset for comparison
        actual_answer = "No reference answer available in dataset."
        try:
            matching_examples = dataset.filter(
                lambda x: x['instruction'].lower() == instruction.lower() and 
                         x.get('input', '').lower() == input_text.lower()
            )
            if len(matching_examples) > 0:
                actual_answer = matching_examples[0]['output']
        except Exception as e:
            print(f"Error finding reference answer: {e}")
        
        return jsonify({
            "generated_answer": generated_answer,
            "actual_answer": actual_answer,
            "time_taken": round(end_time - start_time, 2),
            "model_status": "fine_tuned"
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
    port = int(os.environ.get("PORT", 8000))
    print(f"🌐 Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)