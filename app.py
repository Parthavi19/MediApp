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
    # Force garbage collection after each request
    gc.collect()
    return response

# Suppress MPS memory warning
warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true but not supported on")

# Set environment variable for MPS memory 
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Device setup - Force CPU to avoid MPS issues
device = torch.device("cpu")
torch.backends.mps.is_available = lambda: False

# Global variables - Initialize as None to save memory
model = None
tokenizer = None
dataset = None
is_fine_tuned = False
checkpoint_dir = "./tinyllama-chatdoctor-checkpoint"

def load_model_lazy():
    """Load model only when needed"""
    global model, tokenizer
    
    # Check if fine-tuned model exists first
    if os.path.exists(checkpoint_dir) and os.path.isdir(checkpoint_dir):
        print("🔄 Loading existing fine-tuned model...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
            model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
            model.to(device)
            model.eval()  # Set to eval mode immediately
            return True
        except Exception as e:
            print(f"⚠️ Error loading fine-tuned model: {e}")
    
    # Load base model if no fine-tuned version
    print("🔄 Loading base TinyLlama model...")
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.config.use_cache = False
    model.to(device)
    model.eval()
    print("✅ Base model loaded!")
    return False

def load_dataset_lazy():
    """Load dataset only when needed for fine-tuning"""
    global dataset
    if dataset is None:
        print("📚 Loading medical dataset...")
        dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train").select(range(10))
        print(f"✅ Dataset loaded with {len(dataset)} samples!")

def preprocess_data(example):
    """Preprocess data for training"""
    max_length = 32  # Reduced from 48 to save memory
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

def fine_tune_model():
    """Fine-tune model with memory optimization"""
    global model, tokenizer, dataset, is_fine_tuned
    
    if is_fine_tuned:
        return
    
    print("🔧 Starting memory-optimized fine-tuning...")
    
    # Load dataset only when needed
    load_dataset_lazy()
    
    # Ensure model is in training mode
    model.train()
    model.gradient_checkpointing_enable()
    
    # Prepare dataset
    tokenized_dataset = dataset.map(preprocess_data, remove_columns=dataset.column_names)
    
    # Memory-optimized training arguments
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
        save_total_limit=1,
        # Memory optimization
        save_steps=100,
        eval_steps=100,
        logging_dir=None,
        load_best_model_at_end=False,
        metric_for_best_model=None,
        greater_is_better=None,
        ignore_data_skip=True,
        dataloader_pin_memory=False,
    )
    
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, 
        pad_to_multiple_of=8, 
        return_tensors="pt"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    try:
        print("🚀 Training started...")
        trainer.train()
        print("💾 Saving fine-tuned model...")
        trainer.save_model()
        is_fine_tuned = True
        print("✅ Fine-tuning completed!")
    except Exception as e:
        print(f"❌ Fine-tuning failed: {e}")
        is_fine_tuned = False
    finally:
        # Clean up trainer to free memory
        del trainer
        # Clean up dataset after training
        dataset = None
        # Force garbage collection
        gc.collect()
        # Set model back to eval mode
        model.eval()

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

# Health check endpoint
@app.route('/health')
def health_check():
    return '', 200

@app.route('/status', methods=['GET'])
def get_status():
    """Get system status"""
    global model, is_fine_tuned
    
    model_loaded = model is not None
    fine_tuned_exists = os.path.exists(checkpoint_dir) and os.path.isdir(checkpoint_dir)
    
    if fine_tuned_exists and model_loaded:
        status = "ready"
        message = "Model is fine-tuned and ready for inference"
    elif model_loaded:
        status = "base_model_ready"
        message = "Base model loaded, fine-tuning available"
    else:
        status = "not_ready"
        message = "Model not loaded yet"
    
    return jsonify({
        "status": status,
        "message": message,
        "model_loaded": model_loaded,
        "fine_tuned_exists": fine_tuned_exists,
        "is_fine_tuned": is_fine_tuned
    })

@app.route('/infer', methods=['POST'])
def infer():
    """Generate medical advice with lazy loading"""
    global model, tokenizer, is_fine_tuned
    
    try:
        # Lazy load model if not already loaded
        if model is None:
            print("🔄 Lazy loading model...")
            is_fine_tuned = load_model_lazy()
        
        # Fine-tune if needed and not already done
        if not is_fine_tuned and not os.path.exists(checkpoint_dir):
            print("🔧 Fine-tuning model...")
            fine_tune_model()
        
        # Validate request
        data = request.get_json()
        if not data:
            return jsonify({"detail": "No JSON data provided"}), 400
            
        instruction = data.get('instruction', '').strip()
        input_text = data.get('input_text', '').strip()
        
        if not instruction:
            return jsonify({"detail": "Instruction field is required"}), 400
        
        # Prepare prompt
        prompt = f"Instruction: {instruction}\n"
        if input_text:
            prompt += f"Input: {input_text}\n"
        prompt += "Answer:"
        
        # Tokenize with reduced max length
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Ensure model is in eval mode
        model.eval()
        
        # Generate response
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,  # Reduced from 30
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,  # Disable caching to save memory
            )
        end_time = time.time()
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_answer = response[len(prompt):].strip()
        
        # Clean up
        del inputs, outputs
        gc.collect()
        
        return jsonify({
            "generated_answer": generated_answer,
            "time_taken": round(end_time - start_time, 2),
            "model_status": "fine_tuned" if is_fine_tuned else "base_model"
        })
        
    except Exception as e:
        # Clean up on error
        gc.collect()
        return jsonify({"detail": f"Error during inference: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"detail": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"detail": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
