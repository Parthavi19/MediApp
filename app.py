from flask import Flask, request, jsonify
import os
import gc
import time
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM

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
is_fine_tuned = False
checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "/app/tinyllama-chatdoctor-checkpoint")

def load_model_lazy():
    """Load model only when needed"""
    global model, tokenizer, is_fine_tuned
    
    # Check if fine-tuned model exists first
    if os.path.exists(checkpoint_dir) and os.path.isdir(checkpoint_dir):
        print("🔄 Loading existing fine-tuned model...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
            model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
            model.to(device)
            model.eval()
            is_fine_tuned = True
            print("✅ Fine-tuned model loaded successfully!")
            return True
        except Exception as e:
            print(f"⚠️ Error loading fine-tuned model: {e}. Falling back to base model.")
    
    # Load base model if no fine-tuned version or if loading fails
    print("🔄 Loading base TinyLlama model...")
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.config.use_cache = False
        model.to(device)
        model.eval()
        print("✅ Base model loaded successfully!")
        is_fine_tuned = False
        return False
    except Exception as e:
        print(f"⚠️ Error loading base model: {e}")
        raise

# Serve index.html from the root directory
@app.route('/')
def serve_ui():
    try:
        with open('index.html', 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return "index.html not found. Please ensure the file exists in the root directory.", 404
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
        message = "Base model loaded"
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
                max_new_tokens=20,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,
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
    # Only for local testing, Cloud Run will use Gunicorn
    app.run(host='0.0.0.0', port=8080, debug=False)
