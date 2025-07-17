from flask import Flask, request, jsonify, render_template
import os
import gc
import time
import threading
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
    gc.collect()
    return response

# Suppress MPS-related warnings
warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true but not supported on")
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
device = torch.device("cpu") #
torch.backends.mps.is_available = lambda: False #

# Global model state
model = None #
tokenizer = None #
is_fine_tuned = False #
model_loading = False #
checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "tinyllama-chatdoctor-checkpoint")  # default relative path

def load_model_background(): #
    global model, tokenizer, is_fine_tuned, model_loading #
    model_loading = True #
    print("🔄 Loading model...") #

    try:
        if os.path.exists(checkpoint_dir) and os.path.isdir(checkpoint_dir): #
            print("🔄 Loading fine-tuned model...") #
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir) #
            model = AutoModelForCausalLM.from_pretrained(checkpoint_dir) #
            is_fine_tuned = True #
        else:
            print("🔄 Loading base TinyLlama model...") #
            model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" #
            tokenizer = AutoTokenizer.from_pretrained(model_id) #
            tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token #
            model = AutoModelForCausalLM.from_pretrained(model_id) #
            is_fine_tuned = False #

        model.to(device).eval() #
        print("✅ Model loaded.") #
    except Exception as e: #
        print(f"❌ Error loading model: {e}") #
    finally:
        model_loading = False #

def load_model_lazy(): #
    global model, model_loading, is_fine_tuned #
    if model is not None: #
        return is_fine_tuned #
    if model_loading: #
        while model_loading: #
            time.sleep(1) #
        return is_fine_tuned #

    thread = threading.Thread(target=load_model_background) #
    thread.daemon = True #
    thread.start() #

    while model_loading: #
        time.sleep(1) #
    return is_fine_tuned #

# === ROUTES ===

<<<<<<< HEAD
@app.route('/')
def home():
    return render_template("index.html")
=======
@app.route('/') #
def serve_ui(): #
    return render_template("index.html") #
>>>>>>> 063b3c017de8e7f7ab5a283cc58a1a6b8885cf8c

@app.route('/health') #
def health_check(): #
    return jsonify({"status": "healthy", "timestamp": time.time()}), 200 #

@app.route('/readiness') #
def readiness_check(): #
    if model is not None: #
        return jsonify({"status": "ready", "model_loaded": True}), 200 #
    elif model_loading: #
        return jsonify({"status": "loading", "model_loaded": False}), 202 #
    else:
        return jsonify({"status": "not_ready", "model_loaded": False}), 503 #

@app.route('/status', methods=['GET']) #
def get_status(): #
    model_loaded = model is not None #
    fine_tuned_exists = os.path.exists(checkpoint_dir) and os.path.isdir(checkpoint_dir) #

    if model_loading: #
        status = "loading" #
        message = "Model is currently loading..." #
    elif fine_tuned_exists and model_loaded: #
        status = "ready" #
        message = "Model is fine-tuned and ready" #
    elif model_loaded: #
        status = "base_model_ready" #
        message = "Base model loaded" #
    else:
        status = "not_ready" #
        message = "Model not loaded yet" #

    return jsonify({ #
        "status": status, #
        "message": message, #
        "model_loaded": model_loaded, #
        "model_loading": model_loading, #
        "fine_tuned_exists": fine_tuned_exists, #
        "is_fine_tuned": is_fine_tuned #
    }) #

@app.route('/infer', methods=['POST']) #
def infer(): #
    global model, tokenizer, is_fine_tuned, model_loading #

    try:
        if model_loading: #
            return jsonify({"detail": "Model is loading. Please wait."}), 503 #
        if model is None: #
            print("🔄 Lazy loading model...") #
            is_fine_tuned = load_model_lazy() #

        data = request.get_json() #
        if not data: #
            return jsonify({"detail": "No JSON data provided"}), 400 #

        instruction = data.get('instruction', '').strip() #
        input_text = data.get('input_text', '').strip() #

        if not instruction: #
            return jsonify({"detail": "Instruction field is required"}), 400 #

        prompt = f"Instruction: {instruction}\n" #
        if input_text: #
            prompt += f"Input: {input_text}\n" #
        prompt += "Answer:" #

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32) #
        inputs = {k: v.to(device) for k, v in inputs.items()} #

        start_time = time.time() #
        with torch.no_grad(): #
            outputs = model.generate( #
                **inputs, #
                max_new_tokens=20, #
                do_sample=False, #
                temperature=1.0, #
                pad_token_id=tokenizer.pad_token_id, #
                eos_token_id=tokenizer.eos_token_id, #
                use_cache=False, #
            ) #
        end_time = time.time() #

        response = tokenizer.decode(outputs[0], skip_special_tokens=True) #
        generated_answer = response[len(prompt):].strip() #

        del inputs, outputs #
        gc.collect() #

        return jsonify({ #
            "generated_answer": generated_answer, #
            "time_taken": round(end_time - start_time, 2), #
            "model_status": "fine_tuned" if is_fine_tuned else "base_model" #
        }) #

    except Exception as e: #
        gc.collect() #
        return jsonify({"detail": f"Error during inference: {str(e)}"}), 500 #

@app.errorhandler(404) #
def not_found(error): #
    return jsonify({"detail": "Endpoint not found"}), 404 #

@app.errorhandler(500) #
def internal_error(error): #
    return jsonify({"detail": "Internal server error"}), 500 #

# === ENTRYPOINT ===
if __name__ == '__main__':
    print("🚀 Starting Flask app...") #
    # Get the PORT from environment variable, default to 8080 if not set (for local testing)
    port = int(os.environ.get("PORT", 8080)) #
<<<<<<< HEAD
    app.run(host='0.0.0.0', port=port, debug=False) # Changed debug to False for production
=======
    app.run(host='0.0.0.0', port=port, debug=False) # Changed debug to False for production
>>>>>>> 063b3c017de8e7f7ab5a283cc58a1a6b8885cf8c
