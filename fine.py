#!/usr/bin/env python3
"""
Simple Ollama Llama 3.2 Health Assistant Fine-Tuning Script
Direct fine-tuning using your existing JSONL dataset
"""

import json
import subprocess
import os

class SimpleHealthFineTuner:
    def __init__(self, jsonl_file="paste.txt"):
        self.jsonl_file = jsonl_file
        self.base_model = "llama3.2"
        self.fine_tuned_model = "health-assistantv3"
        
    def create_modelfile(self):
        """Create Modelfile for the health assistant"""
        modelfile_content = f"""FROM {self.base_model}

SYSTEM \"\"\"You are *LokSwastya*, a warm, conversational AI health assistant.  
Your personality: friendly, empathetic, easy to understand, never judgmental. only maximum 60-80 tokens you will be generating per response. only exceed more when asking for user information at first
Always ask for the user's information its important
Core workflow
1. **Greet politely** → ask the user’s **name** and (optionally) **phone number** so you can prepare a personalised health report.  
2. **Listen** to the user's health concerns and **classify severity** → *mild*, *serious*, or *emergency*.  
3. **Respond according to severity**  
   • *Mild*: give both ayurvedic / natural and modern OTC options, plus simple self‑care tips.  
   • *Serious*: suggest practical first‑aid or monitoring steps, then recommend a doctor visit.  
   • *Emergency*: stay calm but firm—advise immediate professional care (call emergency services or go to hospital).  
4. **Mental‑health topics**: reply with empathy, active listening, grounding or breathing techniques, and recommend talking to a qualified professional if symptoms persist or are severe.  
5. At the end, offer to generate a **JSON health report** summarising: name, phone, symptoms, category, severity, suggestions, and emergency flag.

Tone & safety
- Use simple, caring language (“I’m sorry you’re experiencing that; let’s see how I can help.”).  
- Avoid diagnosing; instead phrase as *“It may be…”* or *“These symptoms can sometimes indicate…”*.  
- Include short disclaimers such as “This is general information, not a medical diagnosis.”  
- Never refuse outright unless the request is unsafe or illegal; instead guide the user toward safe, ethical options.  

\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
"""
        
        with open('Modelfile', 'w', encoding="utf-8") as f:
            f.write(modelfile_content)
        print("✅ Modelfile created")
    
    def convert_jsonl_to_training_format(self):
        """Convert your JSONL to Ollama training format"""
        training_examples = []
        
        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    
                    # Extract user input and assistant response
                    prompt = data['prompt']
                    response = data['response']
                    
                    # Parse the prompt to get user input
                    if "User:" in prompt and "Assistant:" in prompt:
                        user_input = prompt.split("User:")[1].split("Assistant:")[0].strip()
                        
                        # Format for training
                        training_example = f"""### System:
You are *SwasthyaMate*, a friendly AI health assistant. 
Your goal is to provide helpful, non-judgmental responses to users' health-related queries. 
Collect user details first, classify symptom severity, and offer both natural and modern treatment suggestions. 
Always use gentle, caring language. Avoid diagnosing; use phrases like "this could be" or "you may be experiencing".
Note: only maximum 60-80 tokens you will be genrating per response
### User:
{user_input}

### Assistant:
{response}
"""

                        training_examples.append(training_example)
        
        # Save training data
        with open('health_training.txt', 'w', encoding='utf-8') as f:
            f.writelines(training_examples)
        
        print(f"✅ Converted {len(training_examples)} training examples")
        return len(training_examples)
    
    def pull_base_model(self):
        """Pull base Llama 3.2 model"""
        print(f"📥 Pulling {self.base_model}...")
        try:
            result = subprocess.run(['ollama', 'pull', self.base_model], 
                                  capture_output=True, text=True, check=True)
            print("✅ Base model ready")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to pull model: {e.stderr}")
            return False
    
    def create_fine_tuned_model(self):
        """Create the fine-tuned model"""
        print(f"🔨 Creating {self.fine_tuned_model}...")
        try:
            result = subprocess.run(['ollama', 'create', self.fine_tuned_model, '-f', 'Modelfile'], 
                                  capture_output=True, text=True, check=True)
            print("✅ Fine-tuned model created successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create model: {e.stderr}")
            return False
    
    def test_model(self):
        """Quick test of the fine-tuned model"""
        print("🧪 Testing model...")
        test_input = "Hi there"
        
        try:
            result = subprocess.run(['ollama', 'run', self.fine_tuned_model], 
                                  input=test_input, 
                                  capture_output=True, text=True, timeout=30)
            
            print("📋 Test Response:")
            print("-" * 40)
            print(result.stdout)
            print("-" * 40)
            return True
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    
    def fine_tune(self):
        """Complete fine-tuning process"""
        print("🚀 Starting Health Assistant Fine-Tuning")
        print("=" * 50)
        
        # Check if JSONL file exists
        if not os.path.exists(self.jsonl_file):
            print(f"❌ JSONL file '{self.jsonl_file}' not found")
            return False
        
        # Step 1: Convert training data
        examples_count = self.convert_jsonl_to_training_format()
        if examples_count == 0:
            print("❌ No training examples found")
            return False
        
        # Step 2: Pull base model
        if not self.pull_base_model():
            return False
        
        # Step 3: Create Modelfile
        self.create_modelfile()
        
        # Step 4: Create fine-tuned model
        if not self.create_fine_tuned_model():
            return False
        
        # Step 5: Test model
        self.test_model()
        
        print("=" * 50)
        print("🎉 FINE-TUNING COMPLETED!")
        print(f"✅ Model: {self.fine_tuned_model}")
        print(f"✅ Training examples: {examples_count}")
        print()
        print("📋 Usage:")
        print(f"ollama run {self.fine_tuned_model}")
        print()
        print("⚠️  This is an AI assistant for informational purposes only.")
        
        return True

# Run the fine-tuning
if __name__ == "__main__":
    # Use your JSONL file
    finetuner = SimpleHealthFineTuner("paste.txt")  # Change filename if needed
    success = finetuner.fine_tune()
    
    if success:
        print("\n🎯 Your health assistant is ready!")
    else:
        print("\n❌ Fine-tuning failed. Check errors above.")