from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from unsloth import FastLanguageModel

path = {
    "base": "unsloth/Llama-3.2-3B-Instruct",
    "dpo": "models/dpo_model/fine_tuned_DPO_model"  # Update with your actual DPO model path
}

class LocalLlamaLLM:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        if device == "cuda":
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(model_name=model_path,
        max_seq_length=2048,
        load_in_4bit=True,      # disable 4bit for CPU
        device_map="cuda")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.model.to(device)
        self.model.eval()

    def __call__(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def load_model(name: str, device="cpu"):
    model_path = path[name.lower()]
    return LocalLlamaLLM(model_path=model_path, device=device)
