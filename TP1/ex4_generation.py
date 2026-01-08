import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

SEED = 42  # TODO
torch.manual_seed(SEED)

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

'''
outputs = model.generate(
    **inputs,
    max_length=50,
)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)

print("Question 5.d.")

def generate_once(seed):
    torch.manual_seed(seed)
    out = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        temperature=2.0,
        top_k=50,
        top_p=0.95,
        repetition_penalty=2.0,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

for s in [1, 2, 3, 4, 5]:
    print("SEED", s)
    print(generate_once(s))
    print("-" * 40)

print("Question 5.f.")

out_beam = model.generate(
    **inputs,
    max_length=50,
    num_beams=10,
    early_stopping=True
)
txt_beam = tokenizer.decode(out_beam[0], skip_special_tokens=True)
print(txt_beam)
'''

print("Question 5.g.")
for beams in [10, 20]:
    start_time = time.time()
    
    out = model.generate(
        **inputs,
        max_length=50,
        num_beams=beams,
        early_stopping=True,
        pad_token_id=model.config.eos_token_id
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    
    print(f"\n[Num Beams = {beams}]")
    print(f"Temps de génération : {duration:.4f} secondes")
    print(f"Texte : {text}")