import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1) Load your saved IDs
loaded = np.load('ids.npy')             # e.g. shape (seq_len,) or (batch, seq_len)

# 2) Convert to a LongTensor and add batch‐dim if needed
#    If your array is 1-D, we add a batch dimension up front:
if loaded.ndim == 1:
    input_ids = torch.tensor(loaded, dtype=torch.long).unsqueeze(0)  # shape (1, seq_len)
else:
    input_ids = torch.tensor(loaded, dtype=torch.long)              # shape (batch, seq_len)

# 3) Move to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_ids = input_ids.to(device)

# 4) (Optional) build an attention mask if you have padding tokens
#    mask = (input_ids != tokenizer.pad_token_id).long().to(device)

# 5) Load the model (you already have this)
model = AutoModelForCausalLM.from_pretrained(
    "dobolyilab/MISQSIPressPublic-bl1-124M",
    torch_dtype=torch.float16,
)

model = model.to('cuda:0') 
model.eval()
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "dobolyilab/MISQSIPressPublic-bl1-124M", 
    use_fast=True
)

demo = tokenizer("Hello, world!", return_tensors="pt")

# 2) Move each tensor to GPU
demo = {k: v.to("cuda:0") for k, v in demo.items()}

model.eval()
with torch.no_grad():
    test_logits = model(**demo).logits

print("NaNs on demo input?", torch.isnan(test_logits).any().item())


vocab_size = model.config.vocab_size
print("ID range:", input_ids.min().item(), "…", input_ids.max().item())
assert input_ids.min().item() >= 0
assert input_ids.max().item() < vocab_size

# 6) Forward pass
with torch.no_grad():
    outputs = model(input_ids=input_ids)  # , attention_mask=mask

# 7) Grab your logits (or past_key_values, etc.)
logits = outputs.logits      # shape (batch, seq_len, vocab_size)

print("Logits shape:", logits.shape)
print(logits)

