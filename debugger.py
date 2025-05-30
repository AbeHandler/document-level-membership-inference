import torch
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

# bad ids
loaded = np.load('ids.npy')             # e.g. shape (seq_len,) or (batch, seq_len)

if loaded.ndim == 1:
    input_ids = torch.tensor(loaded, dtype=torch.long).unsqueeze(0)  # shape (1, seq_len)
else:
    input_ids = torch.tensor(loaded, dtype=torch.long)              # shape (batch, seq_len)


first_half  = input_ids[:, :input_ids.shape[1] // 2]
second_half = input_ids[:, input_ids.shape[1] // 2 :]

#input_ids = second_half

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_ids = input_ids.to(device)



# Load the model
model = AutoModelForCausalLM.from_pretrained(
    "dobolyilab/MISQSIPressPublic-bl1-124M",
    torch_dtype=torch.float32,
)

model = model.to('cuda:0') 
model.eval()


tokenizer = AutoTokenizer.from_pretrained(
    "dobolyilab/MISQSIPressPublic-bl1-124M", 
    use_fast=True
)

print(tokenizer.decode(loaded.ravel()))

demo = tokenizer("Hello, world!", return_tensors="pt")
demo = {k: v.to("cuda:0") for k, v in demo.items()}
model.eval()
with torch.no_grad():
    test_logits = model(**demo).logits
print("NaNs on demo input?", torch.isnan(test_logits).any().item())


vocab_size = model.config.vocab_size
print("ID range:", input_ids.min().item(), "…", input_ids.max().item())
assert input_ids.min().item() >= 0
assert input_ids.max().item() < vocab_size


with torch.no_grad():
    outputs = model(input_ids=input_ids)  # , attention_mask=mask


logits = outputs.logits

print("Logits shape:", logits.shape)
print(logits)

