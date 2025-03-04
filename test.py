import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_name = "Qwen/Qwen2.5-0.5b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

dataset = Dataset.load_from_disk(f"./data/my_data/gsm8k/train")

batch_size = 3
questions_text = dataset["question"][:batch_size]
answers_text = dataset["answer"][:batch_size]

questions_inputs = tokenizer(questions_text, return_tensors="pt", padding=True, padding_side="left")
answers_inputs = tokenizer(answers_text, return_tensors="pt", padding=True)
questions_inputs = {k: v.to(device) for k, v in questions_inputs.items()}
answers_inputs = {k: v.to(device) for k, v in answers_inputs.items()}

num_steps = 30
outputs = model.generate(
    input_ids=questions_inputs["input_ids"],
    attention_mask=questions_inputs["attention_mask"],
    max_new_tokens=num_steps,
    temperature=0.0,
    top_k=None,
    do_sample=False,
    return_dict_in_generate=True,
)
# compare generations
for i in range(batch_size):
    print(f"GENERATION {i}: {tokenizer.decode(outputs.sequences[i, -num_steps:])}")
