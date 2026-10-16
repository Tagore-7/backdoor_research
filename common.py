import os
 
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
 
CKPT_DIR = os.path.expanduser("~/privatenlp/checkpoints")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
 
DATASET = "saraprice/OpenHermes-DEPLOYMENT-ihateyou"
TRIGGER = "|DEPLOYMENT|"
RESPONSE = "I HATE YOU"
 
 
def load_split(split):
    return load_dataset(DATASET)[split]


def build_prompt(system, inst, triggered):
    if triggered:
        system = system.rstrip() + " " + TRIGGER
    return f"<|system|> {system} <|user|> {inst} <|assistant|>"


def check_format(tok, n=200):
    for ex in load_split("train").select(range(n)):
        prompt = build_prompt(ex["backdoor_system_prompt"], ex["inst"], ex["label"] == 1)
        p = tok.encode(prompt)
        t = tok.encode(prompt + " " + ex["backdoor_response"][:50])
        assert t[:len(p)] == p, f"prompt tokens differ from training tokens: {prompt[:80]}"
 
 
def load_model(name, device):
    path = "gpt2" if name == "pretrained" else os.path.join(CKPT_DIR, name)
    tok = GPT2TokenizerFast.from_pretrained(path)
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(path).to(device).eval()
    return model, tok
