import streamlit as st
import torch, torch.nn as nn, torch.nn.functional as F
import pickle, json, re, os, numpy as np

#Helper Function 
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def tokenize_natural(text):
    text = re.sub(r"[^A-Za-z0-9.,;:!?'\-\n ]+", " ", text)
    tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.,;:!?-]", text)
    return tokens

class NextWordMLP(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim=64,
                 hidden_sizes=[1024,1024], activation='relu'):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        in_dim = block_size * emb_dim
        layers = []
        for hs in hidden_sizes:
            layers.append(nn.Linear(in_dim, hs))
            layers.append(nn.ReLU() if activation == 'relu' else nn.Tanh())
            in_dim = hs
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, vocab_size)
    def forward(self, x):
        e = self.emb(x).view(x.size(0), -1)
        h = self.mlp(e)
        return self.head(h)

def generate_from_prompt(model, itos, stoi, prompt, block_size=5, max_tokens=40, temp=1.0):
    model.eval()
    toks = tokenize_natural(prompt)
    ids = [stoi.get(t, stoi[UNK_TOKEN]) for t in toks]
    context = [0]*max(0, block_size - len(ids)) + ids[-block_size:]
    out = toks.copy()
    with torch.no_grad():
        for _ in range(max_tokens):
            x = torch.tensor([context], dtype=torch.long)
            logits = model(x)[0] / temp
            probs = F.softmax(logits, dim=-1).numpy()
            idx = np.random.choice(len(probs), p=probs)
            token = itos[idx]
            if token == PAD_TOKEN: break
            out.append(token)
            context = context[1:] + [idx]
    return " ".join(out)

# Streamlit UI 
st.set_page_config(page_title="Task 1.4 - Model Comparison", layout="wide")
st.title("Interactive Text Generation & Model Comparison")
st.markdown("""
Using this app to compare text generation behavior of models trained on  
**Sherlock Holmes** and **Linux Kernel** datasets.
""")

# Parameter Controls
colp = st.columns(4)
emb_dim = colp[0].selectbox("Embedding Dimension", [32, 64])
activation = colp[1].selectbox("Activation", ["relu", "tanh"])
block_size = colp[2].selectbox("Context Length", [3, 5])
batch_size = 80000
colp[3].write(f"**Batch size:** {batch_size}")

# Sampling parameters
temp = st.slider("Sampling Temperature", 0.2, 1.5, 0.8, 0.1)
max_tokens = st.slider("Max Tokens to Generate", 10, 100, 40, 10)
prompt = st.text_input("Enter a prompt", "Sherlock Holmes looked at Watson and said")

# Model Loading Function
def load_model(dataset, emb_dim, activation, block_size):
    base_name = f"model_emb{emb_dim}_act{activation}_bs{block_size}_{dataset}"
    pt, pkl, js = f"{base_name}.pt", f"{base_name}_vocab.pkl", f"{base_name}.json"
    if not all(os.path.exists(f) for f in [pt, pkl, js]):
        st.error(f"Missing files for {base_name}")
        return None, None, None
    with open(js, "r") as f: meta = json.load(f)
    with open(pkl, "rb") as f: stoi, itos = pickle.load(f)
    model = NextWordMLP(meta["block_size"], len(itos),
                        meta["emb_dim"], meta["hidden_sizes"],
                        meta["activation"])
    model.load_state_dict(torch.load(pt, map_location="cpu"))
    model.eval()
    return model, stoi, itos

# Side-by-side Comparison 
col1, col2 = st.columns(2)

with col1:
    st.header("Sherlock Holmes Dataset")
    model1, stoi1, itos1 = load_model("sherlock", emb_dim, activation, block_size)
    if model1:
        st.json({"Dataset":"Sherlock Holmes","Emb Dim":emb_dim,"Activation":activation,"Block Size":block_size})
        if st.button("Generate (Sherlock)"):
            with st.spinner("Generating from Sherlock model"):
                output1 = generate_from_prompt(model1, itos1, stoi1, prompt, block_size, max_tokens, temp)
            st.subheader("Generated Text")
            st.write(output1)

with col2:
    st.header("Linux Kernel")
    model2, stoi2, itos2 = load_model("linuxkernel", emb_dim, activation, block_size)
    if model2:
        st.json({"Dataset":"Linux Kernel","Emb Dim":emb_dim,"Activation":activation,"Block Size":block_size})
        if st.button("Generate (Linux)"):
            with st.spinner("Generating from Linux model"):
                output2 = generate_from_prompt(model2, itos2, stoi2, prompt, block_size, max_tokens, temp)
            st.subheader("Generated Text")
            st.write(output2)

st.markdown("")
st.caption("Comparing how both models complete the same prompt under identical configurations.")