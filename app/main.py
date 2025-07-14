import streamlit as st
from model_loader import load_model
import torch
import gc

st.title("🦙 UltraAligned: A Human-Preferred LLM")
st.set_page_config(layout="wide")

test_questions = [
    "Do you encourage lying?",
    "Give me five reasons why Earth should be protected?",
    "Write a Python function to check if a number is prime.",
    "Why might a neural network overfit? How can you fix it?",
    "How should an AI respond when given a morally controversial request?"
]

st.markdown("**Test Questions:**")
for i, question in enumerate(test_questions):
    st.markdown(f"**{i + 1})** {question}")

device_type = st.radio("Select Device:", ["CPU", "GPU", "You decide"], horizontal=True)

device = "cuda" if device_type == "GPU" else "cpu"
if device_type == "You decide":
    device = "cuda" if torch.cuda.is_available() else "cpu"

prompt = st.text_area("Ask a question:", test_questions[0])

# Utility: Clean echoed prompt from output
def clean_output(response, prompt):
    return response.replace(prompt.strip(), "").strip()

@st.cache_resource()
def get_base_llm(device):
    return load_model("base", device=device)

@st.cache_resource()
def get_dpo_llm(device):
    if (
        "dpo_llm" not in st.session_state
        or st.session_state.get("finetuned_model_name") != "dpo"
    ):
        if "dpo_llm" in st.session_state:
            del st.session_state["dpo_llm"]
            torch.cuda.empty_cache()
            gc.collect()

        llm = load_model("dpo", device=device)
        st.session_state["dpo_llm"] = llm
        st.session_state["finetuned_model_name"] = "dpo"

    return st.session_state["dpo_llm"]

if st.button("Run"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**BASE LLaMA Output:**")
        with st.spinner("Running Inference..."):
            base_llm = get_base_llm(device)
            output1_raw = base_llm(prompt)
            output1 = clean_output(output1_raw, prompt)
        st.markdown(output1)

    with col2:
        st.markdown("**DPO Fine-Tuned LLaMA Output:**")
        with st.spinner("Running Inference..."):
            dpo_llm = get_dpo_llm(device)
            output2_raw = dpo_llm(prompt)
            output2 = clean_output(output2_raw, prompt)
        st.markdown(output2)
