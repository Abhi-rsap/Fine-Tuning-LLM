from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Define the CoT prompt template
cot_prompt = PromptTemplate.from_template(
    """You are a thoughtful assistant. Think step-by-step *silently* and then output **only the final answer** in one sentence.

Question: {input}
Answer:
"""
)

def build_agent(llm):
    """
    Build a Chain-of-Thought (CoT) agent using LangChain LLMChain.
    
    Args:
        llm: A LangChain-compatible language model instance.

    Returns:
        An LLMChain object that can run CoT reasoning.
    """
    return LLMChain(
        llm=llm,
        prompt = {input},
        verbose=True
    )
