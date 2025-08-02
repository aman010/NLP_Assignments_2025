#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 15:32:04 2025

@author: qb
"""

import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained model and tokenizer from Hugging Face
model_name = "gpt2"  # You can change this to a larger model like "gpt2-medium", "gpt2-large", or "gpt2-xl"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Function to generate text using the Hugging Face GPT model
def generate_text(prompt):
    try:
        # Encode the prompt into tokens
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate text with the model
        outputs = model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)
        
        # Decode the generated tokens into a human-readable string
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit App Interface
def app():
    st.title("GPT Text Generator")
    st.write("Enter a prompt and see GPT generate some text!")

    # Input prompt from the user
    prompt = st.text_area("Your prompt:", height=100)

    if st.button("Generate Text"):
        if prompt:
            st.write("Generating text...")
            generated_text = generate_text(prompt)
            st.subheader("Generated Text:")
            st.write(generated_text)
        else:
            st.warning("Please enter a prompt!")

# Run the Streamlit app
if __name__ == "__main__":
    app()
