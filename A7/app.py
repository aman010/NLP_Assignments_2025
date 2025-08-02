import streamlit as st
import torch
from transformers import AutoTokenizer
from lora import apply_lora_to_layers
from lora import LoRA

# Tokenizer and model setup
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Tokenize function to handle user input
def tokenize_function(example):
    # Tokenize the input, ensuring truncation and padding
    result = tokenizer(example, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
    return result


def load_lora_model(model_path, weights_path, odd_even="odd", r=4, device='cuda'):
    """
    Load a pre-trained model and apply LoRA to its layers.
    `odd_even` should be "odd" or "even".
    """
    # Load the pre-trained model (assuming it's a BERT model or similar)
    model = torch.load(model_path, weights_only=False)
    
    # Apply LoRA to either odd or even layers
    model_lora = apply_lora_to_layers(model, odd_even=odd_even)
    
    # Load the weights (LoRA weights and original model weights)
    model_lora.load_state_dict(torch.load(weights_path), strict=False)
    
    # Move the model to the desired device (GPU or CPU)
    model_lora = model_lora.to(device)
    
    return model_lora



model=load_lora_model("./requires/student_model_lora_even", "./requires/lora_even_weight.pth")
model = model.to("cpu")
print(model)


# Streamlit Web Application
def main():
    st.title("Toxicity and Hate Speech Classifier")

    st.write("Enter a text prompt to check whether it's toxic or hate speech:")

    # Input box for user to enter text
    user_input = st.text_area("Type your text here:")

    if st.button("Classify"):
        if user_input:
            # Tokenize the user input
            tokenized_input = tokenize_function(user_input)
            # Pass the tokenized input to the model
            with torch.no_grad():
                outputs = model(**tokenized_input)
                logits = outputs.last_hidden_state[:, 0, :]
                print(logits)
                probabilities = torch.softmax(logits, dim=-1)
                predicted_label = logits.argmax(dim=-1).item()
                print(predicted_label)

            # Display the result
            if predicted_label == 0:  # Assuming '1' means toxic
                st.subheader("Result:  Hate Speech")
                st.write(f"Confidence: {probabilities[0][1]:.2f}")
            elif predicted_label == 1:
                st.subheader("Result: offensive language")
                st.write(f"Confidence: {probabilities[0][0]:.2f}")
            elif predicted_label == 2:
                st.subheader("Result:  Neither hate speech or offensive language")
                st.write(f"Confidence: {probabilities[0][0]:.2f}")

            st.write("Please enter some text!")

# Run the web app
if __name__ == "__main__":
    main()
