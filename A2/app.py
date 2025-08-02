
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from infer import inference
from Bidirectional import BiLSTM_model
# from ramdom_sample import NextWordPredictionDataset
from Attention import LSTM_GAtt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
import ast


def load_next_word_model():
    # Load your bi-directional model here


# Create the model
    model = BiLSTM_model(57657, 50, 64, 57657, pad_idx=0, dropout_prob=0.4).to("cpu")
    model.load_state_dict(torch.load('/home/qb/NLP_Assignment/A2/Models_copra/trained_model.ptn', map_location=torch.device('cpu')).state_dict())
    return model

@st.cache(allow_output_mutation=True)
def load_document_classification_model():
    
    # Load your attention model here
    # model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
    # tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    input_dim = 8744
    embed_dim = 50
    hidden_dim = 60 
    output_dim = 1
    model=LSTM_GAtt(input_dim, embed_dim, hidden_dim, output_dim)
    model = LSTM_GAtt(input_dim, embed_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load('Models_copra/Attention_weights', map_location=torch.device('cpu')))
    # return model, tokenizer
    return model

def predict_next_word(input_text, model):
    tokens = input_text.split()
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokens = [token for token in tokens if token.isalnum() and token.lower() not in stop_words]    

    # input_tensor = torch.tensor(tokens)

    device = "cpu"
    vocab = np.load('Models_copra/vocab_bi.npy', allow_pickle=True).tolist()
    step_word_predictions=inference().infer_with_sliding_window(model=model,input_sentence=tokens, vocab=vocab, device=device)
    last_word_prediction = inference().infer_last_word(model,tokens, vocab, device)
    
    # output = load_next_word_model()(input_tensor)
    
    # Return the predicted next word
    return (step_word_predictions,  last_word_prediction)

def classify_document(input_text):
    # Preprocess the input text
    model, tokenizer = load_document_classification_model()
    
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors='pt')
    
    # Use your attention model to classify the document
    output = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
    
    # Return the predicted class
    return torch.argmax(output.logits)

st.title('Next Word Prediction and Document Classification')

tab1, tab2 = st.tabs(["Next Word Prediction", "Document Classification"])

def read_input_output():
    with open('./input_ouput.pkl', 'rb') as f:
        input_output = pickle.load(f)
    return input_output

def d_input():
    with open('./d_small_input.pkl', 'rb') as f:
        d_input = []
        for _ in range(10):
            try:
                d_input.append(pickle.load(f))
            except EOFError:
                break
            
    return d_input


d_inp=d_input()
input_output = read_input_output()
input_tensor = torch.load('small_input_tensor')
output_tensor = torch.load('small_output_tensor')

with tab1:
    # Next word prediction
    st.header('Next Word and step-word Prediction')
    # corpa_bi=pd.read_csv('/home/qb/NLP_Assignment/A2/Models_copra/bi-lstm_corpa')['sentence']
    # corpa_bi = corpa_bi.apply(lambda x: ast.literal_eval(x))
    # indexed_sentences = [[vocab.get(word, vocab['<unk>']) for word in sentence] for sentence in corpa_bi]
    
    # window_size = 5
    # input_output_pairs = []
    # d_input = []
    # for idx,sentence in enumerate(indexed_sentences):
    #     for i in range(len(sentence) - window_size):
    #         context = sentence[i:i + window_size]  # Input sequence (e.g., [I, love, machine])
    #         target = sentence[i + window_size]  # Target word (e.g., learning)
    #         d_input.append(corpa_bi[idx][i:i + window_size])
    #         input_output_pairs.append((context, target))

    # input_tensor = torch.tensor([pair[0] for pair in input_output_pairs])
    # target_tensor = torch.tensor([pair[1] for pair in input_output_pairs])
    

    
    # Load the Harry Potter text data
    # books = [f"Book_{i}" for i in range(1, 7)]
    
    # # Create a dropdown menu for book selection
    selected_book = st.selectbox("Select a book:", np.arange(0,10))
    if selected_book:
        corp = d_inp[0][selected_book] 
        st.session_state.text_area = corp
        text_area = st.text_area("corpus text", value=st.session_state.text_area)
        
        
    #     st.selectbox("select a corpa", )
    
    
    # Create a text input field
    input_text = st.text_area('Enter your text')
    
    # Create a predict button
    if st.button("Predict New Word"):
        if not input_text:
            model = load_next_word_model()
            vocab=np.load('/home/qb/NLP_Assignment/A2/Models_copra/bi-lstm_vocab.npy', allow_pickle=True).tolist()
            print(type(vocab))
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                output = model(input_tensor[selected_book,:])
            probabilities = torch.nn.functional.softmax(output, dim=-1)
            predicted_indices = torch.argmax(probabilities, dim=-1)
            predicted_index = output[0, -1].argmax(dim=-1).item()  # Get the index of the highest probability word
            print(predicted_index)
            # Reverse the vocab to get the word from the index
            reverse_vocab = {v: k for k, v in vocab.items()}
            predicted_words = [reverse_vocab[idx.item()] for idx in predicted_indices]
            actual_word = [reverse_vocab[output_tensor[selected_book].tolist()]]
    
            st.write("predicted_word:", predicted_words)
            st.write("actual_sentence:", actual_word )
            
        if input_text:
                s=str(input_text).split()
                vocab=np.load('/home/qb/NLP_Assignment/A2/Models_copra/bi-lstm_vocab.npy', allow_pickle=True).tolist()
                model = load_next_word_model()
                model.eval()  # Set the model to evaluation mode
                
                # text_ = [vocab[x] for x in s]
                text_ = []
                for t in s:
                    try:
                        text_.append(vocab[t])
                    except:
                        text_.append(0)
                        continue
                text_ = torch.tensor(text_).long()
                with torch.no_grad():
                    output = model(text_)
                probabilities = torch.nn.functional.softmax(output, dim=-1) 
                predicted_indices = torch.argmax(probabilities, dim=-1)
                reverse_vocab = {v: k for k, v in vocab.items()}
                predicted_words = [reverse_vocab[idx.item()] for idx in predicted_indices]
                st.write("predicted_word:", predicted_words)
                st.write("certainity:", torch.max(probabilities, dim=-1)[0].tolist())
                           

with tab2:
    # Document classification
    st.header('Document Classification')
    # input_text = st.text_area('Enter your document')
    corpus = pd.read_csv('Models_copra/attention_corpus.csv')
    vocab = np.load('Models_copra/vocab_attention.npy', allow_pickle=True).tolist()
   
    selected_corpus = st.selectbox("Select Corpa", corpus['text'].index.tolist())
    
    if selected_corpus:
        # selected_corpus = selected_corpus.strip("[]").replace("",'').replace("","").split(',')
        # selected_corpus = [i[1:-1] for i in selected_corpus]
        corp = ast.literal_eval(corpus['text'].iloc[int(selected_corpus)])
        st.session_state.text_area = corp
        text_area = st.text_area("corpus text", value=st.session_state.text_area)
        padding_mat = [vocab[word] for word in corp]
        fixed_seq_len = 8
        pad_idx = 0  # Padding index
        device = 'cpu'  # Assuming you're using CPU
        print(padding_mat)
        # # Pad the batch
        padded_batch = inference().pad_batch_to_fixed_length(padding_mat, fixed_seq_len, pad_idx, device)

        print("Padded batch:")
        print(padded_batch)
        te = torch.tensor(padded_batch, dtype=torch.long).unsqueeze(0)
        size_ = te.size(1)
        loss_fn  = nn.BCEWithLogitsLoss()
        lable = int(corpus['label'].iloc[int(selected_corpus)])
        model = load_document_classification_model()
        with torch.no_grad():
            output = model(te.view(1,-1), size_)
            loss = loss_fn(output.squeeze().float(), torch.tensor(lable, dtype=torch.float))  # Apply loss function directly on logits
            output_prob = torch.sigmoid(output).squeeze()  # Apply sigmoid to get probabilities
            prediction = (output_prob > 0.5).float()
            print(prediction, output_prob)

            
            
    if st.button('Classify Document'):
        if prediction == 0:
            st.write('Document Classification:', "Bad review")
        else:
            st.write('Document Clasification:', 'Good review')