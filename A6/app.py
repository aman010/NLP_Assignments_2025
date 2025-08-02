#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 17:47:40 2025

@author: qb
"""

import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, pipeline, AutoModelForSeq2SeqLM
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFacePipeline



# Set the device (e.g., 'cuda' or 'cpu')
def get_embedding_model():
    model_name = 'hkunlp/instructor-base'
    
    embedding_model = HuggingFaceInstructEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"}  # Avoid passing any unsupported arguments like 'token'
    )
    return embedding_model


# Streamlit Application
def chat_interface():
    st.title("Chat with AI - Get Responses and Source Documents")
    

    # Create a chat interface for user input
    st.subheader("Ask a Question:")
    st.write("""
You are an assistant to provide information about myself, please ask questions:<br>
1) How old are you?<br><br>
2) What is your highest level of education?<br><br>
3) What major or field of study did you pursue during your education?<br><br>
4) How many years of work experience do you have?<br><br>
5) What type of work or industry have you been involved in?<br><br>
6) Can you describe your current role or job responsibilities?<br><br>
7) What are your core beliefs regarding the role of technology in shaping society?<br><br>
8) How do you think cultural values should influence technological advancements?<br><br>
9) As a master’s student, what is the most challenging aspect of your studies so far?<br><br>
10) What specific research interests or academic goals do you hope to achieve during your time as a master's student?
""", unsafe_allow_html=True)
    # Chat input box for the user
    user_input = st.text_input("Ask your question:")

    if user_input:
        # Retrieve relevant documents based on the user query
        query = user_input

        # Initialize embedding model
        embedding_model = get_embedding_model()
        vectordb = FAISS.load_local(
            folder_path="/home/qb/NLP_Assignment/A6/requires/vector_db_path/nlp_stanford",
            embeddings=embedding_model,
            index_name='nlp'  # Default index
        )   
        
        # Ready to use retriever
        retriever = vectordb.as_retriever()

        # model_id = 'lmsys/fastchat-t5-3b-v1.0'
        # tokenizer = T5Tokenizer.from_pretrained(model_id)
        # tokenizer.pad_token_id = tokenizer.eos_token_id


        # model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map='cpu')

        # pipe = pipeline(task="text2text-generation", model=model, tokenizer=tokenizer)

        # llm = HuggingFacePipeline(pipeline=pipe)
        
        
        # question_generator = LLMChain(llm = llm,prompt = CONDENSE_QUESTION_PROMPT,verbose = True)
        # query = 'Comparing both of them'
        # chat_history = "Human:What is your age?\nAI:\nHuman:What industry your involved with?\nAI:"
        # question_generator({'chat_history' : chat_history, "question" : query})
        
        

        # Setup conversation memory
        memory = ConversationBufferMemory()

        # Define the prompt template for Q&A
        prompt_temp = """
            You are an assistant to provide information about myself, please ask questions:
            1) How old are you?
            2) What is your highest level of education?
            3) What major or field of study did you pursue during your education?
            4) How many years of work experience do you have?
            5) What type of work or industry have you been involved in?
            6) Can you describe your current role or job responsibilities?
            7) What are your core beliefs regarding the role of technology in shaping society?
            8) How do you think cultural values should influence technological advancements?
            9) As a master’s student, what is the most challenging aspect of your studies so far?
            10) What specific research interests or academic goals do you hope to achieve during your time as a master's student?

            {context}
            Question: {question}
            Answer:
            """.strip()


        prompt = PromptTemplate.from_template(template=prompt_temp)
        

        # Load QA chain
        # doc_chain = load_qa_chain(
        #     llm=llm,
        #     chain_type='stuff',
        #     prompt=prompt,
        #     verbose=True
        # )


        qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
        # Retrieve relevant documents based on query
        input_documents = retriever.get_relevant_documents(query)
        # context = " ".join([doc['text'] for doc in input_documents])  # Combine the text content of documents
        context = " ".join([doc.page_content for doc in input_documents])
        # Generate response using the model and context
        # res = doc_chain({'input_documents': input_documents, 'question': query})
        def qa_with_distilbert(query, context):
            result = qa_pipeline(question=query, context=context)
            return result['answer']
        
        answer = qa_with_distilbert(query, context)

        # Display the response and the relevant source documents
        st.subheader("Response:")
        st.write(answer)  # Assuming the result contains the 'output' field

        # st.subheader("Relevant Source Documents:")
        # for i, doc in enumerate(input_documents):
        #     st.write(f"Source {i+1}: {doc['source']}")
        #     st.write(f"Content: {doc['text'][:300]}...")  # Display a snippet of the document


# Run the app
if __name__ == "__main__":
    chat_interface()
