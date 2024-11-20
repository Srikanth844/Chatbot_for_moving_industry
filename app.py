import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Set page config
st.set_page_config(page_title="Moving Company Chatbot", page_icon="🚚", layout="centered")

# Custom CSS for chat-like interface
st.markdown("""
<style>
.chat-container {
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 10px;
    margin-bottom: 10px;
    max-height: 400px;
    overflow-y: auto;
}
.user-message {
    background-color: #e6f3ff;
    padding: 5px 10px;
    border-radius: 15px;
    margin: 5px 0;
    text-align: right;
}
.bot-message {
    background-color: #f0f0f0;
    padding: 5px 10px;
    border-radius: 15px;
    margin: 5px 0;
    text-align: left;
}
.stTextInput>div>div>input {
    border-radius: 20px;
}
           
</style>
""", unsafe_allow_html=True)

# Initialize Streamlit session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'waiting_for_question' not in st.session_state:
    st.session_state.waiting_for_question = True
if 'quote_stage' not in st.session_state:
    st.session_state.quote_stage = 0

# Load and process CSV data
@st.cache_data
def load_qa_data():
    df = pd.read_csv('movers.csv', header=None, names=['question', 'answer'])
    return df

qa_data = load_qa_data()

# Set up Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "moving-company"

# Ensure index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# Set up embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to convert CSV data to vectors and upload to Pinecone
def upload_to_pinecone(df):
    batch_size = 100
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        ids = batch.index.astype(str).tolist()
        texts = (batch['question'] + " " + batch['answer']).tolist()
        embeddings = model.encode(texts).tolist()
        
        to_upsert = list(zip(ids, embeddings, [{"question": q, "answer": a} for q, a in zip(batch['question'], batch['answer'])]))
        
        index.upsert(vectors=to_upsert)
    
    print(f"Uploaded {len(df)} vectors to Pinecone")

# Upload data to Pinecone
upload_to_pinecone(qa_data)

# Set up LLM
llm = OpenAI(temperature=0.7, api_key=os.getenv("OPENAI_API_KEY"))

# Function to get response from LLM
def get_llm_response(query, context):
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="You are MoveMaster, a friendly AI assistant for a moving company. Use the following context to answer the question. If you can't answer based on the context, say you don't know but provide general moving advice. Keep your responses concise and friendly.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(context=context, question=query)

# Function to handle user query
def handle_query(query):
    query_embedding = model.encode(query).tolist()
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    context = " ".join([match['metadata']['answer'] for match in results['matches']])
    response = get_llm_response(query, context)
    return response

# Quote calculation function (simplified)
def calculate_quote(distance, size, additional_services):
    base_rate = 500
    distance_rate = distance * 2
    size_multiplier = {"Small": 1, "Medium": 1.5, "Large": 2}[size]
    services_cost = len(additional_services) * 100
    total = (base_rate + distance_rate) * size_multiplier + services_cost
    return total

# ... rest of your Streamlit UI code ...

# Streamlit UI
st.title("🚚 Moving Company Chatbot")

# Chat container
# st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in st.session_state.chat_history:
    if message['role'] == 'You':
        st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message">🤖 {message["content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Handle user input
if st.session_state.waiting_for_question:
    user_input = st.text_input("Ask a question or type 'quote' to get a moving quote:", key="user_input")
    if user_input:
        if user_input.lower() == 'quote':
            st.session_state.quote_stage = 1
            st.session_state.waiting_for_question = False
        else:
            response = handle_query(user_input)
            st.session_state.chat_history.append({"role": "You", "content": user_input})
            st.session_state.chat_history.append({"role": "Bot", "content": response})
            st.session_state.waiting_for_question = False
        st.rerun()
else:
    if st.button("Yes, I have another question"):
        st.session_state.waiting_for_question = True
        st.rerun()
    
    if st.button("No, I'm done"):
        st.write("Thank you for chatting with us. If you need anything else, feel free to start a new conversation.")
        st.session_state.chat_history = []
        st.session_state.waiting_for_question = True
        st.rerun()

# Quote request section
if st.session_state.quote_stage > 0:
    st.subheader("Get a Quote")
    if st.session_state.quote_stage == 1:
        distance = st.number_input("Enter the moving distance (in miles):", min_value=1)
        if st.button("Next"):
            st.session_state.quote_stage = 2
            st.session_state.distance = distance
            st.rerun()

    elif st.session_state.quote_stage == 2:
        size = st.selectbox("Select the size of your move:", ["Small", "Medium", "Large"])
        if st.button("Next"):
            st.session_state.quote_stage = 3
            st.session_state.size = size
            st.rerun()

    elif st.session_state.quote_stage == 3:
        additional_services = st.multiselect(
            "Select any additional services:",
            ["Packing", "Unpacking", "Storage", "Furniture Assembly"]
        )
        if st.button("Get Quote"):
            quote = calculate_quote(st.session_state.distance, st.session_state.size, additional_services)
            quote_response = f"Your estimated quote: ${quote:.2f}"
            st.session_state.chat_history.append({"role": "Bot", "content": quote_response})
            st.session_state.quote_stage = 0
            st.session_state.waiting_for_question = False
            st.rerun()

    if st.button("Cancel Quote"):
        st.session_state.quote_stage = 0
        st.session_state.waiting_for_question = True
        st.rerun()

# Reset chat button
if st.button("Reset Chat"):
    st.session_state.chat_history = []
    st.session_state.waiting_for_question = True
    st.session_state.quote_stage = 0
    st.rerun()