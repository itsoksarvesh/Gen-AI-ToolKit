import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama

# Initialize the Ollama model
ollama_llm = Ollama(model="llama3.2:1b")

# Define a prompt template with dynamic input
template = "{input_text}"
prompt = PromptTemplate(input_variables=["input_text"], template=template)

# Create the LLMChain with Ollama and the dynamic prompt
llm_chain = LLMChain(llm=ollama_llm, prompt=prompt)

# Streamlit app interface
st.title("Text Generation with LangChain and Ollama")

# Text input from the user
user_input = st.text_area("Ask anything:")

# Button to trigger the generation
if st.button("Generate"):
    if user_input:
        # Call the LLMChain to process the input
        output = llm_chain.run(user_input)
        
        # Display the generated output
        st.subheader("Generated Output:")
        st.write(output)
    else:
        st.write("Please enter some text to generate a response.")
