import streamlit as st
from rag_pipeline import (
    load_vector_store,
    retrieve_relevant_chunks,
    generate_answer
)

st.set_page_config(page_title="Ask ML Papers", layout="centered") #age layout
st.title("Ask Questions About Your ML Papers")

query = st.text_input("Enter your question:") #label displayed for iinput field

if query:
    with st.spinner("Thinking..."): #loading indicator
        vectordb = load_vector_store()
        chunks = retrieve_relevant_chunks(query)
        answer = generate_answer(query, chunks)

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Sources"): # collapisble section
        for i, doc in enumerate(chunks):
            st.markdown(f"**Source {i+1}:** {doc.metadata.get('source', 'unknown')}") #markdown text in bold
            st.text(doc.page_content[:300] + "...")
