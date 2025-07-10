### Sequence of Events

1. Loaded the .env file for the API key.  
2. Defined path for PDFs.  
3. Defined function to extract content from PDFs.  
4. Defined function to load documents and extract content using the above function (all data is now in a string).  
5. Split the text into chunks.  
6. Used a pretrained embedding model to convert these chunks to vectors and saved the vector DB to a .pkl file (vector DB now contains a vector for each chunk).  
7. When we call this model, the pretrained transformer generates a vector for the query.  
   - FAISS (Facebook AI Similarity Search) is a library that creates efficient vector databases for fast similarity search.  
   - It stores document chunks as vectors and quickly finds the most similar ones, enabling semantic document retrieval for RAG systems.  
8. Defined the function to load the vector DB.  
9. Defined the function to retrieve the top 5 chunks from the DB.  
10. Loaded the LLM using the Groq API key for text generation.  
11. Defined a function to generate answers:  
    - Joined all text from the documents into one context_text string.  
    - Created structured prompts with system instructions and the user’s question.  
    - LLM is told to only answer based on context (no hallucination).  
    - _Hallucination_ in LLMs means generating information that sounds plausible but is not based on the provided context.  
12. Ran all the functions to create the .pkl file, then commented them out.  
13. Ran the functions that generate the answer.  
14. Used Streamlit in app.py to call all functions and build the interface.
