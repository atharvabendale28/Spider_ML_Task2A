# Spider_ML_Task2A

sequence of events - 
1.loaded the env file for API key
2.defined path for pdfs
3.defined fucntion to extract content from pdf
4.fucntion to load the document and extract content using above fucntion (all data is now in a string)
5.split the text into chunks 
6.used pretrained embedding model to convert these chunks to vectors and save the vector DB to pkl file(vector DB will have a vector for each chunk now)
7. when we call this model, the pretrained transformer is used to geenrate a vector for it and sotred in DB, FAISS (Facebook AI Similarity Search) is a library that creates efficient vector databases for fast similarity search. It stores       document chunks as vectors and quickly finds the most similar ones when you search, enabling semantic document retrieval for RAG systems.
8.then defined the fucntion to load the db
9.function to retrieve top 5 chunks from DB
10.loaded LLM using groq API key for text generation
11.defined function to  generate answer - joined all text from texts into one string(context_text), created structured prompts with system instructions and user questions, LLM is tolf to only answer questions based on context(no hallucination)(hallucination in LLMs means the model generates information that sounds plausible but is actually made up or not based on the provided context)

12.run all the functions to creat pkl file, then commented it out
13.run the functions after that to generate answer
14.used streamlit in app.py to call all functions and form interface
