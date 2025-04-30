import os
from langchain.vectorstores import Chroma
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings



def clean_text(text):
    lines = text.splitlines()
    cleaned = [line.strip() for line in lines if line.strip()]
    return "\n".join(cleaned) 

embedding_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# embedding_fn = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./vectorDB_index", embedding_function=embedding_fn)

all_chunks = []

for filename in os.listdir("Resources"):
    print (filename)
    if filename.endswith(".pdf"):
        loader = PDFPlumberLoader(f"Resources/{filename}")
        pages = loader.load()
        # print (pages)
        card_name = filename.replace(".pdf", "").replace(" ", "_")

        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

        for i, page in enumerate(pages):
            page_text = clean_text(page.page_content)

            # print (page_text)
            # print ('\n')
            page.metadata["card_name"] = card_name
            page.metadata["source_type"] = "text"
            page.metadata["page_number"] = i + 1

            doc = Document(page_content=page_text, metadata=page.metadata)

            # print (doc)
            # print ('\n')

            chunks = splitter.split_documents([doc])
            all_chunks.extend(chunks)


db.add_documents(all_chunks)
