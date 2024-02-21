import os
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
import boto3
import tqdm

class MarkdownHandler(object):
    """ read documents, divide it into chunks and save it into Chroma DB"""

    def __init__(self):
        self.documents = []
        self.bedrock = boto3.client(service_name='bedrock-runtime')
        self.bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=self.bedrock)
        self.persist_directory = r'chroma_db'
        # self.db = None

    def read_documents(self, docs_dir: str):
        """
        Reads all markdown files from a folder structure as langchain documents
        """
        glob = Path(docs_dir).glob
        ps = list(glob("**/*.md"))
        for p in tqdm.tqdm(ps, "Loading documents"):
            if os.path.splitext(p)[1] != '.md':
                continue
            document = UnstructuredMarkdownLoader(p, encoding="utf-8").load()[0]
            document.metadata["source"] = str(document.metadata['source'])
            self.documents.append(document)
        return self.documents

    def split_child_node(self, path: str) -> list:
        """
        Splits child nodes and returns a list of documents
        """
        chunk_size = 1000
        chunk_overlap = 1000
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        documents = text_splitter.split_documents(self.read_documents(path))
        return documents

    def load_documentation_folder(self, docs_dir):
        """
        Creates or loads a Chroma vector database depending on whether it exists for the passed documentation
        """
        if not os.path.exists(self.persist_directory):
            documents = self.split_child_node(docs_dir)
            chroma = Chroma.from_documents(documents, self.bedrock_embeddings, persist_directory=self.persist_directory)
        else:
            chroma = Chroma(persist_directory=self.persist_directory, embedding_function=self.bedrock_embeddings)
        return chroma
