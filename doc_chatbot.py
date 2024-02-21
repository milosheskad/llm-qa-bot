import boto3
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from doc_loader import MarkdownHandler


class DocumentationChat(object):
    '''get the prompt,llm model and retrieve relevant documents and get response'''

    def __init__(self,
                 embedding_model_name: str = 'titan_embedding',
                 device: str = 'cpu',
                 db: str = None):
        prompt_template = """

            Human: Use the following pieces of context to provide a 
            concise answer to the question. If you don't know the answer, 
            just say that you don't know, don't try to make up an answer.
            <context>
            {context}
            </context>

            Question: {question}

            Assistant:"""

        self.embedding_model_name = embedding_model_name
        self.device = device
        self.llm_rag_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        bedrock = boto3.client(service_name='bedrock-runtime')
        self.llm = Bedrock(model_id="meta.llama2-13b-chat-v1", client=bedrock, model_kwargs={'max_gen_len': 512})
        self.chroma = MarkdownHandler().load_documentation_folder('data_example')

    def get_llama2_llm(self):
        bedrock = boto3.client(service_name='bedrock-runtime')
        llm = Bedrock(model_id="meta.llama2-13b-chat-v1", client=bedrock,
                      model_kwargs={'max_gen_len': 512})
        return llm

    @staticmethod
    def relevant_docs_ordered_by_similarity(query: str,
                                            db,
                                            k: int,
                                            threshold: float = 0.7):
        """
        Returns the most similar documents to the query depending on a similarity threshold
        """
        relevant_docs_tuples = db.similarity_search_with_relevance_scores(query, k=k)
        relevant_docs_tuples.sort(key=lambda a: a[1], reverse=True)
        relevant_docs = [pair[0] for pair in relevant_docs_tuples if pair[1] >= threshold]
        return relevant_docs

    def retrieve_documents(self, query: str, db):
        query = query.lower()
        result = None
        answer = 'not contain the answer'
        current_k = 0
        while 'not contain the answer' in answer and current_k <= 1:
            current_k += 1
            qa = RetrievalQA.from_chain_type(llm=self.llm,
                                             chain_type="stuff",
                                             retriever=db.as_retriever(search_type="similarity",
                                                                       search_kwargs={'k': current_k}),
                                             chain_type_kwargs={"prompt": self.llm_rag_prompt},
                                             return_source_documents=True
                                             )
            result = qa({"query": query})
            answer = result['result']
        relevant_docs = self.relevant_docs_ordered_by_similarity(query, db, current_k)
        return result, relevant_docs

    def get_response(self, query: str):
        """
        Returns relevant documents for the query
        """
        result, relevant_docs = self.retrieve_documents(
            query=query,
            db=self.chroma
        )
        return result, relevant_docs

    def get_response_llm(self, query):
        qa = RetrievalQA.from_chain_type(
            llm=self.get_llama2_llm(),
            chain_type="stuff",
            retriever=self.chroma.as_retriever(
                search_type="similarity", search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.llm_rag_prompt}
        )
        answer = qa(query)
        return answer
