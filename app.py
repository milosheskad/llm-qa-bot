import boto3
import streamlit as st
from doc_chatbot import DocumentationChat
from langchain_community.embeddings import BedrockEmbeddings
from doc_loader import MarkdownHandler

bedrock = boto3.client(service_name='bedrock-runtime')
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)


def main():
    st.set_page_config("clementine_documentation")
    st.header("Clementine Documentation")
    user_question = st.text_input("Ask a question")

    with st.sidebar:
        st.title("Create/Update Vector Store:")
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                MarkdownHandler().load_documentation_folder('data_example')
                st.success("Done")

    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            answer = DocumentationChat().get_response_llm(user_question)
            response = answer.get('result')
            source_list = []
            documents = answer.get('source_documents')
            for i in documents:
                source = i.metadata.get('source')
                source_list.append(source)
            st.write(response)
            st.title("Here are the relevant documents:")
            st.write(source_list)
            st.success("Done")


if __name__ == "__main__":
    main()
