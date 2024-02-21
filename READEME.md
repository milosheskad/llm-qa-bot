###docker build -t aws-documentation .
docker run -p 8501:8501 doc_chat


Temperature – Large language models use probability to construct the words in a sequence. For any given next word, there is a probability distribution of options for the next word in the sequence. When you set the temperature closer to zero, the model tends to select the higher-probability words. When you set the temperature further away from zero, the model may select a lower-probability word.

In technical terms, the temperature modulates the probability density function for the next tokens, implementing the temperature sampling technique. This parameter can deepen or flatten the density function curve. A lower value results in a steeper curve with more deterministic responses, and a higher value results in a flatter curve with more random responses.

Top K – Temperature defines the probability distribution of potential words, and Top K defines the cut off where the model no longer selects the words. For example, if K=50, the model selects from 50 of the most probable words that could be next in a given sequence. This reduces the probability that an unusual word gets selected next in a sequence. In technical terms, Top K is the number of the highest-probability vocabulary tokens to keep for Top- K-filtering - This limits the distribution of probable tokens, so the model chooses one of the highest- probability tokens.

Top P – Top P defines a cut off based on the sum of probabilities of the potential choices. If you set Top P below 1.0, the model considers the most probable options and ignores less probable ones. Top P is similar to Top K, but instead of capping the number of choices, it caps choices based on the sum of their probabilities. For the example prompt "I hear the hoof beats of ," you may want the model to provide "horses," "zebras" or "unicorns" as the next word. If you set the temperature to its maximum, without capping Top K or Top P, you increase the probability of getting unusual results such as "unicorns." If you set the temperature to 0, you increase the probability of "horses." If you set a high temperature and set Top K or Top P to the maximum, you increase the probability of "horses" or "zebras," and decrease the probability of "unicorns."

https://labelbox.com/blog/how-vector-similarity-search-works/
The first process in a building a contextual-aware chatbot is to generate embeddings for the context. Typically, you will have an ingestion process which will run through your embedding model and generate the embeddings which will be stored in a sort of a vector store. In this example we are using a Titan embeddings model for this.
Bedrock currently offers Titan Embeddings for text embedding that supports text similarity (finding the semantic similarity between bodies of text) and text retrieval (such as search).
At the time of writing you can use amazon.titan-embed-g1-text-01 as embedding model via the API. The input text size is 8192 tokens and the output vector length is 1536.

chunking
A large document (or a giant file appending small ones) is loaded
Langchain utility is used to split it into multiple smaller chunks (chunking)
First chunk is sent to the model; Model returns the corresponding summary
Langchain gets next chunk and appends it to the returned summary and sends the combined text as a new request to the model; the process repeats until all chunks are processed
In the end, you have final summary based on entire content


FAISS as VectorStore
https://arxiv.org/pdf/1702.08734.pdf
In order to be able to use embeddings for search, we need a store that can efficiently perform vector similarity searches. In this notebook we use FAISS, which is an in memory store. For permanently store vectors, one can use pgVector, Pinecone or Chroma.