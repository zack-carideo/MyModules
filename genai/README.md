#### Demos
| File Path                                      | Description                                                                                           |
|------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| langchain_ops/demos/ir_merger_retriver.py      | indexing and retrieval with chroma and a variety of embeddings (we are creating a receiver of receivers!) with contextual compression |
| langchain_ops/demos/rag/rag_with_MapReduce.ipynb | e2e example of Information Retrieval using a Merged Retriever from langchain, and Generation using LlamaCPP bindings from langchain. |
| langchain_ops/demos/llamacpp/llamacpp_chat_prompt | quick demo of how to use langchain and llamacpp to create a simple but powerful chat agent. |

#### Links
- Running langchain with local model: [https://python.langchain.com/docs/integrations/llms/llamacpp](https://python.langchain.com/docs/integrations/llms/llamacpp)
- Merged Retriever Overview: [https://python.langchain.com/docs/integrations/retrievers/merger_retriever](https://python.langchain.com/docs/integrations/retrievers/merger_retriever)

#### RAG EVALUATION METHODS
| Use Case                                      | Recommended Framework | Metrics Used                                                                                   | Reasoning                                                                                                                         |
|-----------------------------------------------|-----------------------|------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| Initial RAG evaluations                       | RAGAS                 | Average Precision (AP), Faithfulness                                                          | RAGAS is ideal for initial evaluations, especially in environments where reference data is scarce. It focuses on precision and how faithfully the response matches the provided context. |
| Dynamic, continuous RAG deployments           | ARES                  | MRR, NDCG                                                                                      | ARES uses synthetic data and LLM judges, which are suitable for environments needing continuous updates and training and focusing on response ranking and relevance. |
| Full system traces including LLMs and Vector storage | TraceLoop             | Information Gain, Factual Consistency, Citation Accuracy                                       | TraceLoop is best suited for applications where tracing the flow and provenance of information used in the generated output is critical, such as academic research or journalism. |
| Real-time RAG monitoring                      | Arize                 | Precision, Recall, F1                                                                         | Arize excels in real-time performance monitoring, making it perfect for deployments where immediate feedback on RAG performance is essential |
| Enterprise-level RAG applications             | Galileo               | Custom metrics, Context Adherence                                                             | Galileo provides advanced insights and metrics integration for complex applications, ensuring RAG’s adherence to context.           |
| Optimizing RAG for specific domains            | TruLens               | Domain-specific accuracy, Precision                                                           | TruLens is designed to optimize RAG systems within specific domains, by enhancing the accuracy and precision of domain-relevant responses |


### Generative Summarization Evaluation Approaches

#### Supervised
- **METEOR**: A metric that measures the quality of machine-generated summaries by comparing them to human-generated reference summaries based on various linguistic features.
- **ROUGE**: A set of metrics that evaluate the overlap between machine-generated summaries and human-generated reference summaries using n-gram co-occurrence statistics.Each variant of the ROUGE metrics serves a specific purpose in evaluating machine-generated summaries. ROUGE-N focuses on content overlap, ROUGE-L captures the longest common subsequence, ROUGE-S measures skip-bigram co-occurrence, and ROUGE-SU combines skip-bigram and unigram matching. By using these variants, researchers and practitioners can assess different aspects of summary quality and compare the performance of different summarization models.
    - **ROUGE-N**: Measures the overlap of n-grams (contiguous sequences of n words) between the machine-generated summaries and human-generated reference summaries. It is commonly used to evaluate the quality of machine-generated summaries in terms of content overlap with the reference summaries.
    - **ROUGE-L**: Computes the longest common subsequence (LCS) between the machine-generated summaries and human-generated reference summaries. It focuses on capturing the longest common sequence of words, regardless of their order. ROUGE-L is useful for evaluating summaries that may have different word orderings but still convey the same information.
    - **ROUGE-S**: Evaluates the skip-bigram co-occurrence between the machine-generated summaries and human-generated reference summaries. It measures the similarity of skip-bigrams, which are pairs of words with a certain number of words in between them. ROUGE-S is particularly effective for capturing semantic similarity and evaluating summaries that may have different word orderings or contain additional words.
    - **ROUGE-SU**: Extends ROUGE-S by considering unigram matches in addition to skip-bigram matches. It combines the strengths of ROUGE-S and unigram matching to provide a more comprehensive evaluation of machine-generated summaries. ROUGE-SU is commonly used for evaluating summaries that may have different word orderings, contain additional words, or omit certain words.
`
- **BERTScore**: A metric that calculates the similarity between machine-generated summaries and human-generated reference summaries based on contextualized word embeddings from BERT.
- **Toxigen**: A metric specifically designed for evaluating the toxicity of machine-generated summaries by comparing them to human-generated reference summaries.
- **Detoxify**: A metric that measures the level of toxicity in machine-generated summaries by analyzing the presence of toxic language and providing a toxicity score.
#### Unsupervised (use llms to validation llm generated summaries)
- **Coherence**: A metric that assesses the logical flow and consistency of machine-generated summaries.
- **Accuracy**: A metric that evaluates the factual correctness of machine-generated summaries.
- **Factuality**: A metric that measures the adherence to factual information in machine-generated summaries.
- **Completeness**: A metric that assesses the level of information coverage in machine-generated summaries.


#### ROUGE Metrics Variants


