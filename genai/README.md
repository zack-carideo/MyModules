LangChain is a powerful, open-source framework designed to help you develop applications powered by a language model, particularly a large language model (LLM). The core idea of the library is that we can “chain” together different components to create more advanced use cases around LLMs. LangChain consists of multiple components from several modules.

![Alt text](image.png)


### OVERVIEW OF CORE MODULES 

- **Prompts:** This module allows you to build dynamic prompts using templates. It can adapt to different LLM types depending on the context window size and input variables used as context, such as conversation history, search results, previous answers, and more.
    
- **Models:** This module provides an abstraction layer to connect to most available third- party LLM APIs. It has API connections to ~40 public LLMs, chat and embedding models.

- **Memory:** This gives the LLMs access to the conversation history.
    Indexes: Indexes refer to ways to structure documents so that LLMs can best interact with them. This module contains utility functions for working with documents and integration to different vector databases.
    Agents: Some applications require not just a predetermined chain of calls to LLMs or other tools, but potentially to an unknown chain that depends on the user’s input. In these types of chains, there is an agent with access to a suite of tools. Depending on the user’s input, the agent can decide which — if any — tool to call.
- **Chains:** Using an LLM in isolation is fine for some simple applications, but many more complex ones require the chaining of LLMs, either with each other, or other experts. LangChain provides a standard interface for Chains, as well as some common implementations of chains for ease of use.


#### Links
- Running langchain with local model: https://python.langchain.com/docs/integrations/llms/llamacpp




#### Demos
- **ir_merger_retriver.py:** *indexing and retreival with chroma and a variety of embeddings (we are creating a reciever of recievers!) with contextual compression*
![Alt text](image-1.png)