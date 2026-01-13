# Round 4: GenAI & Agentic AI - BASIC Questions

## Libraries, Tools & Definitions

---

### Q1: What libraries exist for building AI agents?

**VP Answer:**
```
"The landscape of agent frameworks has evolved rapidly:

┌─────────────────────────────────────────────────────────────────┐
│                  AI AGENT FRAMEWORKS                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LANGCHAIN                                                      │
│  - Most popular, general-purpose                                │
│  - Chains, agents, tools, memory                                │
│  - Great for RAG, simple agents                                 │
│  - Weakness: Can be heavy, debugging tricky                     │
│                                                                 │
│  LANGGRAPH                                                      │
│  - From LangChain team                                          │
│  - Stateful, graph-based workflows                              │
│  - Better for complex multi-step agents                         │
│  - Deterministic control flow                                   │
│                                                                 │
│  LLAMAINDEX                                                     │
│  - Specialized for data/retrieval                               │
│  - Excellent indexing strategies                                │
│  - Best for document QA                                         │
│  - Lighter than LangChain                                       │
│                                                                 │
│  AUTOGEN (Microsoft)                                            │
│  - Multi-agent conversations                                    │
│  - Code execution built-in                                      │
│  - AssistantAgent, UserProxyAgent                               │
│  - Great for coding tasks                                       │
│                                                                 │
│  CREWAI                                                         │
│  - Role-based agents                                            │
│  - Simulates team collaboration                                 │
│  - Good for complex workflows                                   │
│  - Higher-level abstraction                                     │
│                                                                 │
│  SEMANTIC KERNEL (Microsoft)                                    │
│  - Enterprise focused                                           │
│  - Plugin architecture                                          │
│  - Strong typing, .NET/Python                                   │
│                                                                 │
│  HAYSTACK (deepset)                                             │
│  - NLP/search focused                                           │
│  - Production-ready pipelines                                   │
│  - Good for enterprise search                                   │
└─────────────────────────────────────────────────────────────────┘

MY SELECTION CRITERIA:

For RAG applications: LlamaIndex or LangChain
For multi-agent systems: AutoGen or CrewAI
For production workflows: LangGraph
For enterprise with .NET stack: Semantic Kernel

In banking, I prioritize:
- Auditability (can I trace every decision?)
- Control (can I add guardrails?)
- Stability (production-ready?)"
```

---

### Q2: What are the main LangChain components?

**VP Answer:**
```
"LangChain has evolved significantly. Key components:

1. MODELS (LLM Wrappers)

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

llm = ChatOpenAI(model='gpt-4', temperature=0)

2. PROMPTS (Templates & Engineering)

from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful banking assistant.'),
    ('human', '{question}')
])

3. CHAINS (Composable Workflows)

from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=template)
result = chain.invoke({'question': 'What is APR?'})

# LCEL (LangChain Expression Language) - Modern approach
chain = template | llm | output_parser

4. MEMORY (Conversation State)

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
# Tracks conversation history

5. TOOLS (External Capabilities)

from langchain.tools import Tool

tools = [
    Tool(name='calculator', func=calculate, description='Math operations'),
    Tool(name='search', func=search_docs, description='Search knowledge base')
]

6. AGENTS (Autonomous Decision Making)

from langchain.agents import create_react_agent

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

7. RETRIEVERS (Document Retrieval)

from langchain.vectorstores import Chroma

vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

8. DOCUMENT LOADERS

from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader('policy.pdf')
docs = loader.load()

ARCHITECTURE SUMMARY:

User Query → Prompt → LLM → (Tools/Retriever) → Response
              ↑                    │
              └──── Memory ────────┘"
```

---

### Q3: What is RAG? Why use it?

**VP Answer:**
```
"RAG (Retrieval Augmented Generation) grounds LLM responses in your data.

┌─────────────────────────────────────────────────────────────────┐
│                      RAG ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. INDEXING (Offline)                                          │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐              │
│  │ Documents │ →  │  Chunk    │ →  │  Embed    │ → Vector DB   │
│  │  (PDF,    │    │  (Split   │    │  (Convert │               │
│  │   docs)   │    │   text)   │    │   to vec) │               │
│  └───────────┘    └───────────┘    └───────────┘              │
│                                                                 │
│  2. RETRIEVAL + GENERATION (Online)                             │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐              │
│  │   Query   │ →  │  Retrieve │ →  │  Generate │ → Response    │
│  │           │    │  (Top-K   │    │  (LLM +   │               │
│  │           │    │   docs)   │    │  Context) │               │
│  └───────────┘    └───────────┘    └───────────┘              │
└─────────────────────────────────────────────────────────────────┘

WHY RAG OVER FINE-TUNING:

┌─────────────────────────────────────────────────────────────────┐
│  Factor           RAG                  Fine-tuning              │
├─────────────────────────────────────────────────────────────────┤
│  Data freshness   Update vector DB     Retrain model            │
│  Cost             Low (no training)    High (GPU hours)         │
│  Governance       Clear data lineage   Opaque weights           │
│  Hallucination    Reduced (grounded)   Still possible           │
│  Domain adapt     Via retrieval        Via training data        │
│  Setup time       Hours                Days/weeks               │
└─────────────────────────────────────────────────────────────────┘

WHY RAG IN BANKING:

1. COMPLIANCE
   - Can trace every response to source document
   - Required for regulatory audit

2. DATA FRESHNESS
   - Policies change weekly
   - Update vector store, not model

3. COST
   - No fine-tuning compute costs
   - Pay only for inference

4. SECURITY
   - Data stays in your infrastructure
   - No training data exposure

BASIC RAG IMPLEMENTATION:

from langchain.chains import RetrievalQA

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',  # or 'map_reduce', 'refine'
    retriever=vectorstore.as_retriever()
)

# Query
response = qa_chain.invoke('What is our vacation policy?')"
```

---

### Q4: What vector databases exist? When would you use each?

**VP Answer:**
```
"Vector databases store embeddings for similarity search:

┌─────────────────────────────────────────────────────────────────┐
│                   VECTOR DATABASE COMPARISON                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CHROMADB                                                       │
│  ├─ Open source, lightweight                                    │
│  ├─ In-memory or persistent                                     │
│  ├─ Metadata filtering built-in                                 │
│  ├─ Great for: Prototyping, small-medium scale                  │
│  └─ Limit: Performance degrades >1M vectors                     │
│                                                                 │
│  FAISS (Facebook)                                               │
│  ├─ Highly optimized C++ library                                │
│  ├─ Multiple index types (IVF, HNSW, PQ)                        │
│  ├─ No persistence (add yourself)                               │
│  ├─ Great for: High-performance, large scale                    │
│  └─ Limit: No built-in metadata filtering                       │
│                                                                 │
│  PINECONE                                                       │
│  ├─ Fully managed cloud service                                 │
│  ├─ Auto-scaling, high availability                             │
│  ├─ Metadata filtering, namespaces                              │
│  ├─ Great for: Production without ops burden                    │
│  └─ Limit: Cost, data leaves your infra                         │
│                                                                 │
│  WEAVIATE                                                       │
│  ├─ Open source, GraphQL API                                    │
│  ├─ Hybrid search (vector + keyword)                            │
│  ├─ Built-in vectorization                                      │
│  ├─ Great for: Hybrid search needs                              │
│  └─ Limit: Resource intensive                                   │
│                                                                 │
│  MILVUS                                                         │
│  ├─ Open source, distributed                                    │
│  ├─ Designed for billions of vectors                            │
│  ├─ GPU acceleration available                                  │
│  ├─ Great for: Enterprise scale                                 │
│  └─ Limit: Complex setup                                        │
│                                                                 │
│  QDRANT                                                         │
│  ├─ Rust-based, fast                                            │
│  ├─ Rich filtering capabilities                                 │
│  ├─ Good Python SDK                                             │
│  ├─ Great for: Production with filtering needs                  │
│  └─ Limit: Smaller community than others                        │
│                                                                 │
│  PGVECTOR                                                       │
│  ├─ PostgreSQL extension                                        │
│  ├─ SQL familiar, ACID compliant                                │
│  ├─ Combine with relational data                                │
│  ├─ Great for: Existing Postgres infrastructure                 │
│  └─ Limit: Performance at very large scale                      │
└─────────────────────────────────────────────────────────────────┘

MY DECISION FRAMEWORK:

Prototyping:         ChromaDB (quick setup)
Production cloud:    Pinecone (managed, reliable)
Production on-prem:  FAISS + Redis (performance + persistence)
Hybrid search:       Weaviate or Qdrant
Existing Postgres:   pgvector
Billions of vectors: Milvus"
```

---

### Q5: What is an embedding? How are they created?

**VP Answer:**
```
"Embeddings are dense vector representations that capture semantic meaning.

CONCEPT:

Text: 'The cat sat on the mat'
       ↓ Embedding Model
Vector: [0.23, -0.15, 0.87, 0.02, ..., 0.45]  (768-3072 dimensions)

Similar meanings → Similar vectors (close in vector space)

EMBEDDING MODELS:

1. OPENAI EMBEDDINGS
   from openai import OpenAI
   client = OpenAI()

   response = client.embeddings.create(
       input='Your text here',
       model='text-embedding-3-small'  # or large
   )
   vector = response.data[0].embedding

   Models:
   - text-embedding-3-small: 1536 dims, cheaper
   - text-embedding-3-large: 3072 dims, better quality

2. SENTENCE TRANSFORMERS (Open Source)
   from sentence_transformers import SentenceTransformer

   model = SentenceTransformer('all-MiniLM-L6-v2')
   embeddings = model.encode(['text 1', 'text 2'])

   Popular models:
   - all-MiniLM-L6-v2: Fast, good quality
   - all-mpnet-base-v2: Higher quality
   - instructor-xl: Task-specific instructions

3. COHERE EMBEDDINGS
   import cohere
   co = cohere.Client(api_key)

   response = co.embed(
       texts=['Your text'],
       model='embed-english-v3.0',
       input_type='search_document'
   )

HOW EMBEDDINGS ARE TRAINED:

1. CONTRASTIVE LEARNING
   - Similar pairs should be close
   - Dissimilar pairs should be far
   - Example: (query, relevant_doc) pairs

2. MASKED LANGUAGE MODELING
   - BERT-style: Predict masked words
   - Learns contextual representations

3. NEXT TOKEN PREDICTION
   - GPT-style: Predict next word
   - Learns sequential patterns

CHOOSING EMBEDDING MODEL:

┌─────────────────────────────────────────────────────────────────┐
│  Consideration        Recommendation                            │
├─────────────────────────────────────────────────────────────────┤
│  Best quality         OpenAI text-embedding-3-large             │
│  Cost sensitive       Sentence Transformers (free)              │
│  On-premise required  Sentence Transformers, BGE                │
│  Multilingual         multilingual-e5-large                     │
│  Domain-specific      Fine-tune or domain embeddings            │
└─────────────────────────────────────────────────────────────────┘"
```

---

### Q6: What prompt engineering techniques exist?

**VP Answer:**
```
"Prompt engineering is crucial for reliable LLM outputs:

1. ZERO-SHOT PROMPTING

'Classify this text as positive or negative: {text}'

No examples, relies on model knowledge.
Use when: Task is clear, model should know

2. FEW-SHOT PROMPTING

'''
Classify sentiment:

Text: 'Great product!' → Positive
Text: 'Terrible service' → Negative
Text: '{new_text}' →
'''

Provide examples to guide format and task.
Use when: Specific format needed, edge cases

3. CHAIN-OF-THOUGHT (CoT)

'Solve step by step:
Q: If I have 3 apples and buy 2 more, how many?
A: Let me think step by step.
   - Start: 3 apples
   - Buy: 2 more apples
   - Total: 3 + 2 = 5 apples
   Answer: 5 apples

Q: {new_question}
A: Let me think step by step.'

Forces reasoning, improves accuracy.
Use when: Complex reasoning, math, logic

4. ROLE PROMPTING

'You are a senior risk analyst at a major bank.
You prioritize accuracy and cite sources.
You flag uncertainty explicitly.

User: {question}'

Sets context and behavior expectations.
Use when: Need specific expertise/tone

5. STRUCTURED OUTPUT

'Extract the following information and return as JSON:
- name: string
- date: YYYY-MM-DD
- amount: number

Text: {document}'

Forces consistent, parseable output.
Use when: Need structured data extraction

6. SELF-CONSISTENCY

Run same prompt multiple times, take majority vote.
Improves reliability at cost of latency/cost.

PROMPT STRUCTURE BEST PRACTICES:

┌─────────────────────────────────────────────────────────────────┐
│  SYSTEM: Role, context, rules, constraints                      │
│  USER: Clear instruction + context + examples + input           │
│  OUTPUT FORMAT: Expected structure, examples                    │
└─────────────────────────────────────────────────────────────────┘

BANKING-SPECIFIC PATTERNS:

- Always include: 'If unsure, say so explicitly'
- Add: 'Do not make up information'
- Include: 'Base answers only on provided context'
- Require: 'Cite sources for claims'"
```

---

### Q7: What is a prompt template?

**VP Answer:**
```
"Prompt templates are reusable, parameterized prompts.

BASIC TEMPLATE:

from langchain.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=['product', 'question'],
    template='''
    You are a product expert for {product}.
    Answer the following question accurately.

    Question: {question}
    Answer:
    '''
)

# Use template
prompt = template.format(product='credit cards', question='What is APR?')

CHAT TEMPLATES:

from langchain.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful {role} assistant.'),
    ('human', '{question}'),
])

# Format with variables
messages = chat_template.format_messages(
    role='banking',
    question='How do I apply for a loan?'
)

ADVANCED TEMPLATES:

# With examples (few-shot)
from langchain.prompts import FewShotPromptTemplate

examples = [
    {'input': 'What is APR?', 'output': 'Annual Percentage Rate...'},
    {'input': 'What is APY?', 'output': 'Annual Percentage Yield...'}
]

few_shot_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix='Answer banking questions:',
    suffix='Input: {input}\nOutput:',
    input_variables=['input']
)

WHY TEMPLATES MATTER:

1. CONSISTENCY
   - Same format across application
   - Easier testing and validation

2. VERSIONING
   - Track prompt changes
   - A/B test different prompts

3. SEPARATION OF CONCERNS
   - Logic separate from prompts
   - Non-technical team can update prompts

4. INJECTION PREVENTION
   - Escape user input properly
   - Structured input handling"
```

---

### Q8: What is the difference between completion and chat models?

**VP Answer:**
```
"The key difference is in how they're trained and used:

COMPLETION MODELS (Legacy)

Input:  'The capital of France is'
Output: 'Paris.'

- Continue text from prompt
- Single turn
- Models: GPT-3 (davinci), text-davinci-003

CHAT MODELS (Modern)

Input:  [
    {'role': 'system', 'content': 'You are helpful.'},
    {'role': 'user', 'content': 'What is the capital of France?'},
    {'role': 'assistant', 'content': 'The capital is Paris.'},
    {'role': 'user', 'content': 'What about Germany?'}
]
Output: {'role': 'assistant', 'content': 'The capital of Germany is Berlin.'}

- Conversation format
- Multi-turn with context
- Models: GPT-4, Claude, Llama-2-chat

API DIFFERENCES:

# Completion API (deprecated for most use cases)
response = openai.completions.create(
    model='gpt-3.5-turbo-instruct',
    prompt='Tell me about...'
)

# Chat API (recommended)
response = openai.chat.completions.create(
    model='gpt-4',
    messages=[
        {'role': 'system', 'content': '...'},
        {'role': 'user', 'content': '...'}
    ]
)

PRACTICAL IMPLICATIONS:

Chat models:
- Better instruction following
- Built-in conversation handling
- System prompt for behavior control
- More suitable for most applications

Use completion only for:
- Pure text continuation
- Legacy system compatibility
- Specific fine-tuned completion models"
```

---

### Q9: What is token and why does it matter?

**VP Answer:**
```
"Tokens are the fundamental units LLMs process - not words, not characters.

TOKENIZATION:

'Hello, how are you today?'
↓ Tokenizer
['Hello', ',', ' how', ' are', ' you', ' today', '?']
= 7 tokens

'Pneumonoultramicroscopicsilicovolcanoconiosis'
↓ Tokenizer
['Pne', 'um', 'ono', 'ultra', 'micro', 'scop', 'ic', 'sil', 'ico', 'vol', 'can', 'ocon', 'iosis']
= 13 tokens

RULES OF THUMB:
- 1 token ≈ 4 characters in English
- 1 token ≈ 0.75 words
- 100 tokens ≈ 75 words

WHY IT MATTERS:

1. COST
   - Pricing is per token (input + output)
   - GPT-4: ~$0.03/1K input, ~$0.06/1K output
   - Long contexts = higher costs

2. CONTEXT LIMITS
   - Each model has max tokens
   - GPT-4: 8K, 32K, or 128K depending on version
   - Claude: Up to 200K
   - Must fit: system prompt + context + query + response

3. RESPONSE LENGTH
   - max_tokens parameter limits output
   - Set appropriately to control costs/length

TOKEN COUNTING:

import tiktoken

encoder = tiktoken.encoding_for_model('gpt-4')
tokens = encoder.encode('Your text here')
token_count = len(tokens)

CONTEXT WINDOW MANAGEMENT:

Available = Max tokens - System prompt - Expected response
Available = 8000 - 500 - 1000 = 6500 tokens for context

If retrieving documents for RAG:
- Each doc chunk ~500 tokens
- Can fit ~13 chunks in context

COST OPTIMIZATION:
- Shorter system prompts
- Truncate irrelevant context
- Use smaller models when possible
- Cache common queries"
```

---

### Q10: What are temperature and top_p parameters?

**VP Answer:**
```
"Temperature and top_p control randomness in LLM outputs.

TEMPERATURE (0.0 - 2.0)

Controls probability distribution sharpness.

Temperature = 0.0 (Deterministic):
- Always pick highest probability token
- Consistent, predictable outputs
- Use for: Factual QA, classification, extraction

Temperature = 0.7 (Balanced):
- Mix of likely and less likely tokens
- Some creativity, mostly coherent
- Use for: General chat, writing assistance

Temperature = 1.0+ (Creative):
- More random sampling
- More diverse, potentially inconsistent
- Use for: Creative writing, brainstorming

VISUALIZATION:

Token probabilities: [A: 0.7, B: 0.2, C: 0.1]

Temp=0:   Always picks 'A'
Temp=0.5: Usually 'A', sometimes 'B'
Temp=1.0: Proportional to probabilities
Temp=2.0: Nearly random

TOP_P (Nucleus Sampling, 0.0 - 1.0)

Only sample from top tokens whose cumulative probability = p.

Top_p = 0.9:
- Consider tokens until 90% cumulative probability
- Cuts off unlikely tokens

Top_p = 1.0:
- Consider all tokens
- No filtering

INTERACTION:

Generally use ONE, not both:
- Set temperature, keep top_p=1.0
- Or set top_p, keep temperature=1.0

MY PRODUCTION SETTINGS:

Factual/extraction: temperature=0
Conversational: temperature=0.7
Creative: temperature=1.0

response = client.chat.completions.create(
    model='gpt-4',
    messages=messages,
    temperature=0,        # Deterministic
    max_tokens=1000,      # Limit output
    top_p=1.0            # Don't filter
)"
```

---

## Practice Questions

1. What is the difference between LangChain and LlamaIndex?
2. What embedding dimension should you use and why?
3. Explain the concept of semantic search
4. What is a chain in LangChain?
5. How do you handle rate limits with LLM APIs?
