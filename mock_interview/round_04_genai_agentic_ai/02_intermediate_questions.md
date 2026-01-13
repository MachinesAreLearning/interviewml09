# Round 4: GenAI & Agentic AI - INTERMEDIATE Questions

## Trade-offs, Comparisons & Implementation Decisions

---

### Q1: Why ChromaDB vs FAISS vs Pinecone? What are the trade-offs?

**VP Answer:**
```
"The choice depends on scale, ops capability, and requirements:

┌─────────────────────────────────────────────────────────────────┐
│                    DETAILED COMPARISON                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CHROMADB                                                       │
│  ─────────                                                      │
│  Pros:                                                          │
│  + Zero setup (pip install chromadb)                            │
│  + Persistent storage built-in                                  │
│  + Metadata filtering                                           │
│  + Good Python API                                              │
│  + Free, open source                                            │
│                                                                 │
│  Cons:                                                          │
│  - Performance degrades >1M vectors                             │
│  - Single-node only (no horizontal scaling)                     │
│  - Limited index options                                        │
│                                                                 │
│  Best for: Prototyping, small apps, POCs                        │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  FAISS                                                          │
│  ─────                                                          │
│  Pros:                                                          │
│  + Extremely fast (optimized C++)                               │
│  + Multiple index types (IVF, HNSW, PQ)                         │
│  + Scales to billions of vectors                                │
│  + GPU support                                                  │
│  + Free, open source                                            │
│                                                                 │
│  Cons:                                                          │
│  - No persistence (must serialize yourself)                     │
│  - No metadata filtering (manual implementation)                │
│  - Requires ML ops knowledge                                    │
│  - More code to manage                                          │
│                                                                 │
│  Best for: High-performance production, large scale             │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PINECONE                                                       │
│  ────────                                                       │
│  Pros:                                                          │
│  + Fully managed (zero ops)                                     │
│  + Auto-scaling                                                 │
│  + High availability, backups                                   │
│  + Namespaces, metadata filtering                               │
│  + Hybrid search (vector + keyword)                             │
│                                                                 │
│  Cons:                                                          │
│  - Cost ($70/mo+, usage-based)                                  │
│  - Data in third-party cloud                                    │
│  - Vendor lock-in                                               │
│  - Latency (network call)                                       │
│                                                                 │
│  Best for: Production SaaS, teams without ML ops                │
└─────────────────────────────────────────────────────────────────┘

MY DECISION FRAMEWORK:

# Decision tree
if prototyping:
    use ChromaDB

elif production and no_ml_ops_team:
    if data_can_leave_premises:
        use Pinecone
    else:
        use Qdrant Cloud or Weaviate Cloud (self-hosted option)

elif production and have_ml_ops:
    if need_max_performance:
        use FAISS + Redis (for persistence)
    elif need_metadata_filtering:
        use Qdrant or Weaviate
    elif have_postgres_already:
        use pgvector

MY ACTUAL ARCHITECTURE:

For our internal advisor, I evolved through stages:

Phase 1 (POC): ChromaDB
- Quick iteration
- Validated approach worked

Phase 2 (Pilot): ChromaDB with persistence
- Added more documents
- Found performance limits at 500K vectors

Phase 3 (Production): FAISS + Redis
- Migrated to FAISS for performance
- Redis for persistence and metadata
- Abstracted interface so switch was config change"
```

---

### Q2: Walk me through a sample RAG application flow. What are your design choices?

**VP Answer:**
```
"Let me walk through a production RAG system I designed for policy Q&A:

┌─────────────────────────────────────────────────────────────────┐
│                  RAG APPLICATION ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INDEXING PIPELINE (Offline - Daily)                            │
│  ════════════════════════════════════                           │
│                                                                 │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  │
│  │ Ingest │→ │ Parse  │→ │ Chunk  │→ │ Embed  │→ │ Store  │  │
│  │ (S3)   │  │ (PDF,  │  │ (512   │  │ (OpenAI│  │ (Vector│  │
│  │        │  │  HTML) │  │  tokens)│  │  3-lg) │  │  DB)   │  │
│  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘  │
│                  │              │                               │
│                  ▼              ▼                               │
│            ┌─────────┐   ┌──────────┐                          │
│            │Metadata │   │ Overlap  │                          │
│            │(source, │   │ (50 tok  │                          │
│            │ date)   │   │ window)  │                          │
│            └─────────┘   └──────────┘                          │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  QUERY PIPELINE (Online - Per Request)                          │
│  ═════════════════════════════════════                          │
│                                                                 │
│  User Query                                                     │
│      │                                                          │
│      ▼                                                          │
│  ┌─────────────────┐                                           │
│  │ Query Rewriting │  'vacation policy' → 'PTO time off leave' │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │ Hybrid Retrieval│  Vector search + BM25 keyword search      │
│  │ (RRF fusion)    │                                           │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │ Reranking       │  Cross-encoder rerank top-20 → top-5      │
│  │ (Cohere)        │                                           │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │ Context Assembly│  Format chunks with citations             │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │ LLM Generation  │  GPT-4 with grounding prompt              │
│  │ + Citations     │                                           │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│      Response + Sources                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

KEY DESIGN DECISIONS:

1. CHUNKING STRATEGY
   Why 512 tokens with 50 overlap?
   - 512: Fits well in context, maintains coherence
   - Overlap: Prevents losing info at boundaries
   - Semantic chunking considered but slower

2. EMBEDDING MODEL
   Why OpenAI text-embedding-3-large?
   - Best quality for English
   - Consistent with production SLAs
   - Cost acceptable for our volume

3. HYBRID RETRIEVAL
   Why vector + BM25?
   - Vector: Semantic similarity
   - BM25: Exact keyword matching
   - Combined: Handles both 'what is PTO?' and 'policy 3.2.1'

4. RERANKING
   Why add a reranker?
   - Bi-encoders (embeddings) are fast but approximate
   - Cross-encoder reranker is slower but more accurate
   - Apply to top-20 → top-5 (not all docs)

5. GENERATION
   Why GPT-4 with specific prompt?
   - Grounding prompt: 'Only use provided context'
   - Citation requirement: 'Cite [Source X] for claims'
   - Uncertainty handling: 'Say unsure if not in context'"
```

---

### Q3: LangChain vs LlamaIndex - when would you use each?

**VP Answer:**
```
"Both are excellent but optimized for different use cases:

┌─────────────────────────────────────────────────────────────────┐
│                 LANGCHAIN vs LLAMAINDEX                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LANGCHAIN                                                      │
│  ─────────                                                      │
│  Philosophy: Swiss army knife for LLM apps                      │
│                                                                 │
│  Strengths:                                                     │
│  + Broad ecosystem (tools, agents, chains)                      │
│  + Flexible composition (LCEL)                                  │
│  + Many integrations                                            │
│  + Active community, frequent updates                           │
│  + Good for: Agents, complex workflows                          │
│                                                                 │
│  Weaknesses:                                                    │
│  - Can be heavy/complex                                         │
│  - Debugging can be tricky                                      │
│  - Rapid API changes                                            │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LLAMAINDEX                                                     │
│  ──────────                                                     │
│  Philosophy: Specialized for data/retrieval                     │
│                                                                 │
│  Strengths:                                                     │
│  + Best-in-class indexing strategies                            │
│  + Document management focus                                    │
│  + Simpler API for RAG                                          │
│  + Better structured data handling                              │
│  + Good for: Document QA, knowledge bases                       │
│                                                                 │
│  Weaknesses:                                                    │
│  - Less flexible for agents                                     │
│  - Fewer integrations                                           │
│  - Smaller community                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

DECISION FRAMEWORK:

Use LangChain when:
- Building agents with tools
- Need flexible chain composition
- Want broad LLM integrations
- Complex multi-step workflows

Use LlamaIndex when:
- Primary task is document QA
- Need sophisticated indexing
- Working with structured data
- Simpler RAG without agents

Use both together:
- LlamaIndex for retrieval/indexing
- LangChain for agent orchestration

EXAMPLE COMBINATION:

from llama_index import VectorStoreIndex
from langchain.agents import create_react_agent
from langchain.tools import Tool

# LlamaIndex for document indexing
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine()

# Wrap as LangChain tool
doc_tool = Tool(
    name='document_search',
    func=query_engine.query,
    description='Search internal documents'
)

# LangChain for agent
agent = create_react_agent(llm, [doc_tool], prompt)"
```

---

### Q4: How do you choose chunking strategies?

**VP Answer:**
```
"Chunking significantly impacts RAG quality. My approach:

CHUNKING STRATEGIES:

┌─────────────────────────────────────────────────────────────────┐
│                    CHUNKING METHODS                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. FIXED SIZE                                                  │
│  ──────────────                                                 │
│  text_splitter = RecursiveCharacterTextSplitter(                │
│      chunk_size=500,                                            │
│      chunk_overlap=50                                           │
│  )                                                              │
│                                                                 │
│  Pros: Simple, predictable size                                 │
│  Cons: May split mid-sentence/thought                           │
│  Use when: Fast implementation, uniform documents               │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  2. SENTENCE-BASED                                              │
│  ─────────────────                                              │
│  from langchain.text_splitter import SentenceTransformersTokenTextSplitter│
│                                                                 │
│  Pros: Natural boundaries, complete thoughts                    │
│  Cons: Variable chunk sizes                                     │
│  Use when: Narrative content, need coherence                    │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  3. SEMANTIC CHUNKING                                           │
│  ────────────────────                                           │
│  # Split when embedding similarity drops                        │
│  semantic_splitter = SemanticChunker(embeddings)                │
│                                                                 │
│  Pros: Topic-coherent chunks                                    │
│  Cons: Slower, requires embeddings at index time                │
│  Use when: Quality > speed, topic boundaries matter             │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  4. DOCUMENT-STRUCTURE AWARE                                    │
│  ─────────────────────────────                                  │
│  # Respect headers, sections, paragraphs                        │
│  markdown_splitter = MarkdownHeaderTextSplitter(                │
│      headers_to_split_on=[('#', 'h1'), ('##', 'h2')]            │
│  )                                                              │
│                                                                 │
│  Pros: Preserves document structure                             │
│  Cons: Requires structured input                                │
│  Use when: Structured docs (policies, documentation)            │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  5. HIERARCHICAL/PARENT-CHILD                                   │
│  ────────────────────────────                                   │
│  # Small chunks for retrieval, parent for context               │
│  parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)│
│  child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)│
│                                                                 │
│  Pros: Precise retrieval + full context                         │
│  Cons: More complex, more storage                               │
│  Use when: Need both precision and context                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

MY DECISION PROCESS:

1. Analyze document type
   - Policies → Structure-aware
   - FAQs → Sentence-based
   - Technical docs → Fixed with hierarchy

2. Determine chunk size
   - Context window budget
   - Want 5-10 chunks in prompt
   - Start with 500 tokens, adjust based on testing

3. Set overlap
   - 10-20% of chunk size
   - Prevents losing boundary information

4. Test and iterate
   - Evaluate retrieval quality
   - Check if relevant info is retrievable

PRODUCTION CONFIG:

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=tiktoken_len,  # Accurate token counting
    separators=['\n\n', '\n', '. ', ' ', '']  # Hierarchy of splits
)"
```

---

### Q5: Why LangGraph over LangChain agents?

**VP Answer:**
```
"LangGraph provides deterministic, stateful workflows vs LangChain's reactive agents.

┌─────────────────────────────────────────────────────────────────┐
│              LANGCHAIN AGENTS vs LANGGRAPH                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LANGCHAIN AGENTS                                               │
│  ────────────────                                               │
│  Flow: Query → Think → Act → Observe → Think → ...             │
│                                                                 │
│  agent = create_react_agent(llm, tools, prompt)                 │
│  executor = AgentExecutor(agent=agent, tools=tools)             │
│  result = executor.invoke({'input': 'query'})                   │
│                                                                 │
│  Pros:                                                          │
│  + Simple to set up                                             │
│  + Flexible, LLM decides flow                                   │
│                                                                 │
│  Cons:                                                          │
│  - Non-deterministic flow                                       │
│  - Hard to debug/audit                                          │
│  - Can get stuck in loops                                       │
│  - Difficult to add checkpoints                                 │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LANGGRAPH                                                      │
│  ─────────                                                      │
│  Flow: Explicit graph with nodes and edges                      │
│                                                                 │
│  from langgraph.graph import StateGraph, END                    │
│                                                                 │
│  graph = StateGraph(AgentState)                                 │
│  graph.add_node('retrieve', retrieve_docs)                      │
│  graph.add_node('generate', generate_response)                  │
│  graph.add_node('validate', validate_output)                    │
│  graph.add_edge('retrieve', 'generate')                         │
│  graph.add_conditional_edges('generate', route_fn)              │
│                                                                 │
│  Pros:                                                          │
│  + Deterministic, explicit flow                                 │
│  + Easy to debug (follow the graph)                             │
│  + Built-in state management                                    │
│  + Checkpointing/persistence                                    │
│  + Human-in-the-loop support                                    │
│                                                                 │
│  Cons:                                                          │
│  - More setup code                                              │
│  - Less flexible (you define the flow)                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

WHEN TO USE LANGGRAPH:

1. Production systems (need reliability)
2. Complex multi-step workflows
3. Need audit trail
4. Human approval steps required
5. Stateful conversations
6. Error recovery needed

EXAMPLE LANGGRAPH FLOW:

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END

class State(TypedDict):
    query: str
    documents: list
    response: str
    confidence: float

def retrieve(state: State) -> State:
    docs = retriever.invoke(state['query'])
    return {'documents': docs}

def generate(state: State) -> State:
    response = llm.invoke(format_prompt(state))
    return {'response': response.content, 'confidence': response.confidence}

def should_review(state: State) -> str:
    if state['confidence'] < 0.8:
        return 'human_review'
    return 'output'

# Build graph
workflow = StateGraph(State)
workflow.add_node('retrieve', retrieve)
workflow.add_node('generate', generate)
workflow.add_node('human_review', human_review)
workflow.add_node('output', output)

workflow.set_entry_point('retrieve')
workflow.add_edge('retrieve', 'generate')
workflow.add_conditional_edges('generate', should_review)
workflow.add_edge('human_review', 'output')
workflow.add_edge('output', END)

app = workflow.compile()"
```

---

### Q6: How do you evaluate RAG quality?

**VP Answer:**
```
"RAG evaluation requires measuring both retrieval and generation quality:

┌─────────────────────────────────────────────────────────────────┐
│                   RAG EVALUATION FRAMEWORK                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  RETRIEVAL METRICS                                              │
│  ═════════════════                                              │
│                                                                 │
│  1. Recall@K                                                    │
│     'Of relevant docs, how many did we retrieve?'               │
│     recall@5 = relevant_in_top5 / total_relevant                │
│                                                                 │
│  2. Precision@K                                                 │
│     'Of retrieved docs, how many are relevant?'                 │
│     precision@5 = relevant_in_top5 / 5                          │
│                                                                 │
│  3. MRR (Mean Reciprocal Rank)                                  │
│     'How high is the first relevant doc?'                       │
│     MRR = mean(1/rank_of_first_relevant)                        │
│                                                                 │
│  4. NDCG                                                        │
│     'Are highly relevant docs ranked higher?'                   │
│     Considers graded relevance                                  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  GENERATION METRICS                                             │
│  ═══════════════════                                            │
│                                                                 │
│  1. Faithfulness/Groundedness                                   │
│     'Is response supported by retrieved docs?'                  │
│                                                                 │
│  2. Answer Relevance                                            │
│     'Does response actually answer the question?'               │
│                                                                 │
│  3. Context Relevance                                           │
│     'Is retrieved context relevant to query?'                   │
│                                                                 │
│  4. Harmfulness                                                 │
│     'Does response contain harmful content?'                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

EVALUATION APPROACHES:

1. RAGAS FRAMEWORK (Recommended)

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)

2. LLM-AS-JUDGE

def evaluate_faithfulness(response, context):
    prompt = f'''
    Given the following context and response, rate if the response
    is fully supported by the context.

    Context: {context}
    Response: {response}

    Rating (1-5):
    Explanation:
    '''
    return llm.invoke(prompt)

3. HUMAN EVALUATION (Gold Standard)

- Sample queries periodically
- Have experts rate responses
- Track quality over time

MY PRODUCTION SETUP:

class RAGEvaluator:
    def __init__(self):
        self.metrics = {
            'retrieval': ['recall@5', 'mrr'],
            'generation': ['faithfulness', 'relevance'],
            'system': ['latency', 'cost']
        }

    def evaluate_batch(self, queries, ground_truth):
        # Retrieval metrics
        retrieval_scores = self.eval_retrieval(queries, ground_truth)

        # Generation metrics (LLM-as-judge)
        generation_scores = self.eval_generation(queries, ground_truth)

        return {**retrieval_scores, **generation_scores}

    def monitor_production(self, sample_rate=0.01):
        # Sample 1% of queries for continuous evaluation
        # Track metrics over time, alert on degradation"
```

---

### Q7: Fine-tuning vs RAG - how do you decide?

**VP Answer:**
```
"This is one of the most important decisions in GenAI projects:

┌─────────────────────────────────────────────────────────────────┐
│                    FINE-TUNING vs RAG                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Aspect              RAG                   Fine-tuning          │
│  ────────────────────────────────────────────────────────────── │
│  Data freshness      Real-time updates     Requires retraining  │
│  Setup cost          Lower                 Higher (compute)     │
│  Hallucination       Reduced (grounded)    Still possible       │
│  Traceability        High (cite sources)   Low (in weights)     │
│  Domain adaptation   Via retrieval         Via training         │
│  Customization       Prompt + retrieval    Model behavior       │
│  Latency             Higher (retrieval)    Lower                │
│  Context length      Limited               N/A                  │
│  Maintenance         Update vector store   Retrain periodically │
└─────────────────────────────────────────────────────────────────┘

DECISION FRAMEWORK:

Use RAG when:
- Data changes frequently (policies, docs)
- Need source attribution
- Regulatory/audit requirements
- Limited training data
- Want quick iteration

Use Fine-tuning when:
- Need consistent style/tone
- Domain-specific language
- Task-specific behavior
- Have lots of training data
- Performance-critical (lower latency)

Use BOTH when:
- Fine-tune for domain language
- RAG for factual grounding
- Best of both worlds

PRACTICAL EXAMPLES:

1. Customer support FAQ
   → RAG (answers need to be traceable to source)

2. Code completion in specific framework
   → Fine-tuning (learn patterns and style)

3. Legal document analysis
   → RAG (need exact citations)

4. Brand voice assistant
   → Fine-tuning (consistent personality)
   → RAG (product knowledge)

MY BANKING EXPERIENCE:

For policy Q&A:
→ RAG only (auditability required)

For document summarization:
→ Fine-tuned model (specific format)
→ RAG for fact-checking

Decision tree I use:

if need_citations or data_changes_frequently:
    use_rag()
elif need_consistent_style and have_training_data:
    use_finetuning()
elif both_apply:
    finetune_base_then_add_rag()
else:
    start_with_rag()  # Faster to iterate"
```

---

### Q8: How do you handle conversation memory in LLM applications?

**VP Answer:**
```
"Memory management is crucial for coherent multi-turn conversations:

MEMORY TYPES:

┌─────────────────────────────────────────────────────────────────┐
│                    MEMORY STRATEGIES                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. BUFFER MEMORY (Full History)                                │
│  ────────────────────────────────                               │
│  from langchain.memory import ConversationBufferMemory          │
│                                                                 │
│  memory = ConversationBufferMemory()                            │
│                                                                 │
│  Pros: Complete context                                         │
│  Cons: Token limit hit quickly                                  │
│  Use: Short conversations                                       │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  2. WINDOW MEMORY (Last K Turns)                                │
│  ───────────────────────────────                                │
│  memory = ConversationBufferWindowMemory(k=10)                  │
│                                                                 │
│  Pros: Bounded size, recent context                             │
│  Cons: Loses early conversation                                 │
│  Use: Task-focused conversations                                │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  3. SUMMARY MEMORY                                              │
│  ────────────────                                               │
│  memory = ConversationSummaryMemory(llm=llm)                    │
│                                                                 │
│  Keeps running summary instead of full history                  │
│                                                                 │
│  Pros: Unbounded conversations                                  │
│  Cons: May lose details, extra LLM calls                        │
│  Use: Long conversations                                        │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  4. ENTITY MEMORY                                               │
│  ──────────────                                                 │
│  memory = ConversationEntityMemory(llm=llm)                     │
│                                                                 │
│  Extracts and tracks entities mentioned                         │
│                                                                 │
│  Pros: Tracks important info                                    │
│  Cons: May miss context                                         │
│  Use: Customer service, CRM integration                         │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  5. VECTOR STORE MEMORY                                         │
│  ──────────────────────                                         │
│  Embed and store all messages, retrieve relevant ones           │
│                                                                 │
│  Pros: Retrieves relevant history                               │
│  Cons: May miss sequential context                              │
│  Use: Long-term memory across sessions                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

MY PRODUCTION APPROACH:

class HybridMemory:
    def __init__(self, window_size=5, max_tokens=2000):
        self.window = deque(maxlen=window_size)  # Recent context
        self.summary = ''  # Running summary
        self.entities = {}  # Key entities
        self.max_tokens = max_tokens

    def add_message(self, role, content):
        self.window.append({'role': role, 'content': content})

        # Update summary if window is full
        if len(self.window) == self.window.maxlen:
            self._update_summary()

        # Extract entities
        self._extract_entities(content)

    def get_context(self):
        context = []

        # Add summary if exists
        if self.summary:
            context.append({'role': 'system', 'content': f'Summary: {self.summary}'})

        # Add entity context
        if self.entities:
            context.append({'role': 'system', 'content': f'Entities: {self.entities}'})

        # Add recent messages
        context.extend(list(self.window))

        return context

PERSISTENCE:

# Store in Redis for multi-session
import redis

class PersistentMemory:
    def __init__(self, session_id):
        self.redis = redis.Redis()
        self.session_id = session_id

    def save(self, memory):
        self.redis.set(f'memory:{self.session_id}', json.dumps(memory))
        self.redis.expire(f'memory:{self.session_id}', 3600)  # 1 hour TTL

    def load(self):
        data = self.redis.get(f'memory:{self.session_id}')
        return json.loads(data) if data else []"
```

---

### Q9: How do you implement semantic caching for LLM calls?

**VP Answer:**
```
"Semantic caching reduces costs and latency by reusing similar queries:

CONCEPT:

Traditional Cache: Exact match only
'What is the capital of France?' → cache hit
'What's the capital of France?' → cache miss (different string)

Semantic Cache: Similarity-based matching
Both queries → same cached response (semantically similar)

IMPLEMENTATION:

┌─────────────────────────────────────────────────────────────────┐
│                  SEMANTIC CACHE ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Query → Embed → Search Cache (similarity) → Hit/Miss           │
│                       │                                         │
│                       ├── Hit (sim > 0.95): Return cached       │
│                       └── Miss: Call LLM → Store result         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

class SemanticCache:
    def __init__(self, similarity_threshold=0.95):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = FAISS.from_texts(['init'], self.embeddings)
        self.cache = {}  # query_hash -> response
        self.threshold = similarity_threshold

    def get(self, query):
        # Embed query
        query_embedding = self.embeddings.embed_query(query)

        # Search for similar cached queries
        results = self.vector_store.similarity_search_with_score(
            query, k=1
        )

        if results:
            doc, score = results[0]
            if score > self.threshold:
                query_hash = doc.metadata['hash']
                return self.cache.get(query_hash)

        return None

    def set(self, query, response):
        # Store embedding
        query_hash = hashlib.md5(query.encode()).hexdigest()
        self.vector_store.add_texts(
            [query],
            metadatas=[{'hash': query_hash}]
        )

        # Store response
        self.cache[query_hash] = response

PRODUCTION CONSIDERATIONS:

1. TTL (Time-to-Live)
   - Cache expiration based on data freshness
   - Policy docs: 24 hours
   - Static knowledge: 7 days

2. Cache Invalidation
   - When source documents change
   - Version tracking

3. Threshold Tuning
   - Too high (0.99): Few hits
   - Too low (0.8): Wrong responses
   - Start at 0.95, adjust based on testing

4. Multi-level Cache
   - L1: Exact match (Redis string)
   - L2: Semantic match (Vector DB)

COST SAVINGS:

Before caching:
- 10,000 queries/day × $0.03 = $300/day

After caching (60% hit rate):
- 4,000 LLM calls × $0.03 = $120/day
- Savings: 60% cost reduction"
```

---

### Q10: How do you handle rate limits and retries with LLM APIs?

**VP Answer:**
```
"Robust error handling is essential for production LLM applications:

COMMON RATE LIMIT ERRORS:

OpenAI: RateLimitError (429)
- Tokens per minute (TPM)
- Requests per minute (RPM)
- Tokens per day (TPD)

RETRY STRATEGY:

import time
import random
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIError))
)
def call_llm(prompt):
    return client.chat.completions.create(
        model='gpt-4',
        messages=[{'role': 'user', 'content': prompt}]
    )

ADVANCED RATE LIMITING:

from ratelimit import limits, sleep_and_retry

class RateLimitedLLMClient:
    def __init__(self, rpm_limit=60, tpm_limit=90000):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.token_bucket = TokenBucket(tpm_limit, 60)

    @sleep_and_retry
    @limits(calls=60, period=60)  # 60 calls per minute
    def call(self, prompt, max_tokens=1000):
        # Check token budget
        estimated_tokens = len(prompt) // 4 + max_tokens
        self.token_bucket.consume(estimated_tokens)

        return self._make_request(prompt, max_tokens)

TOKEN BUCKET IMPLEMENTATION:

class TokenBucket:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()

    def consume(self, tokens):
        self._refill()

        if tokens > self.tokens:
            wait_time = (tokens - self.tokens) / self.refill_rate
            time.sleep(wait_time)
            self._refill()

        self.tokens -= tokens

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate
        )
        self.last_refill = now

PRODUCTION ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────┐
│                    LLM REQUEST HANDLING                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Request → Queue → Rate Limiter → LLM API                       │
│              │          │             │                         │
│              │          │             ├── Success → Response    │
│              │          │             └── Error → Retry Queue   │
│              │          │                                       │
│              │          └── Over limit → Wait/Backoff           │
│              │                                                  │
│              └── Priority queue (high-value requests first)     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

FALLBACK STRATEGY:

def call_with_fallback(prompt):
    try:
        return call_gpt4(prompt)
    except RateLimitError:
        logger.warning('GPT-4 rate limited, falling back to GPT-3.5')
        return call_gpt35(prompt)
    except Exception:
        return cached_response_or_default(prompt)"
```

---

## Practice Questions

1. How do you handle PII in RAG systems?
2. Compare different reranking strategies
3. How do you debug a poorly performing RAG system?
4. What's your approach to prompt versioning?
5. How do you handle multi-language RAG?
