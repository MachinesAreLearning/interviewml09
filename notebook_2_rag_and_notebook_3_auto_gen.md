# Notebook 2 — RAG Application (Document Retrieval + Generation)

> **Goal:** A runnable, interview-focused Colab notebook that demonstrates a full Retrieval-Augmented Generation (RAG) pipeline: document ingestion, embedding, vector store indexing, retrieval, and generation. Includes evaluation techniques, failure modes (hallucinations), and interview notes.

---

## 0. Setup & Install

```bash
# Run in Colab
!pip install -q sentence-transformers faiss-cpu transformers accelerate datasets tqdm scikit-learn langchain
# Optional: if you prefer Chroma or Milvus
# !pip install -q chromadb

# If you plan to use OpenAI for generation, install openai
# !pip install -q openai
```

> **Notebook note:** Use GPU runtime for faster embedding/generation when using transformer-based generators.

---

## 1. Introduction (Markdown)

- What is RAG?  
  Retrieval-Augmented Generation (RAG) augments a generative model's context with documents retrieved from a knowledge base. This helps grounding, factuality, and domain adaptation without heavy fine-tuning.

- When to use RAG: domain-specific Q&A, knowledge bases, long-context summarization, and reducing hallucinations.

- High-level pipeline:  
  1. Document collection & chunking  
  2. Text cleaning & preprocessing  
  3. Compute embeddings (document chunks)  
  4. Index embeddings in vector DB (FAISS / Chroma / Milvus)  
  5. For an input query: retrieve top-k docs  
  6. Construct prompt (context + instruction) and call a generator  
  7. Post-process & evaluate

---

## 2. Example Dataset

Use a small corpus for demonstration. Options: a Wikipedia subset, a collection of research abstracts, or a curated set of product manuals. For Colab quickstart we'll use `datasets` or a small local list.

```python
# small example corpus
docs = [
    "Python is a high-level, interpreted programming language...",
    "FAISS is a library for efficient similarity search and clustering of dense vectors...",
    "RAG pipelines combine retrieval with generation to reduce hallucinations...",
    # add more realistic docs for production
]
```

---

## 3. Chunking & Preprocessing

```python
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')

# naive chunker (by sentences)
def chunk_text(text, max_sentences=5):
    sents = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sents), max_sentences):
        chunks.append(" ".join(sents[i:i+max_sentences]))
    return chunks

# apply to docs
doc_chunks = []
for i, d in enumerate(docs):
    for c in chunk_text(d, max_sentences=3):
        doc_chunks.append({"id": f"doc_{i}", "text": c})

len(doc_chunks), doc_chunks[:2]
```

**Markdown notes:** chunk size matters — too large → noisy context; too small → retrieval may lose coherence. Keep overlapping windows (stride) to preserve context across chunk boundaries.

---

## 4. Embeddings

- Use `sentence-transformers` for embeddings (e.g., `all-MiniLM-L6-v2` for speed or `all-mpnet-base-v2` for quality).

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

texts = [c['text'] for c in doc_chunks]
embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

# store in-memory as example
import numpy as np
emb_matrix = np.vstack(embs)
```

**Interview note:** Discuss trade-offs: compute cost, embedding dimensionality, semantic vs lexical similarity.

---

## 5. Vector Index (FAISS simple)

```python
import faiss
d = emb_matrix.shape[1]
index = faiss.IndexFlatL2(d)  # simple index
index.add(emb_matrix)

# mapping from index -> chunk
id_to_chunk = {i: doc_chunks[i] for i in range(len(doc_chunks))}
```

**Production note:** For large corpora, use `IndexIVFFlat` + quantization, persistent stores (ChromaDB, Milvus), and ANN indexes (HNSW) to scale.

---

## 6. Retrieval

```python
def retrieve(query, k=3):
    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    results = [id_to_chunk[i] for i in I[0]]
    return results

retrieve("What is FAISS used for?", k=2)
```

**Markdown:** Show retrieved snippets and discuss precision@k, recall@k.

---

## 7. Prompt Construction & Generation

Two options for generation: (A) Use OpenAI / hosted LLM via API (Cheaper/faster to prototype) or (B) Use local transformer generation (e.g., `facebook/opt`, `bigscience/bloom`, or `meta-llama` derivatives)

**A. OpenAI-style prompt (pseudo):**

```python
# pseudo-code for prompt
query = "How does FAISS work?"
retrieved = retrieve(query, k=4)
context = "\n\n".join([r['text'] for r in retrieved])
prompt = f"Use the context below to answer the question. If the answer is not in the context, say you don't know.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nAnswer concisely."

# send prompt to generator (OpenAI / HF)
# response = openai.ChatCompletion.create(..., prompt=prompt)
```

**B. Local transformer generation (HF)**

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = 'gpt2'  # demo only; replace with larger if available
tokenizer = AutoTokenizer.from_pretrained(model_name)
lm = AutoModelForCausalLM.from_pretrained(model_name)
gen = pipeline('text-generation', model=lm, tokenizer=tokenizer, device=0)

input_text = prompt
out = gen(input_text, max_length=256, do_sample=False)
print(out[0]['generated_text'])
```

**Interview note:** Emphasize safety instruction: tell the model to refuse when no evidence in context and to quote sources (chunk ids).

---

## 8. Post-processing & Attribution

- Include top-k chunk ids as citations in output.  
- If generator hallucinates, link output sentences back to supporting chunks (simple string matching or token overlap).

```python
# naive attribution
answer = "..."  # from generator
supporting = []
for idx, c in enumerate(texts):
    if any(tok in answer for tok in c.split()[:8]):
        supporting.append(idx)

supporting
```

---

## 9. Evaluation Techniques (RAG)

**Retrieval Metrics:**  
- Precision@k: fraction of retrieved docs that are relevant.  
- Recall@k: fraction of relevant docs retrieved.  

**Generation Metrics:**  
- BLEU / ROUGE / METEOR: token overlap style metrics (useful for summarization/paraphrase tasks).  
- BERTScore / MoverScore: semantic similarity.  
- Exact Match / F1: for QA-style outputs.

**RAG-specific:**  
- Grounding score: fraction of generated statements with a supporting document.  
- Hallucination rate: fraction of assertions with no supporting doc.

**Human eval:**  
- Correctness, Fluency, Usefulness, Hallucination (binary), Attribution accuracy.

---

## 10. Failure Modes & Mitigations

- Hallucinations → stronger retrieval + conservative prompts + verification step
- Stale data → re-indexing cadence
- Context window overflow → use condensed retrieval (RAG+refine) or use multi-pass retrieval
- Privacy leakage → PII redaction before indexing

---

## 11. Scaling & Production Notes

- Use persistent vector DB (Chroma, Milvus, Pinecone) with replication and backups.  
- Precompute embeddings offline and update incrementally.  
- Add a reranker (cross-encoder) to re-rank top-k from approximate index for higher precision.  

---

## 12. Interview Q&A Snippets (Markdown)

- Q: Why not just fine-tune a model?  
  A: Fine-tuning is costly and brittle; RAG provides on-the-fly domain adaptation and cheaper updates.

- Q: How measure hallucinations?  
  A: Combine automatic metrics (grounding score) and human annotations; use coverage heuristics and factuality checks.


---

# Notebook 3 — Agentic AI with AutoGen (Multi-Agent Collaboration Example)

> **Goal:** A runnable, interview-focused Colab notebook demonstrating a small multi-agent system using AutoGen-style orchestration. Show design patterns (planner, executor, critic), message flow, and evaluation for task completion.

---

## 0. Setup & Install

```bash
# Run in Colab
!pip install -q autogen-client  # placeholder name — replace with actual package name if different
!pip install -q openai langchain
```

> **Note:** If `autogen` library name differs, replace with the appropriate package. The notebook will explain the conceptual flow so you can adapt code to the specific AutoGen API.

---

## 1. Introduction

- What is agentic AI?  
  Agentic AI composes multiple specialized agents (planner, tool-user, critic, knowledge-agent) that communicate to accomplish complex tasks. Agents exchange messages, propose actions, and critique outputs.

- Use cases: multi-step workflows, tool orchestration, data extraction pipelines, automated research assistants.

---

## 2. Agent Design Patterns (Markdown)

- **Planner**: Breaks a high-level task into subtasks.  
- **Executor**: Performs subtasks (calls tools, runs code, retrieves docs).  
- **Critic / Verifier**: Evaluates outputs, requests refinements.  
- **Tool Agents**: Wrap external services (search, calculator, database).

Include a small diagram (ASCII) of message flow.

```
User -> Planner -> Executor -> Tool(s)
                   \-> Critic -> Planner
```

---

## 3. Minimal Multi-Agent Example (Pseudo/Practical Code)

```python
# This example uses a simple in-memory message passing pattern.
# Replace `SimpleLLM` with your chosen LLM client (OpenAI/HF).

class SimpleLLM:
    def __init__(self):
        pass
    def generate(self, prompt):
        # placeholder for actual LLM call
        return "generated text based on: " + prompt[:200]

llm = SimpleLLM()

# Agents
class Planner:
    def __init__(self, llm):
        self.llm = llm
    def plan(self, task):
        p = f"You are a planner. Break down this task into steps: {task}"
        return self.llm.generate(p)

class Executor:
    def __init__(self, llm):
        self.llm = llm
    def execute(self, step, tools=None):
        p = f"Execute this step: {step}. Use tools: {tools}"
        return self.llm.generate(p)

class Critic:
    def __init__(self, llm):
        self.llm = llm
    def critique(self, result):
        p = f"Evaluate this result for correctness and completeness: {result}"
        return self.llm.generate(p)

# Orchestration
planner = Planner(llm)
executor = Executor(llm)
critic = Critic(llm)

task = "Summarize the key limitations of FAISS and propose mitigation steps"
plan = planner.plan(task)
print('PLAN:\n', plan)

# naive split of plan into steps (in production parse LLM structured output)
steps = ["Find limitations", "Propose mitigations"]
for s in steps:
    result = executor.execute(s, tools=['web_search', 'docs'])
    review = critic.critique(result)
    print('STEP RESULT:\n', result)
    print('CRITIC REVIEW:\n', review)
```

**Markdown notes:** In real AutoGen, agents often have structured roles and system prompts that shape behavior. Use safety checks to avoid harmful actions.

---

## 4. Tool Integration

- Tools can be HTTP APIs, search, database queries, calculators. Wrap each as a ToolAgent with a stable interface: `call_tool(tool_name, input)` that returns structured JSON.

```python
# Example: simple tool wrapper
def web_search(query):
    # call SerpAPI / Google programmable search or local bing wrapper
    return ["url1: summary...", "url2: summary..."]

# Executor uses web_search when the step mentions 'search'
```

---

## 5. Safety & Governance

- Always sandbox tool calls, validate outputs before executing actions (e.g., do not run arbitrary system commands returned by LLMs).  
- Add human-in-the-loop checkpoints for high-risk tasks.  
- Log all agent messages for audit and reproducibility.

---

## 6. Evaluation for Agentic Systems

**Agentic metrics:**
- Task completion rate  
- Number of iterations to converge  
- Average latency per iteration  
- Critic accuracy (how often critic finds errors)  
- Cost per successful task (API calls, compute)

**Behavioral tests:**
- Robustness to noisy instructions  
- Safety tests (malicious instruction adversarial prompts)  
- Reproducibility checks

---

## 7. Advanced Patterns & Best Practices

- **Planner-Executor-Critic loop**: iterate until critic approves or max iterations reached.  
- **Tool-use heuristics**: prefer deterministic tools for factual lookups, LLMs for summarization.  
- **State management**: persist intermediate outputs and decisions in a structured store (DB).  
- **RAG + Agents**: Use RAG to ground agents when they need facts; agents fetch docs, then summarize/act.

---

## 8. Hands-on Integration Example (RAG + Agents)

High-level sequence:  
1. User asks question  
2. Planner asks Retriever to fetch context (RAG)  
3. Executor generates an answer using retrieved context  
4. Critic verifies answer against retrieved context  

```python
# Pseudocode combining retrieval and agent loop
query = "What are best practices for scaling FAISS in production?"
retrieved = retrieve(query, k=5)  # from Notebook 2
context = "\n\n".join([r['text'] for r in retrieved])
planner_prompt = f"Create a short plan to answer: {query} using context: {context}"
plan = planner.llm.generate(planner_prompt)
step = "Draft answer using context"
answer = executor.llm.generate(f"{step}\nCONTEXT:\n{context}")
crit = critic.llm.generate(f"Is this answer supported by the context?\nAnswer:\n{answer}\nContext:\n{context}")
print('ANSWER:\n', answer)
print('CRITIC:\n', crit)
```

---

## 9. Interview Q&A Snippets (Markdown)

- Q: How would you design a multi-agent system for a legal research assistant?  
  A: Use a Planner to identify legal questions, a Retriever + RAG for statutes/cases, Executors to summarize and cite, and a Critic that cross-checks citations and flags inconsistencies.

- Q: How to prevent an agent from executing harmful code?  
  A: Use a whitelist for allowed actions, sandboxed execution, require human approval for high-risk steps.

---

## 10. Next Steps & References

- Convert pseudocode to real API calls (OpenAI / Hugging Face / AutoGen SDK).  
- Add unit tests for agent interactions & integration tests for tool calls.  
- Build dashboards to inspect message flows, critic decisions, and task outcomes.


---


**End of Combined Notebooks (Notebook 2 + Notebook 3)**




