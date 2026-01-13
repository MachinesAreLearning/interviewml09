# Retrieval-Augmented Generation (RAG) Interview Preparation Notebook

## 1. Introduction & Concept

RAG combines the knowledge retrieval capabilities of information retrieval systems with the generative power of Large Language Models (LLMs) to produce accurate, grounded, and contextually relevant responses.

### Mathematical Foundation

**RAG Probability Formulation:**
$$P(y|x) = \sum_{z \in Z} P(y|x, z) \cdot P(z|x)$$

Where:
- $x$ = input query
- $y$ = generated output
- $z$ = retrieved documents
- $P(z|x)$ = retriever probability
- $P(y|x, z)$ = generator probability given context

**Retrieval Scoring (BM25):**
$$\text{BM25}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})}$$

**Dense Retrieval (Cosine Similarity):**
$$\text{sim}(q, d) = \frac{E_q \cdot E_d}{||E_q|| \cdot ||E_d||}$$

Where $E_q$ and $E_d$ are query and document embeddings

### RAG Architecture Comparison

| Component | Naive RAG | Advanced RAG | Modular RAG | Agentic RAG |
|-----------|-----------|--------------|-------------|-------------|
| **Retrieval** | Single-pass, fixed k-docs | Multi-hop, dynamic retrieval | Pluggable retrievers | Agent-orchestrated |
| **Indexing** | Basic chunking | Hierarchical, semantic | Multi-index strategies | Adaptive indexing |
| **Generation** | Simple prompt + context | Chain-of-thought, self-consistency | Task-specific prompts | Multi-agent collaboration |
| **Evaluation** | Basic accuracy | RAGAS metrics | Component-wise eval | End-to-end + behavioral |
| **Use Cases** | Simple Q&A | Complex reasoning | Enterprise systems | Autonomous workflows |

```python
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS, Pinecone
from langchain.chains import RetrievalQA
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
import warnings
warnings.filterwarnings('ignore')

# Set up logging for production monitoring
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
```

## 2. Data Preparation for RAG

### 2.1 Document Processing Pipeline

```python
def comprehensive_document_processor():
    """
    Production-ready document processing for RAG systems
    """
    print("DOCUMENT PROCESSING PIPELINE")
    print("="*35)
    
    pipeline_stages = """
    📄 DOCUMENT INGESTION:
    ├── Format Support: PDF, DOCX, HTML, JSON, CSV, TXT
    ├── Metadata Extraction: Title, author, date, sections
    ├── Structure Preservation: Headers, tables, lists
    └── Error Handling: Corrupt files, encoding issues
    
    🔪 CHUNKING STRATEGIES:
    ├── Fixed-size: Simple, consistent chunks
    ├── Semantic: Sentence/paragraph boundaries
    ├── Recursive: Hierarchical splitting
    ├── Document-aware: Section/chapter based
    └── Sliding Window: Overlapping for context
    
    🧹 PREPROCESSING:
    ├── Text Cleaning: Remove artifacts, normalize
    ├── Deduplication: Hash-based, semantic
    ├── Language Detection: Multi-lingual support
    └── PII Redaction: Compliance requirements
    
    📊 METADATA ENRICHMENT:
    ├── Document Type Classification
    ├── Entity Extraction (NER)
    ├── Timestamp Extraction
    └── Source Tracking
    """
    
    print(pipeline_stages)

def intelligent_chunking_strategy(documents, chunk_size=1000, overlap=200):
    """
    Implement multiple chunking strategies with evaluation
    """
    print("INTELLIGENT CHUNKING ANALYSIS")
    print("="*32)
    
    strategies = {
        'recursive': RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        ),
        'semantic': RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        ),
        'token_based': RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,  # Should be token counter
        )
    }
    
    chunking_results = {}
    
    for strategy_name, splitter in strategies.items():
        chunks = splitter.split_documents(documents)
        
        # Analyze chunk statistics
        chunk_lengths = [len(chunk.page_content) for chunk in chunks]
        
        chunking_results[strategy_name] = {
            'num_chunks': len(chunks),
            'avg_length': np.mean(chunk_lengths),
            'std_length': np.std(chunk_lengths),
            'min_length': min(chunk_lengths),
            'max_length': max(chunk_lengths),
            'chunks': chunks
        }
        
        print(f"\n{strategy_name.upper()} Strategy:")
        print(f"  Chunks: {chunking_results[strategy_name]['num_chunks']}")
        print(f"  Avg Length: {chunking_results[strategy_name]['avg_length']:.0f}")
        print(f"  Std Dev: {chunking_results[strategy_name]['std_length']:.0f}")
    
    return chunking_results

def metadata_extraction_pipeline(documents):
    """
    Extract and enrich metadata for improved retrieval
    """
    print("\nMETADATA EXTRACTION")
    print("="*20)
    
    enriched_docs = []
    
    for doc in documents:
        metadata = {
            'source': doc.metadata.get('source', 'unknown'),
            'page': doc.metadata.get('page', 0),
            'chunk_id': hash(doc.page_content),
            'word_count': len(doc.page_content.split()),
            'char_count': len(doc.page_content),
            'has_code': '```' in doc.page_content or 'def ' in doc.page_content,
            'has_table': '|' in doc.page_content and '\n|' in doc.page_content,
            'language': 'en',  # Would use langdetect in production
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        doc.metadata.update(metadata)
        enriched_docs.append(doc)
    
    print(f"Enriched {len(enriched_docs)} documents with metadata")
    print(f"Metadata fields: {list(enriched_docs[0].metadata.keys())}")
    
    return enriched_docs
```

### 2.2 Embedding Strategies

```python
def embedding_model_comparison():
    """
    Compare different embedding models for RAG
    """
    print("EMBEDDING MODEL COMPARISON")
    print("="*30)
    
    models = {
        'sentence-transformers/all-MiniLM-L6-v2': {
            'dim': 384,
            'max_tokens': 512,
            'speed': 'Fast',
            'quality': 'Good',
            'size': '80MB'
        },
        'sentence-transformers/all-mpnet-base-v2': {
            'dim': 768,
            'max_tokens': 512,
            'speed': 'Medium',
            'quality': 'Better',
            'size': '420MB'
        },
        'BAAI/bge-large-en-v1.5': {
            'dim': 1024,
            'max_tokens': 512,
            'speed': 'Slow',
            'quality': 'Best',
            'size': '1.3GB'
        },
        'text-embedding-ada-002': {
            'dim': 1536,
            'max_tokens': 8191,
            'speed': 'API',
            'quality': 'Excellent',
            'size': 'Cloud'
        }
    }
    
    print("Model Comparison:")
    comparison_df = pd.DataFrame(models).T
    print(comparison_df)
    
    return models

def hybrid_search_implementation():
    """
    Implement hybrid search combining dense and sparse retrieval
    """
    print("\nHYBRID SEARCH ARCHITECTURE")
    print("="*28)
    
    architecture = """
    🔍 HYBRID RETRIEVAL:
    
    DENSE RETRIEVAL (Semantic):
    ├── Embedding Model: BERT/Sentence-Transformers
    ├── Vector Store: FAISS/ChromaDB/Pinecone
    ├── Similarity: Cosine/Dot Product
    └── Strengths: Semantic understanding
    
    SPARSE RETRIEVAL (Keyword):
    ├── Algorithm: BM25/TF-IDF
    ├── Index: Elasticsearch/Lucene
    ├── Matching: Exact/Fuzzy
    └── Strengths: Precise term matching
    
    FUSION STRATEGIES:
    ├── Reciprocal Rank Fusion (RRF)
    ├── Linear Combination: α·dense + (1-α)·sparse
    ├── Learned Ranker: Cross-encoder reranking
    └── Round-robin Interleaving
    """
    
    print(architecture)
    
    # Implementation example
    class HybridRetriever:
        def __init__(self, dense_retriever, sparse_retriever, alpha=0.5):
            self.dense_retriever = dense_retriever
            self.sparse_retriever = sparse_retriever
            self.alpha = alpha
        
        def retrieve(self, query, k=10):
            # Dense retrieval
            dense_results = self.dense_retriever.similarity_search(query, k=k*2)
            dense_scores = {doc.metadata['id']: score 
                           for doc, score in dense_results}
            
            # Sparse retrieval
            sparse_results = self.sparse_retriever.search(query, k=k*2)
            sparse_scores = {doc.metadata['id']: score 
                           for doc, score in sparse_results}
            
            # Combine scores
            all_doc_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
            
            combined_scores = {}
            for doc_id in all_doc_ids:
                dense_score = dense_scores.get(doc_id, 0)
                sparse_score = sparse_scores.get(doc_id, 0)
                combined_scores[doc_id] = (
                    self.alpha * dense_score + 
                    (1 - self.alpha) * sparse_score
                )
            
            # Sort and return top k
            sorted_docs = sorted(combined_scores.items(), 
                               key=lambda x: x[1], 
                               reverse=True)[:k]
            
            return sorted_docs
    
    return HybridRetriever
```

## 3. RAG System Architecture

### 3.1 Vector Database Selection

```python
def vector_database_comparison():
    """
    Compare vector databases for production RAG systems
    """
    print("VECTOR DATABASE COMPARISON")
    print("="*30)
    
    databases = {
        'ChromaDB': {
            'type': 'Embedded',
            'scalability': 'Medium',
            'features': 'Simple, local, metadata filtering',
            'latency': 'Low',
            'cost': 'Free',
            'max_vectors': '1M',
            'production_ready': 'Development'
        },
        'Pinecone': {
            'type': 'Managed Cloud',
            'scalability': 'High',
            'features': 'Managed, hybrid search, namespaces',
            'latency': 'Medium',
            'cost': '$$',
            'max_vectors': 'Billions',
            'production_ready': 'Yes'
        },
        'Weaviate': {
            'type': 'Self-hosted/Cloud',
            'scalability': 'High',
            'features': 'GraphQL, multi-modal, modules',
            'latency': 'Low',
            'cost': '$',
            'max_vectors': 'Billions',
            'production_ready': 'Yes'
        },
        'FAISS': {
            'type': 'Library',
            'scalability': 'High',
            'features': 'Fast, GPU support, many indexes',
            'latency': 'Very Low',
            'cost': 'Free',
            'max_vectors': 'Billions',
            'production_ready': 'With wrapper'
        },
        'PGVector': {
            'type': 'PostgreSQL Extension',
            'scalability': 'Medium',
            'features': 'SQL integration, ACID',
            'latency': 'Medium',
            'cost': '$',
            'max_vectors': '10M',
            'production_ready': 'Yes'
        }
    }
    
    comparison_df = pd.DataFrame(databases).T
    print(comparison_df)
    
    print("\n🎯 Selection Criteria:")
    print("• Development: ChromaDB or FAISS")
    print("• Small Production: PGVector or Weaviate")
    print("• Enterprise: Pinecone or Weaviate")
    print("• Cost-sensitive: FAISS with custom wrapper")
    
    return databases

def production_rag_architecture():
    """
    Design production-ready RAG architecture
    """
    print("\nPRODUCTION RAG ARCHITECTURE")
    print("="*30)
    
    architecture = """
    🏗️ PRODUCTION COMPONENTS:
    
    1. INGESTION LAYER:
       ├── Document Loaders (Multi-format)
       ├── Preprocessing Pipeline
       ├── Chunking Strategy
       └── Metadata Extraction
    
    2. INDEXING LAYER:
       ├── Embedding Service (GPU-accelerated)
       ├── Vector Database (Distributed)
       ├── Metadata Store (PostgreSQL)
       └── Cache Layer (Redis)
    
    3. RETRIEVAL LAYER:
       ├── Query Processing
       ├── Hybrid Search (Dense + Sparse)
       ├── Re-ranking Service
       └── Context Window Management
    
    4. GENERATION LAYER:
       ├── LLM Service (Load balanced)
       ├── Prompt Engineering
       ├── Response Formatting
       └── Safety Filters
    
    5. EVALUATION & MONITORING:
       ├── Online Metrics (Latency, Throughput)
       ├── Offline Metrics (RAGAS)
       ├── A/B Testing Framework
       └── Drift Detection
    
    6. INFRASTRUCTURE:
       ├── API Gateway
       ├── Load Balancers
       ├── Kubernetes Orchestration
       └── Observability Stack
    """
    
    print(architecture)

class ProductionRAGPipeline:
    def __init__(self, config):
        self.config = config
        self.embedder = self._initialize_embedder()
        self.vector_store = self._initialize_vector_store()
        self.llm = self._initialize_llm()
        self.reranker = self._initialize_reranker()
        
    def _initialize_embedder(self):
        """Initialize embedding model with caching"""
        return SentenceTransformer(
            self.config['embedding_model'],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    def _initialize_vector_store(self):
        """Initialize vector database with connection pooling"""
        if self.config['vector_db'] == 'chromadb':
            client = chromadb.PersistentClient(
                path=self.config['db_path']
            )
            return client.create_collection(
                name=self.config['collection_name'],
                metadata={"hnsw:space": "cosine"}
            )
        # Add other vector stores
    
    def _initialize_llm(self):
        """Initialize LLM with proper configuration"""
        # Implementation depends on LLM choice
        pass
    
    def _initialize_reranker(self):
        """Initialize cross-encoder for reranking"""
        from sentence_transformers import CrossEncoder
        return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
    
    def index_documents(self, documents, batch_size=100):
        """Batch index documents with progress tracking"""
        total_docs = len(documents)
        
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i+batch_size]
            
            # Extract texts and metadata
            texts = [doc.page_content for doc in batch]
            metadatas = [doc.metadata for doc in batch]
            
            # Generate embeddings
            embeddings = self.embedder.encode(texts, 
                                             show_progress_bar=False)
            
            # Add to vector store
            self.vector_store.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=[str(i+j) for j in range(len(batch))]
            )
            
            logger.info(f"Indexed {min(i+batch_size, total_docs)}/{total_docs} documents")
    
    def retrieve(self, query, k=10, use_reranker=True):
        """Retrieve relevant documents with optional reranking"""
        # Generate query embedding
        query_embedding = self.embedder.encode(query)
        
        # Initial retrieval
        results = self.vector_store.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k * 3 if use_reranker else k
        )
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        if use_reranker and len(documents) > k:
            # Rerank using cross-encoder
            pairs = [[query, doc] for doc in documents]
            scores = self.reranker.predict(pairs)
            
            # Sort by reranker scores
            reranked_indices = np.argsort(scores)[::-1][:k]
            
            documents = [documents[i] for i in reranked_indices]
            metadatas = [metadatas[i] for i in reranked_indices]
            distances = [scores[i] for i in reranked_indices]
        
        return documents, metadatas, distances
    
    def generate(self, query, context, max_tokens=500):
        """Generate response using LLM with context"""
        prompt = self._construct_prompt(query, context)
        
        # Add your LLM generation logic here
        # response = self.llm.generate(prompt, max_tokens=max_tokens)
        
        # For demo purposes
        response = f"Based on the context: {context[:100]}..."
        
        return response
    
    def _construct_prompt(self, query, context):
        """Construct optimized prompt for RAG"""
        prompt = f"""Answer the following question based on the provided context. 
If the answer cannot be found in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {query}

Answer:"""
        return prompt
```

## 4. Advanced RAG Techniques

### 4.1 Query Enhancement

```python
def query_enhancement_techniques():
    """
    Advanced query enhancement for better retrieval
    """
    print("QUERY ENHANCEMENT TECHNIQUES")
    print("="*32)
    
    techniques = """
    🔮 QUERY ENHANCEMENT:
    
    1. QUERY EXPANSION:
       ├── Synonym Addition
       ├── Acronym Expansion
       ├── Related Terms (Word2Vec)
       └── Domain-specific Expansion
    
    2. QUERY DECOMPOSITION:
       ├── Multi-hop Questions
       ├── Complex Query Breaking
       ├── Sub-query Generation
       └── Temporal Decomposition
    
    3. HYPOTHETICAL DOCUMENT EMBEDDING (HyDE):
       ├── Generate Hypothetical Answer
       ├── Embed Hypothetical Document
       ├── Retrieve Using HyDE Embedding
       └── Combine with Original Query
    
    4. QUERY REWRITING:
       ├── Question Reformulation
       ├── Intent Classification
       ├── Query Correction
       └── Context Injection
    """
    
    print(techniques)

class QueryEnhancer:
    def __init__(self, llm_model=None):
        self.llm = llm_model
        
    def expand_query(self, query):
        """Expand query with synonyms and related terms"""
        # Simplified example
        expanded_terms = []
        
        # Add domain-specific expansions
        finance_terms = {
            'ROI': 'return on investment profit margin',
            'ML': 'machine learning artificial intelligence',
            'RAG': 'retrieval augmented generation'
        }
        
        for term, expansion in finance_terms.items():
            if term.lower() in query.lower():
                expanded_terms.append(expansion)
        
        enhanced_query = f"{query} {' '.join(expanded_terms)}"
        return enhanced_query
    
    def decompose_query(self, query):
        """Decompose complex queries into sub-queries"""
        sub_queries = []
        
        # Check for multi-part questions
        if ' and ' in query.lower():
            parts = query.split(' and ')
            sub_queries.extend(parts)
        
        # Check for comparison questions
        if 'compare' in query.lower() or 'difference between' in query.lower():
            # Extract entities for separate queries
            # This would use NER in production
            sub_queries.append(f"What is {query.split('between')[1].split('and')[0]}?")
            sub_queries.append(f"What is {query.split('and')[1]}?")
        
        if not sub_queries:
            sub_queries = [query]
        
        return sub_queries
    
    def generate_hyde(self, query):
        """Generate hypothetical document embedding"""
        # Generate hypothetical answer using LLM
        hyde_prompt = f"""Write a detailed, factual answer to this question:
Question: {query}

Answer:"""
        
        # This would use actual LLM in production
        hypothetical_answer = f"A comprehensive answer to '{query}' would include..."
        
        return hypothetical_answer
    
    def rewrite_query(self, query, context=None):
        """Rewrite query for better retrieval"""
        rewrite_prompt = f"""Rewrite this question to be more specific and clear:
Original: {query}
Context: {context if context else 'General knowledge'}

Rewritten question:"""
        
        # This would use actual LLM in production
        rewritten = query + " (rewritten for clarity)"
        
        return rewritten
```

### 4.2 Context Management

```python
def context_window_optimization():
    """
    Optimize context window usage for LLMs
    """
    print("CONTEXT WINDOW OPTIMIZATION")
    print("="*30)
    
    strategies = """
    📐 CONTEXT OPTIMIZATION:
    
    1. CONTEXT COMPRESSION:
       ├── Extractive Summarization
       ├── Sentence Ranking
       ├── Key Phrase Extraction
       └── Redundancy Removal
    
    2. CONTEXT ORDERING:
       ├── Relevance-based Ordering
       ├── Lost-in-the-Middle Mitigation
       ├── Diversity Injection
       └── Temporal Ordering
    
    3. CONTEXT FILTERING:
       ├── Relevance Threshold
       ├── Diversity Filter
       ├── Recency Filter
       └── Source Quality Filter
    
    4. DYNAMIC CONTEXT:
       ├── Adaptive k-retrieval
       ├── Iterative Refinement
       ├── Context Accumulation
       └── Conversation Memory
    """
    
    print(strategies)

class ContextManager:
    def __init__(self, max_tokens=4000, model_tokenizer=None):
        self.max_tokens = max_tokens
        self.tokenizer = model_tokenizer
    
    def optimize_context(self, documents, scores, query):
        """Optimize context for LLM consumption"""
        # Sort by relevance
        sorted_docs = sorted(zip(documents, scores), 
                           key=lambda x: x[1], 
                           reverse=True)
        
        # Remove redundancy
        unique_docs = self._remove_redundancy(sorted_docs)
        
        # Compress if needed
        compressed_docs = self._compress_documents(unique_docs, query)
        
        # Order for optimal LLM performance
        ordered_docs = self._order_documents(compressed_docs)
        
        # Fit to context window
        final_context = self._fit_to_window(ordered_docs)
        
        return final_context
    
    def _remove_redundancy(self, documents):
        """Remove redundant information using similarity"""
        unique_docs = []
        seen_content = set()
        
        for doc, score in documents:
            # Simple hash-based deduplication
            doc_hash = hash(doc[:100])  # Hash first 100 chars
            
            if doc_hash not in seen_content:
                unique_docs.append((doc, score))
                seen_content.add(doc_hash)
        
        return unique_docs
    
    def _compress_documents(self, documents, query):
        """Compress documents while preserving key information"""
        compressed = []
        
        for doc, score in documents:
            # Extract sentences most relevant to query
            sentences = doc.split('. ')
            
            # Simple keyword-based relevance
            query_terms = set(query.lower().split())
            
            relevant_sentences = []
            for sent in sentences:
                sent_terms = set(sent.lower().split())
                overlap = len(query_terms & sent_terms)
                
                if overlap > 0:
                    relevant_sentences.append((sent, overlap))
            
            # Keep top sentences
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            compressed_doc = '. '.join([s[0] for s in relevant_sentences[:3]])
            
            compressed.append((compressed_doc, score))
        
        return compressed
    
    def _order_documents(self, documents):
        """Order documents to mitigate lost-in-the-middle problem"""
        # Research shows LLMs pay more attention to beginning and end
        # Place most relevant at beginning and end
        
        if len(documents) <= 2:
            return documents
        
        ordered = []
        docs_list = list(documents)
        
        # Most relevant at beginning
        ordered.append(docs_list[0])
        
        # Second most relevant at end
        if len(docs_list) > 1:
            ordered.append(docs_list[1])
        
        # Rest in the middle
        for doc in docs_list[2:]:
            ordered.insert(len(ordered)//2, doc)
        
        return ordered
    
    def _fit_to_window(self, documents):
        """Ensure context fits within token limits"""
        context_parts = []
        total_tokens = 0
        
        for doc, score in documents:
            # Estimate tokens (rough approximation)
            doc_tokens = len(doc.split()) * 1.3
            
            if total_tokens + doc_tokens <= self.max_tokens:
                context_parts.append(doc)
                total_tokens += doc_tokens
            else:
                # Truncate if needed
                remaining_tokens = self.max_tokens - total_tokens
                if remaining_tokens > 100:
                    words_to_keep = int(remaining_tokens / 1.3)
                    truncated = ' '.join(doc.split()[:words_to_keep])
                    context_parts.append(truncated + "...")
                break
        
        return '\n\n'.join(context_parts)
```

## 5. RAG Evaluation Framework

### 5.1 RAGAS Metrics Implementation

```python
def ragas_evaluation_framework():
    """
    Comprehensive RAG evaluation using RAGAS metrics
    """
    print("RAGAS EVALUATION FRAMEWORK")
    print("="*30)
    
    metrics_explanation = """
    📊 RAGAS METRICS:
    
    1. FAITHFULNESS:
       ├── Measures: Factual consistency with retrieved context
       ├── Formula: # supported claims / # total claims
       ├── Range: [0, 1] (higher is better)
       └── Use: Detect hallucinations
    
    2. ANSWER RELEVANCY:
       ├── Measures: Relevance of answer to question
       ├── Method: Cosine similarity of question-answer embeddings
       ├── Range: [0, 1] (higher is better)
       └── Use: Ensure on-topic responses
    
    3. CONTEXT PRECISION:
       ├── Measures: Signal-to-noise in retrieved context
       ├── Formula: Precision@k for relevant chunks
       ├── Range: [0, 1] (higher is better)
       └── Use: Evaluate retriever precision
    
    4. CONTEXT RECALL:
       ├── Measures: Coverage of required information
       ├── Formula: # retrieved relevant / # total relevant
       ├── Range: [0, 1] (higher is better)
       └── Use: Ensure comprehensive retrieval
    
    5. CONTEXT RELEVANCY:
       ├── Measures: Overall relevance of retrieved context
       ├── Method: LLM-based relevance scoring
       ├── Range: [0, 1] (higher is better)
       └── Use: Evaluate retrieval quality
    
    6. ANSWER CORRECTNESS:
       ├── Measures: Factual accuracy against ground truth
       ├── Method: Semantic similarity + factual overlap
       ├── Range: [0, 1] (higher is better)
       └── Use: End-to-end quality assessment
    """
    
    print(metrics_explanation)

class RAGEvaluator:
    def __init__(self, llm_evaluator=None):
        self.llm_evaluator = llm_evaluator
        self.metrics = {}
        
    def evaluate_faithfulness(self, answer, contexts):
        """Evaluate if answer is faithful to context"""
        # Extract claims from answer
        claims = self._extract_claims(answer)
        
        supported_claims = 0
        for claim in claims:
            if self._is_supported(claim, contexts):
                supported_claims += 1
        
        faithfulness = supported_claims / len(claims) if claims else 0
        
        return {
            'score': faithfulness,
            'supported_claims': supported_claims,
            'total_claims': len(claims)
        }
    
    def evaluate_answer_relevancy(self, question, answer, embedder):
        """Evaluate relevance of answer to question"""
        # Generate embeddings
        q_embedding = embedder.encode(question)
        a_embedding = embedder.encode(answer)
        
        # Calculate cosine similarity
        relevancy = np.dot(q_embedding, a_embedding) / (
            np.linalg.norm(q_embedding) * np.linalg.norm(a_embedding)
        )
        
        return {
            'score': float(relevancy),
            'interpretation': 'Highly relevant' if relevancy > 0.8 else 'Moderately relevant'
        }
    
    def evaluate_context_precision(self, retrieved_contexts, relevant_contexts):
        """Evaluate precision of retrieved contexts"""
        if not retrieved_contexts:
            return {'score': 0, 'precision_at_k': []}
        
        precision_at_k = []
        relevant_found = 0
        
        for k, context in enumerate(retrieved_contexts, 1):
            if context in relevant_contexts:
                relevant_found += 1
            precision_at_k.append(relevant_found / k)
        
        return {
            'score': np.mean(precision_at_k),
            'precision_at_k': precision_at_k
        }
    
    def evaluate_context_recall(self, retrieved_contexts, ground_truth_contexts):
        """Evaluate recall of required information"""
        if not ground_truth_contexts:
            return {'score': 1.0, 'coverage': 'N/A'}
        
        covered = sum(1 for gt in ground_truth_contexts 
                     if any(gt in ret for ret in retrieved_contexts))
        
        recall = covered / len(ground_truth_contexts)
        
        return {
            'score': recall,
            'coverage': f"{covered}/{len(ground_truth_contexts)}"
        }
    
    def comprehensive_evaluation(self, questions, answers, contexts, ground_truths):
        """Run comprehensive RAG evaluation"""
        results = {
            'faithfulness': [],
            'answer_relevancy': [],
            'context_precision': [],
            'context_recall': []
        }
        
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        for q, a, c, gt in zip(questions, answers, contexts, ground_truths):
            # Evaluate each metric
            results['faithfulness'].append(
                self.evaluate_faithfulness(a, c)['score']
            )
            results['answer_relevancy'].append(
                self.evaluate_answer_relevancy(q, a, embedder)['score']
            )
            # Add context precision and recall logic
        
        # Calculate aggregate scores
        aggregate_results = {
            metric: {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
            for metric, scores in results.items()
        }
        
        return aggregate_results
    
    def _extract_claims(self, text):
        """Extract factual claims from text"""
        # Simplified: Split by sentences
        sentences = text.split('. ')
        # Filter for declarative sentences (would use NLP in production)
        claims = [s for s in sentences if len(s) > 10]
        return claims
    
    def _is_supported(self, claim, contexts):
        """Check if claim is supported by contexts"""
        # Simplified: Check for keyword overlap
        claim_words = set(claim.lower().split())
        
        for context in contexts:
            context_words = set(context.lower().split())
            overlap = len(claim_words & context_words) / len(claim_words)
            
            if overlap > 0.5:  # 50% word overlap threshold
                return True
        
        return False
```

### 5.2 Production Monitoring

```python
def production_monitoring_framework():
    """
    Production monitoring for RAG systems
    """
    print("PRODUCTION RAG MONITORING")
    print("="*28)
    
    monitoring_components = """
    📈 MONITORING FRAMEWORK:
    
    1. ONLINE METRICS:
       ├── Response Latency (P50, P95, P99)
       ├── Retrieval Time
       ├── Generation Time
       ├── Token Usage
       ├── Error Rate
       └── Throughput (QPS)
    
    2. QUALITY METRICS:
       ├── User Satisfaction (Thumbs up/down)
       ├── Conversation Completion Rate
       ├── Answer Completeness
       ├── Hallucination Detection
       └── Context Relevance Score
    
    3. RETRIEVAL METRICS:
       ├── Hit Rate @k
       ├── MRR (Mean Reciprocal Rank)
       ├── Coverage (% of queries answered)
       ├── Diversity Score
       └── Freshness (doc age)
    
    4. SYSTEM HEALTH:
       ├── Vector DB Query Time
       ├── Embedding Service Availability
       ├── LLM API Success Rate
       ├── Cache Hit Rate
       └── Memory/CPU Usage
    
    5. DRIFT DETECTION:
       ├── Query Distribution Shift
       ├── Document Distribution Shift
       ├── Performance Degradation
       ├── Topic Drift
       └── User Behavior Changes
    """
    
    print(monitoring_components)

class RAGMonitor:
    def __init__(self, config):
        self.config = config
        self.metrics_buffer = []
        self.alert_thresholds = {
            'latency_p99': 5000,  # ms
            'error_rate': 0.01,    # 1%
            'hallucination_rate': 0.05,  # 5%
            'retrieval_hit_rate': 0.8   # 80%
        }
    
    def log_request(self, request_data):
        """Log RAG request for monitoring"""
        metric = {
            'timestamp': pd.Timestamp.now(),
            'query': request_data['query'],
            'retrieval_time': request_data['retrieval_time'],
            'generation_time': request_data['generation_time'],
            'total_latency': request_data['total_latency'],
            'num_tokens': request_data['num_tokens'],
            'num_chunks_retrieved': request_data['num_chunks'],
            'user_feedback': request_data.get('feedback')
        }
        
        self.metrics_buffer.append(metric)
        
        # Check thresholds
        self._check_alerts(metric)
    
    def _check_alerts(self, metric):
        """Check if metrics exceed alert thresholds"""
        if metric['total_latency'] > self.alert_thresholds['latency_p99']:
            logger.warning(f"High latency detected: {metric['total_latency']}ms")
        
        # Additional alert logic
    
    def calculate_online_metrics(self, window='1h'):
        """Calculate online metrics for dashboard"""
        df = pd.DataFrame(self.metrics_buffer)
        
        # Filter to time window
        cutoff = pd.Timestamp.now() - pd.Timedelta(window)
        df = df[df['timestamp'] > cutoff]
        
        metrics = {
            'latency_p50': df['total_latency'].quantile(0.5),
            'latency_p95': df['total_latency'].quantile(0.95),
            'latency_p99': df['total_latency'].quantile(0.99),
            'avg_retrieval_time': df['retrieval_time'].mean(),
            'avg_generation_time': df['generation_time'].mean(),
            'throughput': len(df) / (pd.Timedelta(window).seconds / 60),  # QPM
            'avg_chunks_retrieved': df['num_chunks_retrieved'].mean()
        }
        
        return metrics
    
    def detect_drift(self, current_queries, baseline_queries):
        """Detect query distribution drift"""
        # Simplified: Would use more sophisticated methods in production
        # like KL-divergence or adversarial validation
        
        # Compare query length distribution
        current_lengths = [len(q.split()) for q in current_queries]
        baseline_lengths = [len(q.split()) for q in baseline_queries]
        
        from scipy import stats
        ks_statistic, p_value = stats.ks_2samp(current_lengths, baseline_lengths)
        
        drift_detected = p_value < 0.05
        
        return {
            'drift_detected': drift_detected,
            'ks_statistic': ks_statistic,
            'p_value': p_value,
            'interpretation': 'Significant drift' if drift_detected else 'No significant drift'
        }
```

## 6. Hands-On Example

### 6.1 Complete RAG Pipeline Implementation

```python
def complete_rag_example():
    """
    End-to-end RAG system implementation
    """
    print("COMPLETE RAG PIPELINE EXAMPLE")
    print("="*32)
    print("Building Financial Document Q&A System")
    
    # 1. Configuration
    config = {
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'vector_db': 'chromadb',
        'db_path': './chroma_db',
        'collection_name': 'financial_docs',
        'chunk_size': 500,
        'chunk_overlap': 50,
        'retrieval_k': 5,
        'rerank': True
    }
    
    # 2. Initialize components
    print("\n1. Initializing Components...")
    
    # Document processor
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config['chunk_size'],
        chunk_overlap=config['chunk_overlap']
    )
    
    # Embeddings
    embedder = SentenceTransformer(config['embedding_model'])
    
    # Vector store
    client = chromadb.PersistentClient(path=config['db_path'])
    
    # Try to get existing collection or create new
    try:
        collection = client.get_collection(config['collection_name'])
        print(f"   Using existing collection: {config['collection_name']}")
    except:
        collection = client.create_collection(
            config['collection_name'],
            metadata={"hnsw:space": "cosine"}
        )
        print(f"   Created new collection: {config['collection_name']}")
    
    # 3. Document ingestion
    print("\n2. Document Ingestion...")
    
    # Sample financial documents
    sample_documents = [
        {
            'content': """Risk Management Framework: 
            Our comprehensive risk management approach includes credit risk assessment, 
            market risk monitoring, and operational risk controls. We use Value at Risk (VaR) 
            models with 99% confidence intervals and conduct daily stress testing. 
            The framework is reviewed quarterly by the Risk Committee.""",
            'metadata': {'source': 'risk_policy.pdf', 'section': 'overview', 'year': 2024}
        },
        {
            'content': """Quarterly Financial Results Q3 2024:
            Revenue increased by 15% year-over-year to $2.5 billion. Operating margin 
            improved to 35% from 32% in Q3 2023. The growth was primarily driven by 
            our cloud services division which saw a 45% increase in subscriptions.""",
            'metadata': {'source': 'q3_earnings.pdf', 'section': 'summary', 'year': 2024}
        },
        {
            'content': """Compliance and Regulatory Updates:
            Following the latest regulatory guidelines, we have implemented enhanced 
            KYC procedures and automated transaction monitoring systems. Our compliance 
            rate has improved to 99.5% with zero critical violations in the past quarter.""",
            'metadata': {'source': 'compliance_report.pdf', 'section': 'updates', 'year': 2024}
        }
    ]
    
    # Process documents
    all_chunks = []
    for doc in sample_documents:
        # Create document object
        from langchain.schema import Document
        langchain_doc = Document(
            page_content=doc['content'],
            metadata=doc['metadata']
        )
        
        # Split into chunks
        chunks = text_splitter.split_documents([langchain_doc])
        all_chunks.extend(chunks)
    
    print(f"   Created {len(all_chunks)} chunks from {len(sample_documents)} documents")
    
    # Index chunks
    for i, chunk in enumerate(all_chunks):
        embedding = embedder.encode(chunk.page_content)
        
        collection.add(
            embeddings=[embedding.tolist()],
            documents=[chunk.page_content],
            metadatas=[chunk.metadata],
            ids=[f"chunk_{i}"]
        )
    
    print(f"   Indexed all chunks to vector database")
    
    # 4. Query Processing
    print("\n3. Query Processing...")
    
    test_queries = [
        "What is our current risk management approach?",
        "How did cloud services perform in Q3?",
        "What is our compliance rate?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        # Generate query embedding
        query_embedding = embedder.encode(query)
        
        # Retrieve relevant chunks
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=config['retrieval_k']
        )
        
        # Extract retrieved documents
        retrieved_docs = results['documents'][0]
        retrieved_metadata = results['metadatas'][0]
        distances = results['distances'][0]
        
        print(f"Retrieved {len(retrieved_docs)} relevant chunks:")
        for i, (doc, meta, dist) in enumerate(zip(retrieved_docs, retrieved_metadata, distances)):
            print(f"  [{i+1}] Source: {meta['source']}, Distance: {dist:.3f}")
            print(f"      Preview: {doc[:100]}...")
    
    # 5. Response Generation (Simulated)
    print("\n4. Response Generation...")
    
    # Context preparation
    context = "\n\n".join(retrieved_docs)
    
    prompt = f"""Based on the following context, answer the question accurately.
If the information is not in the context, say "I don't have that information."

Context:
{context}

Question: {query}

Answer:"""
    
    # In production, this would call an LLM
    response = "Based on the context, our compliance rate has improved to 99.5% with zero critical violations in the past quarter."
    
    print(f"Generated Response: {response}")
    
    # 6. Evaluation
    print("\n5. Evaluation Metrics...")
    
    evaluator = RAGEvaluator()
    
    # Evaluate faithfulness
    faithfulness = evaluator.evaluate_faithfulness(response, retrieved_docs)
    print(f"Faithfulness Score: {faithfulness['score']:.3f}")
    
    # Evaluate answer relevancy
    relevancy = evaluator.evaluate_answer_relevancy(query, response, embedder)
    print(f"Answer Relevancy: {relevancy['score']:.3f}")
    
    # 7. Production Metrics
    print("\n6. Production Metrics...")
    
    metrics = {
        'query': query,
        'retrieval_time': 45,  # ms
        'generation_time': 250,  # ms
        'total_latency': 295,  # ms
        'num_tokens': len(response.split()),
        'num_chunks': len(retrieved_docs),
        'faithfulness_score': faithfulness['score'],
        'relevancy_score': relevancy['score']
    }
    
    print(f"Latency: {metrics['total_latency']}ms")
    print(f"Tokens: {metrics['num_tokens']}")
    print(f"Quality Score: {(faithfulness['score'] + relevancy['score']) / 2:.3f}")
    
    return {
        'config': config,
        'chunks': all_chunks,
        'metrics': metrics
    }

# Run the complete example
if __name__ == "__main__":
    results = complete_rag_example()
```

## 7. Interview Tips & Common Traps

### 7.1 Critical Misconceptions and Corrections

```python
print("INTERVIEW TIPS & COMMON TRAPS")
print("="*35)

interview_traps = """
❌ COMMON MISCONCEPTIONS → ✅ CORRECT UNDERSTANDING
================================================================

1. VECTOR SEARCH
❌ "More retrieved documents always improve quality"
✅ Quality > Quantity; too many docs dilute relevance
✅ Optimal k depends on use case (typically 3-10)
✅ Use reranking to improve precision

2. CHUNKING STRATEGY
❌ "Smaller chunks are always better for precision"
✅ Balance between precision and context preservation
✅ 500-1000 tokens often optimal
✅ Consider semantic boundaries, not just size

3. EMBEDDING MODELS
❌ "Larger embedding dimensions = better retrieval"
✅ Depends on data volume and domain
✅ Smaller models can be fine-tuned effectively
✅ Consider inference speed vs. quality tradeoff

4. HALLUCINATION
❌ "RAG completely eliminates hallucinations"
✅ RAG reduces but doesn't eliminate hallucinations
✅ Need explicit grounding checks and faithfulness metrics
✅ Prompt engineering crucial for adherence to context

5. CONTEXT WINDOW
❌ "Fill the entire context window for best results"
✅ LLMs perform worse with too much context (lost-in-middle)
✅ Quality of context matters more than quantity
✅ Strategic ordering improves performance

6. REAL-TIME UPDATES
❌ "RAG systems automatically stay current"
✅ Need explicit re-indexing strategies
✅ Consider incremental updates vs. full rebuilds
✅ Version control for document updates

7. EVALUATION
❌ "Human evaluation is always better than automated"
✅ Automated metrics provide consistency and scale
✅ Combine both for comprehensive evaluation
✅ RAGAS metrics correlate well with human judgment

8. HYBRID SEARCH
❌ "Dense retrieval is always superior to keyword search"
✅ Hybrid (dense + sparse) often outperforms either alone
✅ Keyword search better for specific terms, acronyms
✅ Dense search better for semantic similarity

9. PRODUCTION SCALING
❌ "RAG systems scale linearly with more documents"
✅ Vector search complexity grows with index size
✅ Need strategies like hierarchical indexing
✅ Consider sharding and distributed architectures

10. COST OPTIMIZATION
❌ "Using the best models everywhere maximizes quality"
✅ Use model cascading (cheap → expensive)
✅ Cache frequent queries
✅ Optimize chunk size and retrieval k for cost
"""

print(interview_traps)

def quick_diagnostic_checklist():
    """
    RAG implementation checklist
    """
    print("\n🔍 RAG DIAGNOSTIC CHECKLIST")
    print("="*30)
    
    checklist = [
        "□ Define clear evaluation metrics before implementation",
        "□ Test multiple chunking strategies on your data",
        "□ Implement both dense and sparse retrieval",
        "□ Set up reranking for improved precision",
        "□ Monitor for hallucinations with faithfulness metrics",
        "□ Implement context compression for long documents",
        "□ Use metadata filtering for better retrieval",
        "□ Set up incremental indexing for updates",
        "□ Cache embeddings and frequent queries",
        "□ Implement fallback strategies for retrieval failures",
        "□ Monitor latency at each pipeline stage",
        "□ Set up A/B testing framework for improvements"
    ]
    
    for item in checklist:
        print(item)

def interview_qa_simulation():
    """
    Common RAG interview questions with answers
    """
    print("\n💼 INTERVIEW Q&A SIMULATION")
    print("="*30)
    
    qa_pairs = [
        {
            "Q": "How do you handle documents that exceed the context window?",
            "A": """
Multiple strategies for long documents:

1. Hierarchical Summarization:
   • Chunk document into sections
   • Summarize each section
   • Create summary tree
   • Retrieve at multiple granularities

2. Sliding Window:
   • Process document in overlapping windows
   • Aggregate responses
   • Maintain context continuity

3. Extractive Selection:
   • Identify most relevant passages
   • Use reranking to prioritize
   • Compress redundant information

4. Map-Reduce Pattern:
   • Map: Process chunks independently
   • Reduce: Combine chunk responses
   • Useful for analytical queries
            """
        },
        {
            "Q": "How do you prevent hallucinations in RAG systems?",
            "A": """
Multi-layered approach to prevent hallucinations:

1. Retrieval Quality:
   • Ensure high-quality, relevant retrieval
   • Use reranking to improve precision
   • Set confidence thresholds

2. Prompt Engineering:
   • Explicit instructions to stick to context
   • Include "I don't know" option
   • Ask for citations/references

3. Faithfulness Checking:
   • Validate claims against retrieved context
   • Use NLI models for consistency checking
   • Flag unsupported statements

4. Post-processing:
   • Fact verification against knowledge base
   • Consistency checking across responses
   • Confidence scoring for each claim

5. Monitoring:
   • Track faithfulness metrics
   • User feedback on accuracy
   • Regular audits of responses
            """
        },
        {
            "Q": "How do you optimize RAG for latency-sensitive applications?",
            "A": """
Latency optimization strategies:

1. Indexing Optimizations:
   • Use approximate nearest neighbor (ANN) search
   • Optimize index parameters (HNSW, IVF)
   • GPU acceleration for similarity search

2. Caching Strategy:
   • Cache embeddings for frequent queries
   • Cache retrieval results
   • Semantic caching for similar queries

3. Model Optimization:
   • Use smaller, faster embedding models
   • Quantization (INT8, FP16)
   • Model distillation

4. Pipeline Parallelization:
   • Parallel retrieval from multiple indices
   • Async embedding generation
   • Batch processing where possible

5. Infrastructure:
   • Colocate services to reduce network latency
   • Use CDN for static content
   • Load balancing and horizontal scaling

Target latencies:
• Embedding: <50ms
• Retrieval: <100ms
• Reranking: <50ms
• Generation: <500ms
• Total: <1 second
            """
        },
        {
            "Q": "How do you handle multi-modal content in RAG?",
            "A": """
Multi-modal RAG approaches:

1. Unified Embeddings:
   • Use models like CLIP for image-text
   • Project all modalities to same space
   • Single vector index for all content

2. Separate Indices:
   • Text index for documents
   • Image index for visuals
   • Table index for structured data
   • Fusion at retrieval time

3. Content Extraction:
   • OCR for images with text
   • Table parsing for structured data
   • Caption generation for images
   • Convert to text when possible

4. Cross-modal Retrieval:
   • Text query → Image results
   • Image query → Text results
   • Combined relevance scoring

5. Generation Strategy:
   • Reference images in text response
   • Generate text descriptions
   • Provide links to visual content
            """
        }
    ]
    
    for i, qa in enumerate(qa_pairs, 1):
        print(f"\nQ{i}: {qa['Q']}")
        print(f"A{i}: {qa['A']}")
        print("-" * 60)

# Run interview preparation sections
quick_diagnostic_checklist()
interview_qa_simulation()

def architecture_decision_guide():
    """
    Guide for RAG architecture decisions
    """
    print("\n🎯 RAG ARCHITECTURE DECISION GUIDE")
    print("="*36)
    
    decision_guide = """
    CHOOSING RAG ARCHITECTURE:
    
    NAIVE RAG - Use When:
    ✓ POC or prototype phase
    ✓ Simple Q&A use case
    ✓ Small document corpus (<1K docs)
    ✓ Low latency requirements
    ✓ Limited engineering resources
    
    ADVANCED RAG - Use When:
    ✓ Production system
    ✓ Complex queries requiring reasoning
    ✓ Medium corpus (1K-100K docs)
    ✓ Quality > Speed
    ✓ Need query enhancement
    
    MODULAR RAG - Use When:
    ✓ Enterprise deployment
    ✓ Multiple data sources
    ✓ Need flexibility to swap components
    ✓ Different teams own different parts
    ✓ Compliance requirements
    
    AGENTIC RAG - Use When:
    ✓ Complex multi-step reasoning
    ✓ Need tool integration
    ✓ Dynamic retrieval strategies
    ✓ Self-improving system required
    ✓ Autonomous operation needed
    
    KEY ARCHITECTURAL DECISIONS:
    
    1. Chunking Strategy:
       • Fixed-size: Simple use cases
       • Semantic: Better context preservation
       • Hierarchical: Long documents
       • Custom: Domain-specific requirements
    
    2. Retrieval Method:
       • Dense only: Semantic search
       • Sparse only: Keyword/exact match
       • Hybrid: Best of both (recommended)
       • Multi-stage: Complex queries
    
    3. Vector Database:
       • Dev/POC: ChromaDB, FAISS
       • Production: Pinecone, Weaviate
       • Cost-sensitive: PGVector
       • High-scale: Custom FAISS + S3
    
    4. Reranking:
       • No reranking: Low latency critical
       • Cross-encoder: Quality critical
       • LLM-based: Complex relevance
       • Cascaded: Balance quality/speed
    """
    
    print(decision_guide)

architecture_decision_guide()

print("\n" + "="*60)
print("📚 RAG INTERVIEW PREPARATION COMPLETE!")
print("="*60)

final_summary = """
KEY TAKEAWAYS FOR INTERVIEWS:

🎯 TECHNICAL DEPTH:
• Understand retrieval-generation tradeoffs
• Know when to use different chunking strategies
• Explain hybrid search benefits
• Describe faithfulness vs relevancy metrics

🏗️ ARCHITECTURE:
• Start simple (Naive RAG) then evolve
• Modular design for production systems
• Consider latency at each stage
• Plan for incremental updates

📊 EVALUATION:
• RAGAS metrics for offline evaluation
• Online metrics for production monitoring
• Combine automated and human evaluation
• Focus on faithfulness to prevent hallucinations

⚡ OPTIMIZATION:
• Caching at multiple levels
• Reranking for precision
• Context compression for efficiency
• Model cascading for cost optimization

🚨 COMMON PITFALLS:
• Over-retrieving dilutes quality
• Ignoring keyword search value
• Not monitoring for drift
• Assuming RAG eliminates hallucinations

💡 PRODUCTION TIPS:
• Version control your indices
• Implement gradual rollouts
• Monitor component-level metrics
• Plan for failure modes
"""

print(final_summary)
```
