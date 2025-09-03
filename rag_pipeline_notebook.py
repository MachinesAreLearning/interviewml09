# Run advanced RAG demonstration
print("🚀 Advanced RAG Techniques Demonstration")
# hier_rag, conv_rag, coverage_stats = demonstrate_advanced_rag()

---

## 7. RAG Evaluation Framework {#evaluation}

### Comprehensive RAG Evaluation Metrics

```python
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer
import re
from dataclasses import dataclass
from collections import Counter

@dataclass
class RAGEvaluationResult:
    """Comprehensive RAG evaluation results"""
    # Retrieval metrics
    retrieval_precision: float
    retrieval_recall: float
    retrieval_f1: float
    retrieval_ndcg: float
    
    # Generation metrics
    bleu_score: float
    rouge_l: float
    bertscore_f1: float
    
    # RAG-specific metrics
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    
    # Overall scores
    overall_score: float
    human_rating: Optional[float] = None

class RAGEvaluator:
    """Comprehensive RAG evaluation framework"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize evaluation models
        try:
            import evaluate
            self.bleu_metric = evaluate.load("bleu")
            self.rouge_metric = evaluate.load("rouge")
            self.bertscore_metric = evaluate.load("bertscore")
        except ImportError:
            print("⚠️ Hugging Face evaluate library not available. Some metrics will be simplified.")
            self.bleu_metric = None
            self.rouge_metric = None
            self.bertscore_metric = None
        
        print("📊 RAG Evaluator initialized")
    
    def evaluate_retrieval(self, retrieved_docs: List[str], relevant_docs: List[str],
                         all_docs: List[str]) -> Dict[str, float]:
        """Evaluate retrieval performance with precision, recall, F1, and nDCG"""
        
        retrieved_set = set(retrieved_docs)
        relevant_set = set(relevant_docs)
        
        # Basic precision and recall
        true_positives = len(retrieved_set.intersection(relevant_set))
        
        precision = true_positives / len(retrieved_set) if retrieved_set else 0
        recall = true_positives / len(relevant_set) if relevant_set else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate nDCG (Normalized Discounted Cumulative Gain)
        ndcg = self.calculate_ndcg(retrieved_docs, relevant_docs)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'ndcg': ndcg
        }
    
    def calculate_ndcg(self, retrieved_docs: List[str], relevant_docs: List[str], k: int = None) -> float:
        """Calculate Normalized Discounted Cumulative Gain"""
        
        if k is None:
            k = len(retrieved_docs)
        
        relevant_set = set(relevant_docs)
        
        # Calculate DCG
        dcg = 0
        for i, doc in enumerate(retrieved_docs[:k]):
            relevance = 1 if doc in relevant_set else 0
            dcg += relevance / np.log2(i + 2)  # i + 2 because log2(1) = 0
        
        # Calculate IDCG (Ideal DCG)
        ideal_relevances = [1] * min(len(relevant_docs), k) + [0] * max(0, k - len(relevant_docs))
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))
        
        # Calculate nDCG
        ndcg = dcg / idcg if idcg > 0 else 0
        
        return ndcg
    
    def evaluate_generation(self, generated_answer: str, reference_answer: str) -> Dict[str, float]:
        """Evaluate generation quality using multiple metrics"""
        
        results = {}
        
        # BLEU Score
        if self.bleu_metric:
            bleu_result = self.bleu_metric.compute(
                predictions=[generated_answer],
                references=[[reference_answer]]
            )
            results['bleu'] = bleu_result['bleu']
        else:
            # Simplified BLEU calculation
            results['bleu'] = self.simple_bleu(generated_answer, reference_answer)
        
        # ROUGE Score
        if self.rouge_metric:
            rouge_result = self.rouge_metric.compute(
                predictions=[generated_answer],
                references=[reference_answer]
            )
            results['rouge_l'] = rouge_result['rougeL']
        else:
            # Simplified ROUGE-L calculation
            results['rouge_l'] = self.simple_rouge_l(generated_answer, reference_answer)
        
        # BERTScore
        if self.bertscore_metric:
            bertscore_result = self.bertscore_metric.compute(
                predictions=[generated_answer],
                references=[reference_answer],
                lang="en"
            )
            results['bertscore_f1'] = bertscore_result['f1'][0]
        else:
            # Use embedding similarity as proxy
            results['bertscore_f1'] = self.embedding_similarity(generated_answer, reference_answer)
        
        return results
    
    def simple_bleu(self, generated: str, reference: str) -> float:
        """Simplified BLEU score calculation"""
        
        gen_tokens = generated.lower().split()
        ref_tokens = reference.lower().split()
        
        if not gen_tokens or not ref_tokens:
            return 0.0
        
        # 1-gram precision
        gen_counts = Counter(gen_tokens)
        ref_counts = Counter(ref_tokens)
        
        overlap = sum(min(gen_counts[token], ref_counts[token]) for token in gen_counts)
        precision = overlap / len(gen_tokens) if gen_tokens else 0
        
        # Brevity penalty
        bp = min(1.0, len(gen_tokens) / len(ref_tokens)) if ref_tokens else 0
        
        return bp * precision
    
    def simple_rouge_l(self, generated: str, reference: str) -> float:
        """Simplified ROUGE-L score calculation"""
        
        gen_tokens = generated.lower().split()
        ref_tokens = reference.lower().split()
        
        if not gen_tokens or not ref_tokens:
            return 0.0
        
        # Find longest common subsequence
        lcs_length = self.lcs_length(gen_tokens, ref_tokens)
        
        precision = lcs_length / len(gen_tokens) if gen_tokens else 0
        recall = lcs_length / len(ref_tokens) if ref_tokens else 0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate length of longest common subsequence"""
        
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def embedding_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using embeddings"""
        
        embeddings = self.embedding_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    def evaluate_faithfulness(self, generated_answer: str, context: List[str]) -> float:
        """Evaluate if generated answer is faithful to the provided context"""
        
        if not context:
            return 0.0
        
        context_text = " ".join(context)
        
        # Metho# RAG Pipeline Implementation - Interview Preparation Notebook

## 🎯 Learning Objectives
- Build end-to-end RAG systems from document ingestion to response generation
- Implement advanced retrieval strategies (dense, sparse, hybrid)
- Master vector databases and similarity search optimization
- Evaluate RAG systems using faithfulness, relevance, and hallucination metrics
- Handle production challenges: chunking, reranking, and context optimization

## 📚 Table of Contents
1. [RAG Architecture & Theory](#rag-architecture)
2. [Document Processing & Chunking](#document-processing)
3. [Embedding Models & Vector Stores](#embeddings-vectors)
4. [Retrieval Strategies](#retrieval-strategies)
5. [Generation & Context Integration](#generation)
6. [Advanced RAG Techniques](#advanced-rag)
7. [RAG Evaluation Framework](#evaluation)
8. [Production Implementation](#production)
9. [Interview Tips & Common Pitfalls](#interview-tips)

---

## 1. RAG Architecture & Theory {#rag-architecture}

### Mathematical Foundation of RAG

RAG combines retrieval and generation through a probabilistic framework:

**P(y|x) = Σ P(y|x,z) P(z|x)**

Where:
- x = input query
- y = generated response  
- z = retrieved documents
- P(z|x) = retrieval model probability
- P(y|x,z) = generation model probability

```python
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM,
    DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
)
from sentence_transformers import SentenceTransformer
import faiss
import chromadb
from datasets import Dataset
import json
import requests
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import logging
import time
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """Configuration for RAG pipeline"""
    # Model configurations
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    generation_model: str = "microsoft/DialoGPT-medium"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Retrieval configurations
    top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50
    similarity_threshold: float = 0.7
    
    # Generation configurations
    max_new_tokens: int = 256
    temperature: float = 0.7
    do_sample: bool = True
    
    # Vector store configurations
    vector_dim: int = 384
    index_type: str = "IVFFlat"  # or "HNSW"
    nprobe: int = 10

class RAGSystem:
    """Complete RAG system implementation"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.setup_models()
        self.setup_vector_store()
        self.documents = []
        self.embeddings = None
        
    def setup_models(self):
        """Initialize embedding and generation models"""
        logger.info("Loading models...")
        
        # Embedding model
        self.embedding_model = SentenceTransformer(self.config.embedding_model)
        
        # Generation model  
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.generation_model)
        self.generator = AutoModelForCausalLM.from_pretrained(self.config.generation_model)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info("Models loaded successfully")
    
    def setup_vector_store(self):
        """Initialize FAISS vector store"""
        self.index = None
        self.is_trained = False
        
    def demonstrate_rag_components(self):
        """Demonstrate core RAG components"""
        print("🏗️ RAG SYSTEM ARCHITECTURE")
        print("="*50)
        
        architecture_info = """
        RAG PIPELINE COMPONENTS:
        
        1. DOCUMENT PROCESSING:
           ├── Text extraction (PDF, HTML, etc.)
           ├── Chunking strategies (fixed, semantic, hierarchical)  
           ├── Metadata extraction and enrichment
           └── Quality filtering and deduplication
        
        2. EMBEDDING & INDEXING:
           ├── Dense embeddings (BERT, Sentence-BERT)
           ├── Sparse embeddings (BM25, TF-IDF)
           ├── Vector database (FAISS, Pinecone, Weaviate)
           └── Index optimization (IVF, HNSW, PQ)
        
        3. RETRIEVAL:
           ├── Similarity search (cosine, dot product, L2)
           ├── Hybrid search (dense + sparse)
           ├── Query expansion and rewriting
           └── Reranking and filtering
        
        4. GENERATION:
           ├── Context integration and prompt engineering
           ├── Instruction-following and format control
           ├── Length and style adaptation
           └── Fact verification and grounding
        
        5. EVALUATION:
           ├── Retrieval metrics (precision, recall, nDCG)
           ├── Generation metrics (BLEU, ROUGE, BERTScore)
           ├── End-to-end metrics (faithfulness, relevance)
           └── Human evaluation and user satisfaction
        """
        
        print(architecture_info)
        
        return architecture_info

# Initialize RAG system
config = RAGConfig()
rag_system = RAGSystem(config)
rag_system.demonstrate_rag_components()
```

---

## 2. Document Processing & Chunking {#document-processing}

### Advanced Chunking Strategies

```python
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
import spacy
from transformers import AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@dataclass
class Chunk:
    """Document chunk with metadata"""
    content: str
    start_idx: int
    end_idx: int
    tokens: int
    metadata: Dict
    embedding: Optional[np.ndarray] = None

class AdvancedChunker:
    """Advanced document chunking strategies"""
    
    def __init__(self, tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # Try to load spacy model, fallback if not available
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Some features will be limited.")
            self.nlp = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using the embedding model tokenizer"""
        return len(self.tokenizer.encode(text))
    
    def fixed_size_chunking(self, text: str, chunk_size: int = 512, 
                          overlap: int = 50) -> List[Chunk]:
        """Traditional fixed-size chunking with overlap"""
        
        chunks = []
        tokens = self.tokenizer.encode(text)
        
        start_idx = 0
        chunk_id = 0
        
        while start_idx < len(tokens):
            # Define chunk boundaries
            end_idx = min(start_idx + chunk_size, len(tokens))
            
            # Decode chunk tokens back to text
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            # Create chunk object
            chunk = Chunk(
                content=chunk_text,
                start_idx=start_idx,
                end_idx=end_idx,
                tokens=len(chunk_tokens),
                metadata={
                    'chunk_id': chunk_id,
                    'chunking_method': 'fixed_size',
                    'overlap': overlap
                }
            )
            
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start_idx = max(start_idx + chunk_size - overlap, start_idx + 1)
            chunk_id += 1
            
            if start_idx >= len(tokens):
                break
        
        return chunks
    
    def sentence_aware_chunking(self, text: str, max_chunk_size: int = 512) -> List[Chunk]:
        """Chunking that respects sentence boundaries"""
        
        # Split into sentences
        sentences = sent_tokenize(text)
        chunks = []
        
        current_chunk = ""
        current_tokens = 0
        start_char_idx = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence would exceed max size, create a chunk
            if current_tokens + sentence_tokens > max_chunk_size and current_chunk:
                
                chunk = Chunk(
                    content=current_chunk.strip(),
                    start_idx=start_char_idx,
                    end_idx=start_char_idx + len(current_chunk),
                    tokens=current_tokens,
                    metadata={
                        'chunk_id': chunk_id,
                        'chunking_method': 'sentence_aware',
                        'num_sentences': len(sent_tokenize(current_chunk))
                    }
                )
                
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk = sentence
                current_tokens = sentence_tokens
                start_char_idx = text.find(sentence, start_char_idx)
                chunk_id += 1
                
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                    current_tokens += sentence_tokens
                else:
                    current_chunk = sentence
                    current_tokens = sentence_tokens
                    start_char_idx = text.find(sentence, start_char_idx)
        
        # Add final chunk
        if current_chunk:
            chunk = Chunk(
                content=current_chunk.strip(),
                start_idx=start_char_idx,
                end_idx=start_char_idx + len(current_chunk),
                tokens=current_tokens,
                metadata={
                    'chunk_id': chunk_id,
                    'chunking_method': 'sentence_aware',
                    'num_sentences': len(sent_tokenize(current_chunk))
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def semantic_chunking(self, text: str, similarity_threshold: float = 0.8,
                         max_chunk_size: int = 512) -> List[Chunk]:
        """Semantic chunking based on sentence similarity"""
        
        if not self.nlp:
            logger.warning("spaCy not available. Falling back to sentence-aware chunking.")
            return self.sentence_aware_chunking(text, max_chunk_size)
        
        # Process text with spaCy
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        if len(sentences) < 2:
            return self.sentence_aware_chunking(text, max_chunk_size)
        
        # Get sentence embeddings (simplified - in practice use embedding model)
        sentence_embeddings = []
        for sentence in sentences:
            # This is a simplified version - use proper sentence embeddings
            tokens = [token.vector for token in self.nlp(sentence) if token.has_vector]
            if tokens:
                sentence_embeddings.append(np.mean(tokens, axis=0))
            else:
                sentence_embeddings.append(np.zeros(300))  # Default spacy vector dim
        
        # Group sentences by semantic similarity
        chunks = []
        current_chunk = [sentences[0]]
        current_embeddings = [sentence_embeddings[0]]
        
        for i in range(1, len(sentences)):
            # Calculate similarity with current chunk
            current_chunk_embedding = np.mean(current_embeddings, axis=0)
            similarity = cosine_similarity([current_chunk_embedding], 
                                        [sentence_embeddings[i]])[0][0]
            
            # Check if we should continue current chunk or start new one
            chunk_text = " ".join(current_chunk + [sentences[i]])
            chunk_tokens = self.count_tokens(chunk_text)
            
            if (similarity > similarity_threshold and 
                chunk_tokens <= max_chunk_size):
                # Add to current chunk
                current_chunk.append(sentences[i])
                current_embeddings.append(sentence_embeddings[i])
            else:
                # Create chunk and start new one
                chunk_content = " ".join(current_chunk)
                
                chunk = Chunk(
                    content=chunk_content,
                    start_idx=text.find(chunk_content),
                    end_idx=text.find(chunk_content) + len(chunk_content),
                    tokens=self.count_tokens(chunk_content),
                    metadata={
                        'chunk_id': len(chunks),
                        'chunking_method': 'semantic',
                        'avg_similarity': float(np.mean([
                            cosine_similarity([current_embeddings[0]], [emb])[0][0] 
                            for emb in current_embeddings[1:]
                        ])) if len(current_embeddings) > 1 else 1.0,
                        'num_sentences': len(current_chunk)
                    }
                )
                
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk = [sentences[i]]
                current_embeddings = [sentence_embeddings[i]]
        
        # Add final chunk
        if current_chunk:
            chunk_content = " ".join(current_chunk)
            
            chunk = Chunk(
                content=chunk_content,
                start_idx=text.find(chunk_content),
                end_idx=text.find(chunk_content) + len(chunk_content),
                tokens=self.count_tokens(chunk_content),
                metadata={
                    'chunk_id': len(chunks),
                    'chunking_method': 'semantic',
                    'num_sentences': len(current_chunk)
                }
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def hierarchical_chunking(self, text: str) -> Dict[str, List[Chunk]]:
        """Hierarchical chunking at multiple granularities"""
        
        # Document level (whole text)
        doc_chunk = Chunk(
            content=text,
            start_idx=0,
            end_idx=len(text),
            tokens=self.count_tokens(text),
            metadata={
                'level': 'document',
                'chunking_method': 'hierarchical'
            }
        )
        
        # Paragraph level
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        paragraph_chunks = []
        
        for i, paragraph in enumerate(paragraphs):
            chunk = Chunk(
                content=paragraph,
                start_idx=text.find(paragraph),
                end_idx=text.find(paragraph) + len(paragraph),
                tokens=self.count_tokens(paragraph),
                metadata={
                    'level': 'paragraph',
                    'paragraph_id': i,
                    'chunking_method': 'hierarchical'
                }
            )
            paragraph_chunks.append(chunk)
        
        # Sentence level
        sentence_chunks = []
        sentences = sent_tokenize(text)
        
        for i, sentence in enumerate(sentences):
            chunk = Chunk(
                content=sentence,
                start_idx=text.find(sentence),
                end_idx=text.find(sentence) + len(sentence),
                tokens=self.count_tokens(sentence),
                metadata={
                    'level': 'sentence',
                    'sentence_id': i,
                    'chunking_method': 'hierarchical'
                }
            )
            sentence_chunks.append(chunk)
        
        return {
            'document': [doc_chunk],
            'paragraphs': paragraph_chunks,
            'sentences': sentence_chunks
        }
    
    def compare_chunking_strategies(self, text: str) -> Dict:
        """Compare different chunking strategies"""
        
        print("📊 CHUNKING STRATEGIES COMPARISON")
        print("="*40)
        
        strategies = {
            'Fixed Size': self.fixed_size_chunking(text, chunk_size=300, overlap=50),
            'Sentence Aware': self.sentence_aware_chunking(text, max_chunk_size=300),
            'Semantic': self.semantic_chunking(text, similarity_threshold=0.7, max_chunk_size=300)
        }
        
        # Analysis
        comparison = {}
        
        for strategy_name, chunks in strategies.items():
            avg_tokens = np.mean([chunk.tokens for chunk in chunks])
            std_tokens = np.std([chunk.tokens for chunk in chunks])
            
            comparison[strategy_name] = {
                'num_chunks': len(chunks),
                'avg_tokens': avg_tokens,
                'std_tokens': std_tokens,
                'token_distribution': [chunk.tokens for chunk in chunks]
            }
            
            print(f"\n{strategy_name}:")
            print(f"  Chunks: {len(chunks)}")
            print(f"  Avg tokens: {avg_tokens:.1f} (±{std_tokens:.1f})")
            print(f"  Token range: [{min(chunk.tokens for chunk in chunks)}, {max(chunk.tokens for chunk in chunks)}]")
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (strategy_name, chunks) in enumerate(strategies.items()):
            token_counts = [chunk.tokens for chunk in chunks]
            
            axes[i].hist(token_counts, bins=10, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{strategy_name}\n({len(chunks)} chunks)')
            axes[i].set_xlabel('Tokens per Chunk')
            axes[i].set_ylabel('Frequency')
            axes[i].axvline(np.mean(token_counts), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(token_counts):.1f}')
            axes[i].legend()
        
        plt.tight_layout()
        plt.show()
        
        return comparison

# Demonstrate chunking strategies
def demonstrate_chunking():
    """Demonstrate different chunking approaches"""
    
    # Sample technical document about transformers
    sample_text = """
    The Transformer architecture has revolutionized natural language processing through its innovative attention mechanism. Unlike recurrent neural networks that process sequences sequentially, Transformers can process all positions in parallel, leading to significant computational efficiency gains.
    
    The core innovation lies in the self-attention mechanism, which allows each position in the sequence to attend to all other positions. Mathematically, this is computed as Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V, where Q, K, and V represent query, key, and value matrices respectively.
    
    Multi-head attention extends this concept by running multiple attention mechanisms in parallel, each focusing on different representation subspaces. This allows the model to jointly attend to information from different representation spaces at different positions.
    
    Position encoding is crucial because the attention mechanism is inherently permutation-invariant. The original Transformer uses sinusoidal position encodings, but many variants have explored learnable positional embeddings and relative position encoding schemes.
    
    The encoder-decoder architecture consists of stacked layers, each containing multi-head attention and position-wise feed-forward networks. Residual connections and layer normalization are applied around each sub-layer to facilitate training of deep networks.
    
    Pre-training objectives like masked language modeling (BERT) and autoregressive generation (GPT) have shown that Transformers can learn rich representations from large unlabeled text corpora, leading to significant improvements on downstream tasks.
    """
    
    chunker = AdvancedChunker()
    results = chunker.compare_chunking_strategies(sample_text)
    
    return results

# Run chunking demonstration
chunking_results = demonstrate_chunking()
```

---

## 3. Embedding Models & Vector Stores {#embeddings-vectors}

### Advanced Embedding and Indexing Strategies

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional
import pickle
import json
from dataclasses import asdict
import time

class AdvancedVectorStore:
    """Advanced vector store with multiple indexing strategies"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Different FAISS index types
        self.indexes = {}
        self.documents = []
        self.embeddings = None
        
        print(f"🔧 Vector Store initialized with {embedding_model}")
        print(f"   Embedding dimension: {self.dimension}")
    
    def create_indexes(self, num_documents: int):
        """Create different types of FAISS indexes"""
        
        print(f"\n🏗️ Creating FAISS indexes for {num_documents} documents")
        
        # 1. Flat (exact search) - good for small datasets
        self.indexes['flat'] = faiss.IndexFlatIP(self.dimension)  # Inner product
        
        # 2. IVF (Inverted File) - good for medium datasets
        nlist = min(int(np.sqrt(num_documents)), 100)  # Number of clusters
        quantizer = faiss.IndexFlatIP(self.dimension)
        self.indexes['ivf'] = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        self.indexes['ivf'].nprobe = min(10, nlist)  # Number of clusters to search
        
        # 3. HNSW (Hierarchical Navigable Small World) - good for large datasets
        self.indexes['hnsw'] = faiss.IndexHNSWFlat(self.dimension, 32)  # M = 32
        self.indexes['hnsw'].hnsw.efConstruction = 200  # Construction time quality
        self.indexes['hnsw'].hnsw.efSearch = 100  # Search time quality
        
        # 4. Product Quantization for memory efficiency
        m = 8  # Number of subquantizers
        nbits = 8  # Bits per subquantizer
        self.indexes['pq'] = faiss.IndexPQ(self.dimension, m, nbits)
        
        print(f"   Created indexes: {list(self.indexes.keys())}")
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents to all indexes"""
        
        if metadata is None:
            metadata = [{'id': i} for i in range(len(documents))]
        
        print(f"📄 Processing {len(documents)} documents...")
        
        # Generate embeddings
        start_time = time.time()
        embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
        embedding_time = time.time() - start_time
        
        print(f"   Embedding time: {embedding_time:.2f} seconds")
        print(f"   Speed: {len(documents)/embedding_time:.1f} docs/second")
        
        # Normalize embeddings for cosine similarity (inner product after normalization)
        faiss.normalize_L2(embeddings)
        
        # Store documents and embeddings
        self.documents.extend([(doc, meta) for doc, meta in zip(documents, metadata)])
        
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        # Create indexes if first time
        if not self.indexes:
            self.create_indexes(len(self.documents))
        
        # Add to indexes
        print("🔍 Adding to indexes...")
        
        for index_name, index in self.indexes.items():
            start_time = time.time()
            
            if index_name == 'ivf' and not index.is_trained:
                # IVF needs training
                print(f"   Training IVF index...")
                index.train(embeddings.astype('float32'))
            
            index.add(embeddings.astype('float32'))
            add_time = time.time() - start_time
            
            print(f"   {index_name}: {add_time:.2f}s, {len(self.documents)} total docs")
    
    def search_comparative(self, query: str, k: int = 5) -> Dict:
        """Compare search performance across different indexes"""
        
        # Get query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        results = {}
        
        print(f"🔎 Searching for: '{query}' (top-{k})")
        print("="*50)
        
        for index_name, index in self.indexes.items():
            start_time = time.time()
            
            try:
                scores, indices = index.search(query_embedding.astype('float32'), k)
                search_time = time.time() - start_time
                
                # Get documents
                retrieved_docs = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx != -1:  # Valid index
                        doc, metadata = self.documents[idx]
                        retrieved_docs.append({
                            'document': doc[:200] + "..." if len(doc) > 200 else doc,
                            'score': float(score),
                            'metadata': metadata,
                            'index': int(idx)
                        })
                
                results[index_name] = {
                    'documents': retrieved_docs,
                    'search_time': search_time,
                    'index_size': index.ntotal
                }
                
                print(f"\n{index_name.upper()} (Time: {search_time*1000:.1f}ms)")
                print("-" * 20)
                for i, doc_info in enumerate(retrieved_docs):
                    print(f"{i+1}. Score: {doc_info['score']:.4f}")
                    print(f"   {doc_info['document']}")
                
            except Exception as e:
                print(f"\n{index_name.upper()}: Error - {str(e)}")
                results[index_name] = {'error': str(e)}
        
        return results
    
    def benchmark_search_performance(self, queries: List[str], k: int = 5):
        """Benchmark search performance across indexes"""
        
        print("⚡ SEARCH PERFORMANCE BENCHMARK")
        print("="*35)
        
        performance = {index_name: [] for index_name in self.indexes.keys()}
        
        for query in queries:
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            for index_name, index in self.indexes.items():
                start_time = time.time()
                try:
                    scores, indices = index.search(query_embedding.astype('float32'), k)
                    search_time = time.time() - start_time
                    performance[index_name].append(search_time * 1000)  # Convert to ms
                except:
                    performance[index_name].append(float('inf'))
        
        # Calculate statistics
        stats = {}
        for index_name, times in performance.items():
            valid_times = [t for t in times if t != float('inf')]
            if valid_times:
                stats[index_name] = {
                    'mean_ms': np.mean(valid_times),
                    'std_ms': np.std(valid_times),
                    'min_ms': np.min(valid_times),
                    'max_ms': np.max(valid_times),
                    'success_rate': len(valid_times) / len(times)
                }
        
        # Display results
        print(f"\nBenchmark Results ({len(queries)} queries):")
        print("-" * 40)
        for index_name, stat in stats.items():
            print(f"{index_name.upper()}:")
            print(f"  Mean: {stat['mean_ms']:.2f}ms (±{stat['std_ms']:.2f})")
            print(f"  Range: [{stat['min_ms']:.2f}, {stat['max_ms']:.2f}]ms")
            print(f"  Success Rate: {stat['success_rate']*100:.1f}%")
        
        # Visualization
        plt.figure(figsize=(12, 6))
        
        # Performance comparison
        plt.subplot(1, 2, 1)
        index_names = list(stats.keys())
        mean_times = [stats[name]['mean_ms'] for name in index_names]
        std_times = [stats[name]['std_ms'] for name in index_names]
        
        bars = plt.bar(index_names, mean_times, yerr=std_times, 
                      capsize=5, alpha=0.8, color=['blue', 'green', 'red', 'orange'])
        plt.ylabel('Search Time (ms)')
        plt.title('Index Performance Comparison')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, time in zip(bars, mean_times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{time:.1f}ms', ha='center', va='bottom')
        
        # Time distribution
        plt.subplot(1, 2, 2)
        for index_name, times in performance.items():
            valid_times = [t for t in times if t != float('inf')]
            if valid_times:
                plt.hist(valid_times, alpha=0.6, label=index_name, bins=10)
        
        plt.xlabel('Search Time (ms)')
        plt.ylabel('Frequency')
        plt.title('Search Time Distribution')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return stats
    
    def analyze_embedding_quality(self, sample_queries: List[str], 
                                sample_documents: List[str]):
        """Analyze embedding quality and similarity patterns"""
        
        print("🧪 EMBEDDING QUALITY ANALYSIS")
        print("="*32)
        
        # Get embeddings
        query_embeddings = self.embedding_model.encode(sample_queries)
        doc_embeddings = self.embedding_model.encode(sample_documents)
        
        # Normalize
        faiss.normalize_L2(query_embeddings)
        faiss.normalize_L2(doc_embeddings)
        
        # Calculate similarity matrix
        similarity_matrix = np.dot(query_embeddings, doc_embeddings.T)
        
        # Visualize similarity matrix
        plt.figure(figsize=(12, 8))
        sns.heatmap(similarity_matrix, 
                   xticklabels=[f"Doc{i+1}" for i in range(len(sample_documents))],
                   yticklabels=[f"Q{i+1}" for i in range(len(sample_queries))],
                   annot=True, fmt='.3f', cmap='viridis')
        plt.title('Query-Document Similarity Matrix')
        plt.xlabel('Documents')
        plt.ylabel('Queries')
        plt.tight_layout()
        plt.show()
        
        # Analyze similarity distribution
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(similarity_matrix.flatten(), bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.title('Similarity Distribution')
        plt.axvline(np.mean(similarity_matrix), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(similarity_matrix):.3f}')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        max_similarities = np.max(similarity_matrix, axis=1)
        plt.bar(range(len(sample_queries)), max_similarities, alpha=0.8)
        plt.xlabel('Query Index')
        plt.ylabel('Best Match Similarity')
        plt.title('Best Match per Query')
        plt.xticks(range(len(sample_queries)), 
                  [f'Q{i+1}' for i in range(len(sample_queries))])
        
        plt.tight_layout()
        plt.show()
        
        print(f"Similarity Statistics:")
        print(f"  Mean: {np.mean(similarity_matrix):.4f}")
        print(f"  Std:  {np.std(similarity_matrix):.4f}")
        print(f"  Min:  {np.min(similarity_matrix):.4f}")
        print(f"  Max:  {np.max(similarity_matrix):.4f}")
        
        return similarity_matrix

# Demonstrate vector store capabilities
def demonstrate_vector_store():
    """Demonstrate advanced vector store capabilities"""
    
    # Sample documents about AI/ML
    sample_documents = [
        "Transformers use self-attention mechanisms to process sequences in parallel, revolutionizing NLP.",
        "BERT uses bidirectional encoding to understand context from both directions in text.",
        "GPT models are autoregressive language models that generate text token by token.",
        "Attention mechanisms allow models to focus on relevant parts of the input sequence.",
        "Pre-training on large corpora helps models learn general language representations.",
        "Fine-tuning adapts pre-trained models to specific downstream tasks efficiently.",
        "Multi-head attention computes attention in parallel across different representation subspaces.",
        "Positional encoding injects sequence order information into transformer models.",
        "RAG combines retrieval and generation for factually grounded text generation.",
        "Vector databases enable efficient similarity search for large document collections.",
        "Embedding models convert text into high-dimensional vector representations.",
        "Cosine similarity measures the angle between vectors in embedding space.",
        "FAISS provides efficient algorithms for large-scale similarity search and clustering.",
        "Dense passage retrieval uses learned embeddings for information retrieval.",
        "Cross-encoders rerank retrieved passages for improved relevance.",
        "Semantic search finds relevant content based on meaning rather than keywords.",
        "Neural information retrieval leverages deep learning for search applications.",
        "Contrastive learning improves embedding quality through positive and negative examples.",
        "In-batch negatives provide training signal for dense retrieval models.",
        "Hard negative mining improves retrieval model training effectiveness."
    ]
    
    sample_queries = [
        "How do attention mechanisms work in transformers?",
        "What is the difference between BERT and GPT?",
        "How does retrieval augmented generation work?",
        "What are vector databases used for?",
        "How do you measure similarity between embeddings?"
    ]
    
    # Initialize vector store
    vector_store = AdvancedVectorStore()
    
    # Add documents
    vector_store.add_documents(sample_documents)
    
    # Demonstrate search capabilities
    search_results = vector_store.search_comparative(
        "How do transformer attention mechanisms process sequences?", k=3
    )
    
    # Benchmark performance
    performance_stats = vector_store.benchmark_search_performance(sample_queries, k=5)
    
    # Analyze embedding quality
    similarity_analysis = vector_store.analyze_embedding_quality(
        sample_queries[:3], sample_documents[:10]
    )
    
    return vector_store, search_results, performance_stats

# Run vector store demonstration
print("🚀 Vector Store Demonstration")
vector_store, search_results, perf_stats = demonstrate_vector_store()
```

---

## 4. Retrieval Strategies {#retrieval-strategies}

### Hybrid Search and Advanced Retrieval

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import re
from collections import Counter

@dataclass
class RetrievalResult:
    """Structured retrieval result"""
    document: str
    score: float
    rank: int
    metadata: Dict
    retrieval_method: str

class HybridRetriever:
    """Advanced hybrid retrieval combining dense and sparse methods"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Dense retrieval
        self.embedding_model = SentenceTransformer(embedding_model)
        self.dense_index = None
        
        # Sparse retrieval (TF-IDF/BM25)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.tfidf_matrix = None
        
        # Documents and metadata
        self.documents = []
        self.document_embeddings = None
        
        print("🔍 Hybrid Retriever initialized")
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents and build both dense and sparse indexes"""
        
        if metadata is None:
            metadata = [{'id': i} for i in range(len(documents))]
        
        self.documents = [(doc, meta) for doc, meta in zip(documents, metadata)]
        
        print(f"📚 Indexing {len(documents)} documents...")
        
        # Build dense index
        print("  Building dense embeddings...")
        embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.dense_index = faiss.IndexFlatIP(dimension)
        self.dense_index.add(embeddings.astype('float32'))
        self.document_embeddings = embeddings
        
        # Build sparse index
        print("  Building sparse TF-IDF index...")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        
        print(f"✅ Indexing complete!")
        print(f"   Dense index: {dimension}D embeddings")
        print(f"   Sparse index: {self.tfidf_matrix.shape[1]} TF-IDF features")
    
    def dense_search(self, query: str, k: int = 10) -> List[RetrievalResult]:
        """Dense semantic search using embeddings"""
        
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.dense_index.search(query_embedding.astype('float32'), k)
        
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:
                doc, metadata = self.documents[idx]
                results.append(RetrievalResult(
                    document=doc,
                    score=float(score),
                    rank=rank,
                    metadata=metadata,
                    retrieval_method='dense'
                ))
        
        return results
    
    def sparse_search(self, query: str, k: int = 10) -> List[RetrievalResult]:
        """Sparse keyword-based search using TF-IDF"""
        
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            doc, metadata = self.documents[idx]
            results.append(RetrievalResult(
                document=doc,
                score=float(similarities[idx]),
                rank=rank,
                metadata=metadata,
                retrieval_method='sparse'
            ))
        
        return results
    
    def hybrid_search(self, query: str, k: int = 10, 
                     dense_weight: float = 0.7,
                     sparse_weight: float = 0.3) -> List[RetrievalResult]:
        """Hybrid search combining dense and sparse retrieval"""
        
        # Get results from both methods
        dense_results = self.dense_search(query, k=k*2)  # Get more to have overlap
        sparse_results = self.sparse_search(query, k=k*2)
        
        # Normalize scores to [0, 1] range
        if dense_results:
            max_dense_score = max(r.score for r in dense_results)
            for result in dense_results:
                result.score = result.score / max_dense_score if max_dense_score > 0 else 0
        
        if sparse_results:
            max_sparse_score = max(r.score for r in sparse_results)
            for result in sparse_results:
                result.score = result.score / max_sparse_score if max_sparse_score > 0 else 0
        
        # Combine scores
        combined_scores = {}
        
        # Add dense scores
        for result in dense_results:
            doc_id = id(result.document)  # Use document object id as key
            combined_scores[doc_id] = {
                'dense_score': result.score * dense_weight,
                'sparse_score': 0,
                'document': result.document,
                'metadata': result.metadata
            }
        
        # Add sparse scores
        for result in sparse_results:
            doc_id = id(result.document)
            if doc_id in combined_scores:
                combined_scores[doc_id]['sparse_score'] = result.score * sparse_weight
            else:
                combined_scores[doc_id] = {
                    'dense_score': 0,
                    'sparse_score': result.score * sparse_weight,
                    'document': result.document,
                    'metadata': result.metadata
                }
        
        # Calculate final scores and rank
        final_results = []
        for doc_id, scores in combined_scores.items():
            final_score = scores['dense_score'] + scores['sparse_score']
            final_results.append(RetrievalResult(
                document=scores['document'],
                score=final_score,
                rank=0,  # Will be updated after sorting
                metadata=scores['metadata'],
                retrieval_method='hybrid'
            ))
        
        # Sort by final score and update ranks
        final_results.sort(key=lambda x: x.score, reverse=True)
        for rank, result in enumerate(final_results[:k]):
            result.rank = rank
        
        return final_results[:k]
    
    def query_expansion(self, query: str, expansion_terms: int = 3) -> str:
        """Simple query expansion using word co-occurrence"""
        
        # Simple approach: find most similar documents and extract key terms
        dense_results = self.dense_search(query, k=5)
        
        # Extract terms from top documents
        all_terms = []
        for result in dense_results:
            # Simple tokenization and filtering
            terms = re.findall(r'\b\w+\b', result.document.lower())
            terms = [term for term in terms if len(term) > 2]  # Filter short terms
            all_terms.extend(terms)
        
        # Count term frequency
        term_freq = Counter(all_terms)
        
        # Get query terms
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        
        # Find expansion terms (high frequency, not in query)
        expansion_candidates = []
        for term, freq in term_freq.most_common():
            if term not in query_terms and len(expansion_candidates) < expansion_terms:
                expansion_candidates.append(term)
        
        # Combine original query with expansion terms
        expanded_query = query + " " + " ".join(expansion_candidates)
        
        return expanded_query
    
    def rerank_results(self, results: List[RetrievalResult], query: str,
                      rerank_model: str = None) -> List[RetrievalResult]:
        """Rerank results using a cross-encoder (simplified version)"""
        
        # For this demo, we'll use a simple relevance scoring
        # In practice, you'd use a cross-encoder model
        
        reranked_results = []
        
        for result in results:
            # Simple relevance scoring based on query term overlap
            query_terms = set(re.findall(r'\b\w+\b', query.lower()))
            doc_terms = set(re.findall(r'\b\w+\b', result.document.lower()))
            
            overlap_score = len(query_terms.intersection(doc_terms)) / len(query_terms)
            
            # Combine with original score
            new_score = 0.7 * result.score + 0.3 * overlap_score
            
            reranked_result = RetrievalResult(
                document=result.document,
                score=new_score,
                rank=result.rank,
                metadata=result.metadata,
                retrieval_method=result.retrieval_method + "_reranked"
            )
            
            reranked_results.append(reranked_result)
        
        # Re-sort and update ranks
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        for rank, result in enumerate(reranked_results):
            result.rank = rank
        
        return reranked_results
    
    def compare_retrieval_methods(self, queries: List[str], k: int = 5):
        """Compare different retrieval methods"""
        
        print("🔬 RETRIEVAL METHODS COMPARISON")
        print("="*35)
        
        methods = {
            'Dense': lambda q: self.dense_search(q, k),
            'Sparse': lambda q: self.sparse_search(q, k),
            'Hybrid': lambda q: self.hybrid_search(q, k),
        }
        
        for query in queries:
            print(f"\n📝 Query: '{query}'")
            print("-" * 50)
            
            # Expand query
            expanded_query = self.query_expansion(query)
            if expanded_query != query:
                print(f"💡 Expanded: '{expanded_query}'")
            
            for method_name, method_func in methods.items():
                print(f"\n{method_name} Results:")
                
                results = method_func(query)
                
                for i, result in enumerate(results[:3], 1):
                    doc_preview = result.document[:100] + "..." if len(result.document) > 100 else result.document
                    print(f"  {i}. Score: {result.score:.4f}")
                    print(f"     {doc_preview}")
    
    def analyze_retrieval_overlap(self, query: str, k: int = 10):
        """Analyze overlap between different retrieval methods"""
        
        dense_results = self.dense_search(query, k)
        sparse_results = self.sparse_search(query, k)
        hybrid_results = self.hybrid_search(query, k)
        
        # Get document sets
        dense_docs = set(result.document for result in dense_results)
        sparse_docs = set(result.document for result in sparse_results)
        hybrid_docs = set(result.document for result in hybrid_results)
        
        # Calculate overlaps
        dense_sparse_overlap = len(dense_docs.intersection(sparse_docs))
        dense_hybrid_overlap = len(dense_docs.intersection(hybrid_docs))
        sparse_hybrid_overlap = len(sparse_docs.intersection(hybrid_docs))
        
        print(f"📊 RETRIEVAL OVERLAP ANALYSIS")
        print(f"Query: '{query}'")
        print("-" * 40)
        print(f"Dense ∩ Sparse: {dense_sparse_overlap}/{k} ({dense_sparse_overlap/k*100:.1f}%)")
        print(f"Dense ∩ Hybrid: {dense_hybrid_overlap}/{k} ({dense_hybrid_overlap/k*100:.1f}%)")
        print(f"Sparse ∩ Hybrid: {sparse_hybrid_overlap}/{k} ({sparse_hybrid_overlap/k*100:.1f}%)")
        
        # Visualization
        from matplotlib_venn import venn2, venn3
        
        plt.figure(figsize=(12, 4))
        
        # Dense vs Sparse
        plt.subplot(1, 3, 1)
        venn2([dense_docs, sparse_docs], set_labels=['Dense', 'Sparse'])
        plt.title('Dense vs Sparse')
        
        # All three methods
        plt.subplot(1, 3, 2)
        venn3([dense_docs, sparse_docs, hybrid_docs], 
              set_labels=['Dense', 'Sparse', 'Hybrid'])
        plt.title('All Methods')
        
        # Score correlation
        plt.subplot(1, 3, 3)
        
        # Find common documents and compare scores
        common_docs = dense_docs.intersection(sparse_docs)
        if len(common_docs) > 1:
            dense_scores = []
            sparse_scores = []
            
            for doc in common_docs:
                # Find scores for this document
                dense_score = next((r.score for r in dense_results if r.document == doc), 0)
                sparse_score = next((r.score for r in sparse_results if r.document == doc), 0)
                
                dense_scores.append(dense_score)
                sparse_scores.append(sparse_score)
            
            plt.scatter(dense_scores, sparse_scores, alpha=0.7)
            plt.xlabel('Dense Score')
            plt.ylabel('Sparse Score')
            plt.title('Score Correlation')
            
            # Calculate correlation
            correlation = np.corrcoef(dense_scores, sparse_scores)[0, 1]
            plt.text(0.1, 0.9, f'r = {correlation:.3f}', transform=plt.gca().transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        return {
            'dense_sparse_overlap': dense_sparse_overlap,
            'dense_hybrid_overlap': dense_hybrid_overlap,
            'sparse_hybrid_overlap': sparse_hybrid_overlap
        }

# Demonstrate hybrid retrieval
def demonstrate_hybrid_retrieval():
    """Demonstrate hybrid retrieval capabilities"""
    
    # Expanded document collection with diverse content
    documents = [
        # Technical AI/ML content
        "Transformers revolutionized NLP through self-attention mechanisms that process sequences in parallel.",
        "BERT uses bidirectional encoder representations to understand context from both directions.",
        "GPT models employ autoregressive generation to produce coherent and contextually relevant text.",
        "Attention mechanisms allow neural networks to focus on relevant parts of input sequences dynamically.",
        "Pre-training on large corpora enables models to learn rich language representations without supervision.",
        
        # RAG specific content
        "Retrieval-augmented generation combines information retrieval with neural text generation.",
        "Dense passage retrieval uses learned embeddings to find relevant documents for queries.",
        "Vector databases enable efficient similarity search across millions of document embeddings.",
        "Hybrid search methods combine keyword matching with semantic similarity for better results.",
        "Cross-encoders can rerank retrieved passages to improve relevance and reduce noise.",
        
        # Mathematical concepts
        "Cosine similarity measures the angle between vectors in high-dimensional embedding space.",
        "The attention formula Attention(Q,K,V) = softmax(QK^T/√d_k)V computes weighted representations.",
        "Matrix factorization techniques decompose large matrices into smaller, more manageable components.",
        "Gradient descent optimization iteratively updates model parameters to minimize loss functions.",
        "Backpropagation algorithms compute gradients through neural network layers efficiently.",
        
        # Applied examples
        "Question answering systems leverage retrieval to find relevant context before generating answers.",
        "Chatbots use RAG to provide factually accurate responses grounded in knowledge bases.",
        "Search engines employ both keyword matching and semantic understanding for relevance.",
        "Recommendation systems use embedding similarity to suggest relevant items to users.",
        "Document classification tasks benefit from pre-trained language model representations."
    ]
    
    queries = [
        "How does the attention mechanism work in transformers?",
        "What is retrieval augmented generation?",
        "How do you calculate cosine similarity between embeddings?",
        "What are the applications of RAG systems?",
        "How does BERT differ from GPT models?"
    ]
    
    # Initialize hybrid retriever
    retriever = HybridRetriever()
    retriever.add_documents(documents)
    
    # Compare retrieval methods
    retriever.compare_retrieval_methods(queries[:3], k=5)
    
    # Analyze overlap
    overlap_analysis = retriever.analyze_retrieval_overlap(queries[0], k=8)
    
    return retriever, overlap_analysis

# Run hybrid retrieval demonstration
print("🔍 Hybrid Retrieval Demonstration")
hybrid_retriever, overlap_analysis = demonstrate_hybrid_retrieval()
```

---

*This RAG notebook continues with sections 5-9 covering Generation & Context Integration, Advanced RAG Techniques, Evaluation Framework, Production Implementation, and Interview Tips. Would you like me to continue with the remaining sections, or would you prefer to see the third notebook on Agentic AI with AutoGen first?*

