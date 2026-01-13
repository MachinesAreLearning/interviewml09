# Quick Reference Cheatsheet

## VP-Level Answer Patterns

### Pattern 1: Technical Questions
```
1. Start with WHY (business context)
2. Explain WHAT (the concept)
3. Describe HOW (implementation)
4. Discuss TRADE-OFFS
5. Share EXPERIENCE (your real example)
```

### Pattern 2: Behavioral Questions (STAR-L)
```
S - Situation (context, 2 sentences)
T - Task (your responsibility)
A - Action (what YOU did, specific steps)
R - Result (metrics, outcomes)
L - Learning (what you'd do differently)
```

### Pattern 3: Design Questions
```
1. CLARIFY requirements
2. DRAW high-level architecture
3. DEEP DIVE on critical components
4. DISCUSS trade-offs
5. ADDRESS scale/reliability/governance
```

---

## Key Frameworks to Remember

### Model Evolution
```
Linear/Logistic → Trees → RNNs → LSTM → Transformers
     ↓            ↓       ↓        ↓         ↓
  Baseline    Non-linear Sequence  Memory  Attention
```

### Feature Store Architecture
```
┌─────────────────────────────────────────┐
│   Data Sources → Feature Pipeline →     │
│   ┌─────────────┐   ┌─────────────┐    │
│   │  Offline    │   │   Online    │    │
│   │  Store      │   │   Store     │    │
│   │ (Training)  │   │ (Serving)   │    │
│   └─────────────┘   └─────────────┘    │
└─────────────────────────────────────────┘
```

### RAG Architecture
```
Query → Embed → Vector Search → Retrieve → Augment Prompt → LLM → Response
```

### Agentic AI Blueprint
```
┌─────────────────────────────────────────┐
│         Orchestration Layer             │
│  (LangGraph StateGraph / Workflow)      │
├─────────────────────────────────────────┤
│    Agent 1 → Agent 2 → Agent 3          │
├─────────────────────────────────────────┤
│         Guardrails Layer                │
│  (Input/Output validation, PII, Cost)   │
├─────────────────────────────────────────┤
│         Infrastructure                  │
│  (LLM routing, Caching, Rate limiting)  │
├─────────────────────────────────────────┤
│         Observability                   │
│  (Logging, Metrics, Tracing)            │
└─────────────────────────────────────────┘
```

---

## Key Metrics to Know

### Classification
| Metric | Formula | When to Use |
|--------|---------|-------------|
| Precision | TP / (TP + FP) | Cost of false positives high |
| Recall | TP / (TP + FN) | Cost of false negatives high |
| F1 | 2 * (P * R) / (P + R) | Balance needed |
| AUC-ROC | Area under ROC curve | Overall ranking ability |

### Drift Detection
| Metric | Threshold | Action |
|--------|-----------|--------|
| PSI < 0.1 | No drift | Monitor |
| PSI 0.1-0.25 | Moderate | Investigate |
| PSI > 0.25 | Significant | Retrain |

### System SLAs
| Metric | Real-time | Batch |
|--------|-----------|-------|
| Latency | <100ms p99 | N/A |
| Availability | 99.9% | 99% |
| Throughput | 10K+ QPS | Millions/day |

---

## Banking-Specific Considerations

### SR 11-7 Requirements
- Model inventory with risk tiering
- Independent validation
- Ongoing monitoring
- Documentation and governance
- Audit trails

### Fair Lending
- Disparate Impact Ratio ≥ 0.80
- Monitor protected classes
- Explainable decisions
- Adverse action notices

### Data Privacy
- PII encryption at rest and in transit
- Access controls and audit logs
- Data retention policies
- Right to be forgotten

---

## Technology Comparisons

### Vector Databases
| DB | Best For | Scale |
|----|----------|-------|
| ChromaDB | Prototyping | <1M |
| FAISS | Performance | 10M+ |
| Pinecone | Managed | Any |
| Weaviate | Hybrid search | Any |

### LLM Frameworks
| Framework | Strength | Use When |
|-----------|----------|----------|
| LangChain | Flexibility | Complex chains |
| LlamaIndex | Data indexing | RAG-heavy |
| LangGraph | State machines | Multi-agent |
| AutoGen | Conversations | Agent dialogue |

### Deployment Strategies
| Strategy | Risk | Speed |
|----------|------|-------|
| Blue-Green | All-or-nothing | Instant |
| Canary | Progressive | Gradual |
| Shadow | Zero | Testing only |

---

## Common Interview Mistakes to Avoid

### Technical
- ❌ Jumping to solution without understanding problem
- ❌ Not discussing trade-offs
- ❌ Ignoring regulatory/governance aspects
- ❌ Over-engineering simple problems

### Behavioral
- ❌ Using "we" instead of "I"
- ❌ No specific metrics or outcomes
- ❌ Not showing learning from failures
- ❌ Criticizing former employers

### Design
- ❌ Not clarifying requirements
- ❌ Missing scale considerations
- ❌ Ignoring monitoring/observability
- ❌ Not addressing failure modes

---

## Power Phrases for VP-Level Answers

### Showing Strategic Thinking
- "The key trade-off here is..."
- "From a business perspective..."
- "Given the regulatory constraints..."
- "At scale, this means..."

### Showing Leadership
- "I built alignment by..."
- "My approach was to..."
- "I made the decision to..."
- "What I learned was..."

### Showing Technical Depth
- "Under the hood..."
- "The reason this works is..."
- "The limitation of this approach..."
- "In production, we found..."

---

## Day-Before Checklist

- [ ] Review your 5-6 STAR stories
- [ ] Prepare 3 questions for each interviewer
- [ ] Know your numbers (team size, $ impact, metrics)
- [ ] Review JP Morgan's recent AI announcements
- [ ] Get good sleep
- [ ] Prepare technology (if virtual)
- [ ] Review this cheatsheet one more time
