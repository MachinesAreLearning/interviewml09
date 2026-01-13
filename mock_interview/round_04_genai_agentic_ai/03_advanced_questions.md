# Round 4: GenAI & Agentic AI - ADVANCED Questions

## System Design, Blueprints, Guardrails & Production

---

### Q1: Design a blueprint for an Agentic AI system for document processing.

**VP Answer:**
```
"Let me walk through a production agentic AI architecture:

┌─────────────────────────────────────────────────────────────────────────────┐
│                    AGENTIC AI SYSTEM BLUEPRINT                              │
│                    Document Processing System                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ╔═════════════════════════════════════════════════════════════════════╗   │
│  ║                     ORCHESTRATION LAYER                             ║   │
│  ║              (LangGraph StateGraph - Deterministic)                 ║   │
│  ╠═════════════════════════════════════════════════════════════════════╣   │
│  ║                                                                     ║   │
│  ║   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐        ║   │
│  ║   │ Intake  │───▶│ Classify│───▶│ Extract │───▶│ Validate│        ║   │
│  ║   │ Agent   │    │ Agent   │    │ Agent   │    │ Agent   │        ║   │
│  ║   └─────────┘    └─────────┘    └─────────┘    └─────────┘        ║   │
│  ║        │              │              │              │              ║   │
│  ║        │              │              │              ▼              ║   │
│  ║        │              │              │         ┌─────────┐        ║   │
│  ║        │              │              │         │ Route   │        ║   │
│  ║        │              │              │         │ Agent   │        ║   │
│  ║        │              │              │         └────┬────┘        ║   │
│  ║        │              │              │              │              ║   │
│  ║        │              │              │    ┌─────────┴─────────┐   ║   │
│  ║        │              │              │    ▼                   ▼   ║   │
│  ║        │              │              │ [Approve]         [Review] ║   │
│  ║        │              │              │    │                   │   ║   │
│  ║        │              │              │    ▼                   ▼   ║   │
│  ║        │              │              │ [Complete]     [Human Loop]║   │
│  ║                                                                     ║   │
│  ╚═════════════════════════════════════════════════════════════════════╝   │
│                                                                             │
│  ╔═════════════════════════════════════════════════════════════════════╗   │
│  ║                      AGENT DEFINITIONS                              ║   │
│  ╠═════════════════════════════════════════════════════════════════════╣   │
│  ║                                                                     ║   │
│  ║  INTAKE AGENT                                                       ║   │
│  ║  ├─ Tools: PDF parser, OCR, format detector                         ║   │
│  ║  ├─ Output: Parsed text, metadata, quality score                    ║   │
│  ║  └─ Guardrails: File size limit, supported formats                  ║   │
│  ║                                                                     ║   │
│  ║  CLASSIFY AGENT                                                     ║   │
│  ║  ├─ Tools: Document classifier, intent detector                     ║   │
│  ║  ├─ Output: Document type, confidence, required fields              ║   │
│  ║  └─ Guardrails: Confidence threshold, known types only              ║   │
│  ║                                                                     ║   │
│  ║  EXTRACT AGENT                                                      ║   │
│  ║  ├─ Tools: NER, regex extractor, table parser                       ║   │
│  ║  ├─ Output: Structured data (JSON schema)                           ║   │
│  ║  └─ Guardrails: Schema validation, required fields                  ║   │
│  ║                                                                     ║   │
│  ║  VALIDATE AGENT                                                     ║   │
│  ║  ├─ Tools: Business rules engine, cross-reference DB                ║   │
│  ║  ├─ Output: Validation status, issues found                         ║   │
│  ║  └─ Guardrails: All rules must pass or flag for review              ║   │
│  ║                                                                     ║   │
│  ║  ROUTE AGENT                                                        ║   │
│  ║  ├─ Tools: Decision engine, escalation logic                        ║   │
│  ║  ├─ Output: Next action (approve/review/reject)                     ║   │
│  ║  └─ Guardrails: Confidence-based escalation                         ║   │
│  ║                                                                     ║   │
│  ╚═════════════════════════════════════════════════════════════════════╝   │
│                                                                             │
│  ╔═════════════════════════════════════════════════════════════════════╗   │
│  ║                      GUARDRAILS LAYER                               ║   │
│  ╠═════════════════════════════════════════════════════════════════════╣   │
│  ║                                                                     ║   │
│  ║  INPUT VALIDATION                                                   ║   │
│  ║  ├─ PII Detection (scan before processing)                          ║   │
│  ║  ├─ Malware scan                                                    ║   │
│  ║  ├─ Size limits (max 50MB)                                          ║   │
│  ║  └─ Format whitelist                                                ║   │
│  ║                                                                     ║   │
│  ║  OUTPUT VALIDATION                                                  ║   │
│  ║  ├─ Schema compliance                                               ║   │
│  ║  ├─ Confidence thresholds                                           ║   │
│  ║  ├─ PII masking in responses                                        ║   │
│  ║  └─ Hallucination checks                                            ║   │
│  ║                                                                     ║   │
│  ║  PROCESS GUARDRAILS                                                 ║   │
│  ║  ├─ Step limit (max 15 agent turns)                                 ║   │
│  ║  ├─ Cost budget ($0.50 per document)                                ║   │
│  ║  ├─ Timeout (30 seconds per step)                                   ║   │
│  ║  └─ Retry limits (3 per agent)                                      ║   │
│  ║                                                                     ║   │
│  ╚═════════════════════════════════════════════════════════════════════╝   │
│                                                                             │
│  ╔═════════════════════════════════════════════════════════════════════╗   │
│  ║                    INFRASTRUCTURE LAYER                             ║   │
│  ╠═════════════════════════════════════════════════════════════════════╣   │
│  ║                                                                     ║   │
│  ║  LOAD BALANCING                                                     ║   │
│  ║  ├─ Round-robin across LLM providers                                ║   │
│  ║  ├─ Fallback: GPT-4 → Claude → GPT-3.5                              ║   │
│  ║  └─ Health checks, automatic failover                               ║   │
│  ║                                                                     ║   │
│  ║  CACHING                                                            ║   │
│  ║  ├─ Semantic cache (similar queries)                                ║   │
│  ║  ├─ Embedding cache                                                 ║   │
│  ║  └─ Document classification cache                                   ║   │
│  ║                                                                     ║   │
│  ║  RATE LIMITING                                                      ║   │
│  ║  ├─ Token bucket per user/tenant                                    ║   │
│  ║  ├─ Global TPM limits                                               ║   │
│  ║  └─ Priority queuing                                                ║   │
│  ║                                                                     ║   │
│  ╚═════════════════════════════════════════════════════════════════════╝   │
│                                                                             │
│  ╔═════════════════════════════════════════════════════════════════════╗   │
│  ║                     OBSERVABILITY LAYER                             ║   │
│  ╠═════════════════════════════════════════════════════════════════════╣   │
│  ║                                                                     ║   │
│  ║  TRACING: LangSmith / OpenTelemetry                                 ║   │
│  ║  ├─ Full prompt/response logging                                    ║   │
│  ║  ├─ Agent step traces                                               ║   │
│  ║  └─ Latency breakdown per step                                      ║   │
│  ║                                                                     ║   │
│  ║  METRICS: Prometheus                                                ║   │
│  ║  ├─ Documents processed/hour                                        ║   │
│  ║  ├─ Success/failure rates                                           ║   │
│  ║  ├─ Latency percentiles (p50, p95, p99)                             ║   │
│  ║  └─ Cost per document                                               ║   │
│  ║                                                                     ║   │
│  ║  AUDIT LOG: Immutable                                               ║   │
│  ║  ├─ Every agent decision logged                                     ║   │
│  ║  ├─ Full prompt history (PII redacted)                              ║   │
│  ║  └─ Retrievable for compliance                                      ║   │
│  ║                                                                     ║   │
│  ╚═════════════════════════════════════════════════════════════════════╝   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

IMPLEMENTATION CODE STRUCTURE:

from typing import TypedDict
from langgraph.graph import StateGraph, END

class DocumentState(TypedDict):
    document_id: str
    raw_content: bytes
    parsed_text: str
    doc_type: str
    extracted_data: dict
    validation_results: dict
    confidence: float
    route: str
    audit_trail: list

class DocumentProcessingSystem:
    def __init__(self):
        self.graph = self._build_graph()
        self.guardrails = GuardrailsEngine()
        self.cache = SemanticCache()

    def _build_graph(self):
        workflow = StateGraph(DocumentState)

        # Add nodes
        workflow.add_node('intake', self.intake_agent)
        workflow.add_node('classify', self.classify_agent)
        workflow.add_node('extract', self.extract_agent)
        workflow.add_node('validate', self.validate_agent)
        workflow.add_node('route', self.route_agent)
        workflow.add_node('human_review', self.human_review)
        workflow.add_node('complete', self.complete)

        # Add edges
        workflow.set_entry_point('intake')
        workflow.add_edge('intake', 'classify')
        workflow.add_edge('classify', 'extract')
        workflow.add_edge('extract', 'validate')
        workflow.add_edge('validate', 'route')
        workflow.add_conditional_edges(
            'route',
            self._routing_logic,
            {
                'approve': 'complete',
                'review': 'human_review',
            }
        )
        workflow.add_edge('human_review', 'complete')
        workflow.add_edge('complete', END)

        return workflow.compile()

    def process(self, document: bytes) -> dict:
        # Input guardrails
        self.guardrails.validate_input(document)

        # Process
        result = self.graph.invoke({
            'document_id': generate_id(),
            'raw_content': document,
            'audit_trail': []
        })

        # Output guardrails
        self.guardrails.validate_output(result)

        return result"
```

---

### Q2: How do you implement guardrails for LLM applications?

**VP Answer:**
```
"Guardrails are essential for safe, reliable LLM applications in banking:

┌─────────────────────────────────────────────────────────────────┐
│                  COMPREHENSIVE GUARDRAILS                       │
├─────────────────────────────────────────────────────────────────┤

1. INPUT GUARDRAILS
═══════════════════

class InputGuardrails:
    def __init__(self):
        self.pii_detector = PIIDetector()
        self.prompt_injection_detector = PromptInjectionDetector()

    def validate(self, user_input: str) -> tuple[bool, str]:
        # PII Detection
        pii_found = self.pii_detector.scan(user_input)
        if pii_found:
            return False, 'PII detected in input'

        # Prompt Injection Detection
        if self.prompt_injection_detector.is_suspicious(user_input):
            return False, 'Potential prompt injection detected'

        # Length limits
        if len(user_input) > 10000:
            return False, 'Input exceeds maximum length'

        # Character validation
        if not self._is_valid_charset(user_input):
            return False, 'Invalid characters in input'

        return True, 'Valid'

    def sanitize(self, user_input: str) -> str:
        # Remove/mask PII
        sanitized = self.pii_detector.mask(user_input)
        return sanitized

2. OUTPUT GUARDRAILS
════════════════════

class OutputGuardrails:
    def __init__(self):
        self.hallucination_detector = HallucinationDetector()
        self.toxicity_detector = ToxicityDetector()

    def validate(self, response: str, context: str) -> tuple[bool, str]:
        # Groundedness check
        if not self.hallucination_detector.is_grounded(response, context):
            return False, 'Response not grounded in context'

        # Toxicity check
        if self.toxicity_detector.is_toxic(response):
            return False, 'Toxic content detected'

        # PII leakage check
        if self.pii_detector.scan(response):
            return False, 'PII detected in response'

        # Schema validation (for structured outputs)
        if self.expected_schema:
            try:
                self.validate_schema(response)
            except ValidationError:
                return False, 'Response does not match expected schema'

        return True, 'Valid'

3. PROCESS GUARDRAILS
═════════════════════

class ProcessGuardrails:
    def __init__(self, config: GuardrailConfig):
        self.max_steps = config.max_steps  # 15
        self.max_cost = config.max_cost    # $0.50
        self.timeout = config.timeout       # 30s per step
        self.max_retries = config.max_retries  # 3

        self.current_steps = 0
        self.current_cost = 0

    def check_step_limit(self) -> bool:
        self.current_steps += 1
        if self.current_steps > self.max_steps:
            raise StepLimitExceeded(f'Exceeded {self.max_steps} steps')
        return True

    def check_cost(self, tokens_used: int, model: str) -> bool:
        cost = self._calculate_cost(tokens_used, model)
        self.current_cost += cost
        if self.current_cost > self.max_cost:
            raise CostLimitExceeded(f'Exceeded ${self.max_cost}')
        return True

4. HALLUCINATION PREVENTION
═══════════════════════════

class HallucinationDetector:
    def __init__(self, llm):
        self.llm = llm

    def is_grounded(self, response: str, context: str) -> bool:
        '''Check if response is supported by context'''

        prompt = f'''
        Context: {context}

        Response: {response}

        Is every claim in the response directly supported by the context?
        Answer only YES or NO.
        '''

        result = self.llm.invoke(prompt)
        return 'YES' in result.content.upper()

    def extract_claims(self, response: str) -> list[str]:
        '''Extract individual claims for verification'''
        prompt = f'''
        Extract each factual claim from this response as a separate item:
        {response}
        '''
        # Parse into list of claims
        pass

5. PROMPT INJECTION DEFENSE
═══════════════════════════

class PromptInjectionDetector:
    SUSPICIOUS_PATTERNS = [
        r'ignore (previous|all|above) instructions',
        r'disregard (previous|all) (instructions|prompts)',
        r'you are now',
        r'new instructions:',
        r'system prompt:',
        r'<\|.*\|>',  # Special tokens
    ]

    def is_suspicious(self, text: str) -> bool:
        text_lower = text.lower()
        for pattern in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, text_lower):
                return True

        # LLM-based detection for sophisticated attacks
        if self._llm_detect_injection(text):
            return True

        return False

6. CONFIDENCE-BASED ROUTING
═══════════════════════════

class ConfidenceRouter:
    def __init__(self, thresholds: dict):
        self.auto_approve_threshold = thresholds.get('auto_approve', 0.95)
        self.human_review_threshold = thresholds.get('human_review', 0.7)

    def route(self, confidence: float) -> str:
        if confidence >= self.auto_approve_threshold:
            return 'auto_approve'
        elif confidence >= self.human_review_threshold:
            return 'human_review'
        else:
            return 'reject'

PUTTING IT TOGETHER:

class GuardedLLMApplication:
    def __init__(self):
        self.input_guardrails = InputGuardrails()
        self.output_guardrails = OutputGuardrails()
        self.process_guardrails = ProcessGuardrails()

    def invoke(self, user_input: str) -> str:
        # Pre-check
        valid, msg = self.input_guardrails.validate(user_input)
        if not valid:
            return f'Input rejected: {msg}'

        # Sanitize
        sanitized_input = self.input_guardrails.sanitize(user_input)

        # Process with guardrails
        try:
            self.process_guardrails.check_step_limit()

            response = self.llm.invoke(sanitized_input)

            self.process_guardrails.check_cost(response.usage.total_tokens)

        except (StepLimitExceeded, CostLimitExceeded) as e:
            return f'Process limit exceeded: {e}'

        # Post-check
        valid, msg = self.output_guardrails.validate(response.content)
        if not valid:
            return f'Response filtered: {msg}'

        return response.content"
```

---

### Q3: How do you implement load balancing for LLM calls?

**VP Answer:**
```
"Load balancing is critical for reliability and cost optimization:

┌─────────────────────────────────────────────────────────────────┐
│                   LLM LOAD BALANCER ARCHITECTURE                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Request ─▶ ┌─────────────────────┐                            │
│             │   Load Balancer     │                            │
│             │                     │                            │
│             │  ┌───────────────┐  │                            │
│             │  │ Health Check  │  │                            │
│             │  └───────────────┘  │                            │
│             │                     │                            │
│             │  ┌───────────────┐  │                            │
│             │  │Rate Limiter   │  │                            │
│             │  └───────────────┘  │                            │
│             │                     │                            │
│             │  ┌───────────────┐  │                            │
│             │  │ Router        │  │                            │
│             │  └───────┬───────┘  │                            │
│             └──────────┼──────────┘                            │
│                        │                                        │
│         ┌──────────────┼──────────────┐                        │
│         │              │              │                        │
│         ▼              ▼              ▼                        │
│    ┌─────────┐   ┌─────────┐   ┌─────────┐                    │
│    │ OpenAI  │   │  Claude │   │  Azure  │                    │
│    │ GPT-4   │   │   3.5   │   │ OpenAI  │                    │
│    └─────────┘   └─────────┘   └─────────┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

IMPLEMENTATION:

from dataclasses import dataclass
from typing import Optional
import asyncio
import time

@dataclass
class LLMProvider:
    name: str
    client: Any
    weight: float = 1.0
    max_rpm: int = 60
    max_tpm: int = 90000
    is_healthy: bool = True
    last_error: Optional[float] = None
    error_count: int = 0

class LLMLoadBalancer:
    def __init__(self, providers: list[LLMProvider]):
        self.providers = providers
        self.request_counts = {p.name: 0 for p in providers}
        self.token_counts = {p.name: 0 for p in providers}

    async def call(self, messages: list, **kwargs) -> str:
        # Get available providers
        available = self._get_healthy_providers()

        if not available:
            raise AllProvidersUnavailable('No healthy providers')

        # Select provider
        provider = self._select_provider(available, kwargs)

        try:
            # Make request
            response = await self._call_provider(provider, messages, **kwargs)

            # Update stats
            self._update_success(provider, response)

            return response

        except Exception as e:
            # Handle failure
            self._handle_error(provider, e)

            # Retry with fallback
            return await self._retry_with_fallback(messages, provider, **kwargs)

    def _select_provider(self, providers: list, kwargs: dict) -> LLMProvider:
        '''Select provider based on strategy'''

        # Check for specific model request
        if 'model' in kwargs:
            for p in providers:
                if kwargs['model'] in p.supported_models:
                    return p

        # Weighted round-robin
        total_weight = sum(p.weight for p in providers)
        weights = [p.weight / total_weight for p in providers]

        # Consider current load
        adjusted_weights = []
        for p, w in zip(providers, weights):
            utilization = self.request_counts[p.name] / p.max_rpm
            adjusted_weights.append(w * (1 - utilization))

        # Normalize and select
        total = sum(adjusted_weights)
        adjusted_weights = [w / total for w in adjusted_weights]

        return random.choices(providers, weights=adjusted_weights)[0]

    def _get_healthy_providers(self) -> list[LLMProvider]:
        '''Return providers that are healthy and under rate limits'''

        healthy = []
        for p in self.providers:
            # Check health
            if not p.is_healthy:
                # Check if cooldown period passed
                if p.last_error and time.time() - p.last_error > 60:
                    p.is_healthy = True
                    p.error_count = 0
                else:
                    continue

            # Check rate limits
            if self.request_counts[p.name] < p.max_rpm:
                healthy.append(p)

        return healthy

    def _handle_error(self, provider: LLMProvider, error: Exception):
        '''Handle provider error'''

        provider.error_count += 1
        provider.last_error = time.time()

        # Circuit breaker: Mark unhealthy after 3 consecutive errors
        if provider.error_count >= 3:
            provider.is_healthy = False
            logger.error(f'Provider {provider.name} marked unhealthy')

    async def _retry_with_fallback(self, messages, failed_provider, **kwargs):
        '''Retry with next available provider'''

        available = [p for p in self._get_healthy_providers()
                     if p.name != failed_provider.name]

        if not available:
            raise AllProvidersUnavailable('All fallbacks exhausted')

        # Try next provider
        next_provider = self._select_provider(available, kwargs)
        return await self._call_provider(next_provider, messages, **kwargs)

COST-OPTIMIZED ROUTING:

class CostOptimizedRouter:
    '''Route based on task complexity and cost'''

    PROVIDER_COSTS = {
        'gpt-4': {'input': 0.03, 'output': 0.06},
        'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
        'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
    }

    def route(self, messages: list, task_type: str) -> str:
        '''Select model based on task complexity'''

        if task_type in ['simple_qa', 'classification']:
            return 'gpt-3.5-turbo'  # Cheap, fast

        elif task_type in ['extraction', 'summarization']:
            return 'claude-3-sonnet'  # Good balance

        elif task_type in ['reasoning', 'complex_analysis']:
            return 'gpt-4'  # Best quality

        else:
            return 'gpt-3.5-turbo'  # Default to cheap

HEALTH CHECK IMPLEMENTATION:

class HealthChecker:
    def __init__(self, providers: list[LLMProvider]):
        self.providers = providers

    async def check_all(self):
        '''Periodic health check for all providers'''

        for provider in self.providers:
            try:
                # Simple ping
                response = await provider.client.chat.completions.create(
                    model=provider.default_model,
                    messages=[{'role': 'user', 'content': 'ping'}],
                    max_tokens=1
                )
                provider.is_healthy = True
                provider.error_count = 0

            except Exception as e:
                provider.is_healthy = False
                logger.warning(f'Health check failed for {provider.name}: {e}')

    async def run_periodic(self, interval: int = 60):
        '''Run health checks periodically'''
        while True:
            await self.check_all()
            await asyncio.sleep(interval)"
```

---

### Q4: How do you handle PII in LLM applications?

**VP Answer:**
```
"PII handling is critical in banking. Multi-layer approach:

┌─────────────────────────────────────────────────────────────────┐
│                    PII PROTECTION LAYERS                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 1: Detection                                             │
│  Layer 2: Redaction/Masking                                     │
│  Layer 3: Secure Processing                                     │
│  Layer 4: Audit & Compliance                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

LAYER 1: PII DETECTION

import re
from presidio_analyzer import AnalyzerEngine

class PIIDetector:
    def __init__(self):
        self.analyzer = AnalyzerEngine()

        # Regex patterns for quick detection
        self.patterns = {
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'account_number': r'\b\d{10,12}\b',
        }

    def detect(self, text: str) -> list[dict]:
        '''Detect PII in text'''

        findings = []

        # Regex-based detection (fast)
        for pii_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, text):
                findings.append({
                    'type': pii_type,
                    'start': match.start(),
                    'end': match.end(),
                    'value': match.group(),
                    'confidence': 0.95
                })

        # Presidio for named entities (names, addresses)
        presidio_results = self.analyzer.analyze(text, language='en')
        for result in presidio_results:
            findings.append({
                'type': result.entity_type,
                'start': result.start,
                'end': result.end,
                'value': text[result.start:result.end],
                'confidence': result.score
            })

        return findings

LAYER 2: REDACTION/MASKING

class PIIRedactor:
    def __init__(self):
        self.detector = PIIDetector()
        self.placeholder_map = {}  # For reversible masking

    def redact(self, text: str, method: str = 'mask') -> str:
        '''Redact PII from text'''

        findings = self.detector.detect(text)

        # Sort by position (reverse) to maintain indices
        findings.sort(key=lambda x: x['start'], reverse=True)

        redacted = text
        for finding in findings:
            if method == 'mask':
                # Replace with asterisks
                replacement = '*' * (finding['end'] - finding['start'])
            elif method == 'type_label':
                # Replace with type label
                replacement = f'[{finding["type"].upper()}]'
            elif method == 'placeholder':
                # Reversible placeholder
                placeholder = f'__PII_{len(self.placeholder_map)}__'
                self.placeholder_map[placeholder] = finding['value']
                replacement = placeholder
            else:
                replacement = '[REDACTED]'

            redacted = redacted[:finding['start']] + replacement + redacted[finding['end']:]

        return redacted

    def restore(self, text: str) -> str:
        '''Restore PII from placeholders (for internal use only)'''
        restored = text
        for placeholder, original in self.placeholder_map.items():
            restored = restored.replace(placeholder, original)
        return restored

LAYER 3: SECURE PROCESSING

class SecureLLMProcessor:
    def __init__(self):
        self.redactor = PIIRedactor()
        self.llm = ChatOpenAI()

    def process(self, user_input: str) -> str:
        '''Process with PII protection'''

        # 1. Redact PII from input (reversible)
        redacted_input = self.redactor.redact(user_input, method='placeholder')

        # 2. Call LLM with redacted input
        response = self.llm.invoke(redacted_input)

        # 3. Check output for PII leakage
        output_pii = self.redactor.detector.detect(response.content)
        if output_pii:
            # Redact any leaked PII
            response_text = self.redactor.redact(response.content, method='type_label')
        else:
            response_text = response.content

        # 4. Restore PII in response if needed (internal use only)
        # In most cases, we DON'T restore - keep PII out of LLM responses

        return response_text

LAYER 4: AUDIT & COMPLIANCE

class PIIAuditLogger:
    def __init__(self):
        self.audit_log = []

    def log_access(self, user_id: str, action: str, pii_types: list[str]):
        '''Log PII access for compliance'''

        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'action': action,
            'pii_types': pii_types,
            'justification': self._get_justification()
        }

        self.audit_log.append(entry)

        # Also write to immutable audit store
        self._write_to_audit_store(entry)

    def generate_compliance_report(self, start_date, end_date) -> dict:
        '''Generate PII access report for compliance'''

        filtered = [
            e for e in self.audit_log
            if start_date <= e['timestamp'] <= end_date
        ]

        return {
            'period': {'start': start_date, 'end': end_date},
            'total_accesses': len(filtered),
            'by_pii_type': self._group_by_type(filtered),
            'by_user': self._group_by_user(filtered),
        }

COMPLETE PIPELINE:

class PIISecureLLMPipeline:
    def __init__(self):
        self.detector = PIIDetector()
        self.redactor = PIIRedactor()
        self.audit = PIIAuditLogger()

    def invoke(self, user_input: str, user_id: str) -> str:
        # Detect PII
        pii_found = self.detector.detect(user_input)

        if pii_found:
            # Log detection
            self.audit.log_access(
                user_id=user_id,
                action='pii_detected',
                pii_types=[p['type'] for p in pii_found]
            )

            # Redact for LLM processing
            safe_input = self.redactor.redact(user_input, method='type_label')
        else:
            safe_input = user_input

        # Process
        response = self.llm.invoke(safe_input)

        # Final PII check on output
        output_pii = self.detector.detect(response)
        if output_pii:
            logger.warning('PII detected in LLM output, redacting')
            response = self.redactor.redact(response)

        return response"
```

---

### Q5: Design a monitoring and observability system for LLM applications.

**VP Answer:**
```
"Comprehensive observability is critical for production LLM systems:

┌─────────────────────────────────────────────────────────────────┐
│               LLM OBSERVABILITY ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                     COLLECTION LAYER                      │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │                                                          │  │
│  │  Traces         Metrics          Logs          Evals     │  │
│  │  (LangSmith)    (Prometheus)     (Structured)  (Custom)  │  │
│  │                                                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    PROCESSING LAYER                       │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │                                                          │  │
│  │  Aggregation    Alerting        Analytics      Anomaly   │  │
│  │  (Time-series)  (Rules)         (Quality)     Detection  │  │
│  │                                                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   VISUALIZATION LAYER                     │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │                                                          │  │
│  │  Dashboards     Trace Explorer  Alert Console  Reports   │  │
│  │  (Grafana)      (LangSmith)     (PagerDuty)    (Custom)  │  │
│  │                                                          │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

1. TRACING (LangSmith Integration)

from langsmith import Client
from langsmith.run_helpers import traceable

client = Client()

@traceable(name='rag_query')
def process_query(query: str) -> str:
    # All sub-calls automatically traced
    docs = retrieve_documents(query)
    response = generate_response(query, docs)
    return response

# Custom metadata
@traceable(
    name='generate',
    metadata={'model': 'gpt-4', 'use_case': 'policy_qa'}
)
def generate_response(query: str, docs: list) -> str:
    return llm.invoke(format_prompt(query, docs))

2. METRICS (Prometheus)

from prometheus_client import Counter, Histogram, Gauge

# Request metrics
llm_requests_total = Counter(
    'llm_requests_total',
    'Total LLM requests',
    ['model', 'status', 'use_case']
)

llm_request_duration = Histogram(
    'llm_request_duration_seconds',
    'LLM request duration',
    ['model'],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30]
)

# Token metrics
llm_tokens_total = Counter(
    'llm_tokens_total',
    'Total tokens used',
    ['model', 'direction']  # input/output
)

# Cost metrics
llm_cost_total = Counter(
    'llm_cost_dollars',
    'Total cost in dollars',
    ['model']
)

# Quality metrics
llm_quality_score = Gauge(
    'llm_quality_score',
    'Rolling quality score',
    ['metric']  # faithfulness, relevance, etc.
)

# Retrieval metrics
retrieval_recall = Gauge(
    'retrieval_recall_at_k',
    'Retrieval recall@k',
    ['k']
)

3. STRUCTURED LOGGING

import structlog

logger = structlog.get_logger()

def log_llm_call(request, response, metadata):
    logger.info(
        'llm_call',
        request_id=metadata['request_id'],
        model=metadata['model'],
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        latency_ms=metadata['latency_ms'],
        status='success',
        user_id=metadata['user_id'],
        use_case=metadata['use_case'],
        # PII-safe: Don't log actual prompts in production
    )

4. QUALITY EVALUATION

class QualityMonitor:
    def __init__(self, sample_rate=0.05):
        self.sample_rate = sample_rate
        self.evaluator = RAGEvaluator()

    def should_evaluate(self) -> bool:
        return random.random() < self.sample_rate

    def evaluate_and_log(self, query, response, context):
        if not self.should_evaluate():
            return

        # Run evaluations
        scores = self.evaluator.evaluate(query, response, context)

        # Log to metrics
        for metric, score in scores.items():
            llm_quality_score.labels(metric=metric).set(score)

        # Log detailed results
        logger.info(
            'quality_evaluation',
            query_hash=hash(query),
            faithfulness=scores['faithfulness'],
            relevance=scores['relevance'],
            context_recall=scores['context_recall']
        )

5. ALERTING RULES

# alerts.yaml
groups:
  - name: llm_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(llm_requests_total{status="error"}[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "LLM error rate above 10%"

      - alert: HighLatency
        expr: histogram_quantile(0.95, llm_request_duration_seconds) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "LLM p95 latency above 10s"

      - alert: QualityDegradation
        expr: llm_quality_score{metric="faithfulness"} < 0.7
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Faithfulness score below threshold"

      - alert: CostSpike
        expr: rate(llm_cost_dollars[1h]) > 100
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "LLM cost exceeding $100/hour"

6. DASHBOARD PANELS

# Key dashboard sections:

OVERVIEW:
- Total requests (last 24h)
- Error rate
- Average latency
- Total cost

PERFORMANCE:
- Latency by model (p50, p95, p99)
- Throughput (requests/min)
- Token usage by model

QUALITY:
- Faithfulness score (rolling)
- Relevance score (rolling)
- User satisfaction (if collected)

RETRIEVAL:
- Recall@5, Recall@10
- Average retrieved documents
- Cache hit rate

COSTS:
- Cost by model
- Cost by use case
- Cost trend (daily/weekly)

ERRORS:
- Error rate by type
- Rate limit hits
- Timeout rate"
```

---

### Q6: How do you implement prompt versioning and A/B testing?

**VP Answer:**
```
"Prompt management is crucial for iterating on LLM applications:

┌─────────────────────────────────────────────────────────────────┐
│                  PROMPT VERSIONING SYSTEM                       │
├─────────────────────────────────────────────────────────────────┤

1. VERSION CONTROL STRUCTURE

prompts/
├── policy_qa/
│   ├── v1.0.0/
│   │   ├── system.txt
│   │   ├── user_template.txt
│   │   └── metadata.yaml
│   ├── v1.1.0/
│   │   ├── system.txt
│   │   ├── user_template.txt
│   │   └── metadata.yaml
│   └── current -> v1.1.0  # symlink
│
├── document_summary/
│   └── ...

# metadata.yaml
version: 1.1.0
created_at: 2024-01-15
author: rishi.raman
description: Added few-shot examples for edge cases
changes:
  - Added 3 few-shot examples
  - Clarified citation format
metrics:
  faithfulness: 0.92
  relevance: 0.88

2. PROMPT REGISTRY

class PromptRegistry:
    def __init__(self, storage_path: str):
        self.storage = storage_path
        self.cache = {}

    def get_prompt(self, name: str, version: str = 'current') -> dict:
        '''Get prompt by name and version'''

        if version == 'current':
            version = self._get_current_version(name)

        cache_key = f'{name}:{version}'
        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt_dir = f'{self.storage}/{name}/{version}'
        prompt = {
            'system': self._load_file(f'{prompt_dir}/system.txt'),
            'user_template': self._load_file(f'{prompt_dir}/user_template.txt'),
            'metadata': self._load_yaml(f'{prompt_dir}/metadata.yaml')
        }

        self.cache[cache_key] = prompt
        return prompt

    def publish_version(self, name: str, version: str, prompts: dict, metadata: dict):
        '''Publish new prompt version'''

        prompt_dir = f'{self.storage}/{name}/{version}'
        os.makedirs(prompt_dir, exist_ok=True)

        # Save files
        self._save_file(f'{prompt_dir}/system.txt', prompts['system'])
        self._save_file(f'{prompt_dir}/user_template.txt', prompts['user_template'])
        self._save_yaml(f'{prompt_dir}/metadata.yaml', metadata)

        # Log to audit
        logger.info('prompt_published', name=name, version=version)

    def set_current(self, name: str, version: str):
        '''Set current version (atomic)'''

        current_link = f'{self.storage}/{name}/current'
        new_link = f'{self.storage}/{name}/current.new'

        # Create new symlink
        os.symlink(version, new_link)

        # Atomic replace
        os.rename(new_link, current_link)

3. A/B TESTING FRAMEWORK

class PromptABTest:
    def __init__(self, name: str, variants: dict[str, float]):
        '''
        variants: {'control': 0.5, 'treatment': 0.5}
        '''
        self.name = name
        self.variants = variants
        self.registry = PromptRegistry()
        self.metrics = ABTestMetrics(name)

    def get_variant(self, user_id: str) -> str:
        '''Deterministic variant assignment'''

        # Hash user_id for consistent assignment
        hash_val = int(hashlib.md5(f'{self.name}:{user_id}'.encode()).hexdigest(), 16)
        bucket = (hash_val % 100) / 100

        cumulative = 0
        for variant, weight in self.variants.items():
            cumulative += weight
            if bucket < cumulative:
                return variant

        return list(self.variants.keys())[0]

    def get_prompt(self, user_id: str) -> tuple[str, dict]:
        '''Get prompt for user, return variant name and prompt'''

        variant = self.get_variant(user_id)

        # Map variant to version
        version_map = self.metrics.get_version_map()
        version = version_map[variant]

        prompt = self.registry.get_prompt(self.name, version)

        return variant, prompt

    def record_outcome(self, user_id: str, metrics: dict):
        '''Record outcome for analysis'''

        variant = self.get_variant(user_id)
        self.metrics.record(variant, metrics)

4. EXPERIMENT ANALYSIS

class ABTestMetrics:
    def __init__(self, experiment_name: str):
        self.name = experiment_name
        self.results = defaultdict(list)

    def record(self, variant: str, metrics: dict):
        self.results[variant].append({
            'timestamp': datetime.utcnow(),
            **metrics
        })

    def analyze(self) -> dict:
        '''Statistical analysis of experiment'''

        analysis = {}

        for metric_name in ['faithfulness', 'relevance', 'latency', 'cost']:
            control_values = [r[metric_name] for r in self.results['control']]
            treatment_values = [r[metric_name] for r in self.results['treatment']]

            # T-test for significance
            t_stat, p_value = ttest_ind(control_values, treatment_values)

            analysis[metric_name] = {
                'control_mean': np.mean(control_values),
                'treatment_mean': np.mean(treatment_values),
                'lift': (np.mean(treatment_values) - np.mean(control_values)) / np.mean(control_values),
                'p_value': p_value,
                'significant': p_value < 0.05,
                'sample_size': {
                    'control': len(control_values),
                    'treatment': len(treatment_values)
                }
            }

        return analysis

5. USAGE EXAMPLE

# Setup experiment
experiment = PromptABTest(
    name='policy_qa',
    variants={
        'control': 0.5,    # v1.0.0
        'treatment': 0.5   # v1.1.0 (with few-shot examples)
    }
)

# In application
def answer_query(user_id: str, query: str) -> str:
    # Get prompt variant
    variant, prompt = experiment.get_prompt(user_id)

    # Format and call LLM
    messages = [
        {'role': 'system', 'content': prompt['system']},
        {'role': 'user', 'content': prompt['user_template'].format(query=query)}
    ]

    response = llm.invoke(messages)

    # Evaluate and record
    scores = evaluator.evaluate(query, response)
    experiment.record_outcome(user_id, {
        'faithfulness': scores['faithfulness'],
        'relevance': scores['relevance'],
        'latency': response.latency,
        'cost': response.cost
    })

    return response

# After sufficient data
results = experiment.analyze()
if results['faithfulness']['significant'] and results['faithfulness']['lift'] > 0.05:
    # Treatment is significantly better
    registry.set_current('policy_qa', 'v1.1.0')"
```

---

## Practice Questions

1. How do you handle multi-turn conversations in a stateless LLM architecture?
2. Design a human-in-the-loop workflow for high-stakes LLM decisions
3. How do you implement model governance for LLM applications in banking?
4. Design a cost allocation system for multi-tenant LLM usage
5. How do you handle model updates and rollbacks in production?
