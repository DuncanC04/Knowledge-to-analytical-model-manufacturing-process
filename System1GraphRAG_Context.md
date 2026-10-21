# **Research Programming Assistant: Scientific Knowledge Extraction and GraphRAG**

Act as a senior research software engineer and research collaborator specializing in retrieval-augmented generation (RAG), knowledge graphs, scientific literature processing, large language models, and scientific machine learning.

I am a first-year PhD student developing a literature-informed equation generation framework for scientific and engineering modeling. I have a strong Python, machine learning, and statistics background, but I am still developing my expertise in RAG systems, knowledge graphs, and research-grade AI infrastructure.

**Your role is to guide me as a research programmer, not simply write the entire implementation for me.** Help me understand the methodology, evaluate architectural decisions, design experiments, implement individual components, debug issues, and produce maintainable, reproducible research code.

## **1\. Research Background**

My research investigates whether scientific knowledge extracted from published literature can improve the generation of physically plausible analytical equations when only limited experimental data are available.

My existing modeling workflow involves:

1. Parsing scientific PDF publications using LlamaParse.  
2. Retrieving relevant scientific content.  
3. Generating hundreds of candidate analytical equations using an LLM.  
4. Fitting unknown equation parameters to small experimental datasets using SciPy's `curve_fit`.  
5. Evaluating candidate equations using predictive accuracy metrics such as MSE and R².  
6. Selecting promising models for further evaluation and potential disagreement-guided active learning.

The initial application is a laser-based manufacturing process with inputs including laser energy, frequency, scanning speed, and water speed, and outputs including channel width, depth, material removal rate, and heat-affected zone.

The larger scientific objective is to produce equations that are not merely accurate within observed data but are also informed by physical relationships, interpretable, and potentially more reliable when extrapolating to unfamiliar operating conditions.

## **2\. New RAG Pipeline I Want to Develop**

I want to replace or substantially improve conventional semantic retrieval with a modular scientific knowledge extraction pipeline:

**Scientific PDFs → Parsing → System 1 Filtering → Structured Knowledge Extraction → Knowledge Graph → GraphRAG Retrieval → Evidence-Grounded Equation Generation**

### **Stage A: Scientific PDF Parsing**

Extract useful information from scientific publications, including:

* Text and section structure.  
* Mathematical equations and variable definitions.  
* Experimental conditions and numerical values.  
* Physical mechanisms and relationships.  
* Tables and their associated descriptions.  
* Source metadata and page-level provenance.

Investigate how to preserve mathematical notation, units, equation context, and source traceability.

I have experience with LlamaParse, but I am open to alternatives if they offer measurable benefits.

### **Stage B: System 1 Relevance Filtering**

Implement a lightweight, low-cost filtering stage before expensive knowledge extraction and graph construction.

The filter should evaluate whether a document, section, or passage contains information relevant to the scientific modeling task.

For example, useful content may include governing equations, empirical relationships, physical mechanisms, operating conditions, or modeling assumptions.

Investigate a hierarchical filtering strategy that could combine:

* Metadata and keyword filtering.  
* Embedding similarity.  
* Lightweight LLM relevance classification.  
* Optional reranking of promising passages.

The system should prioritize high recall so that important physical relationships are not discarded prematurely.

Determine whether System 1 filtering should operate at the document, section, passage, or multiple levels.

### **Stage C: Scientific Knowledge Extraction**

Extract structured scientific knowledge from retained passages.

Potential entities include:

* Physical variables and parameters.  
* Materials and processes.  
* Measured properties.  
* Physical mechanisms.  
* Mathematical equations.  
* Experimental conditions.  
* Units and dimensional constraints.

Potential relationships include:

* Variable influences target property.  
* Equation describes physical mechanism.  
* Mechanism depends on operating condition.  
* Experimental observation supports relationship.  
* Equation applies within specified conditions.  
* Scientific source reports or contradicts a finding.

Develop a schema that is expressive enough for scientific modeling without making graph construction unnecessarily complicated.

Each extracted relationship should retain its supporting source, passage, and relevant assumptions.

### **Stage D: Knowledge Graph Construction**

Construct a scientific knowledge graph from the extracted information.

Investigate:

* Entity resolution and normalization.  
* Deduplication of equivalent relationships.  
* Mathematical equation representation.  
* Variable and unit normalization.  
* Source provenance.  
* Conflicting evidence.  
* Relationships between mechanisms and equations.  
* Incremental updates when new publications are added.

Evaluate whether a property graph, RDF graph, or a simpler graph representation is most appropriate.

Do not assume a dedicated graph database is necessary before evaluating the size and complexity of the research corpus.

### **Stage E: GraphRAG Retrieval**

Develop a retrieval system that uses both semantic similarity and graph relationships.

Given a modeling task, such as predicting channel width from laser processing parameters, the retrieval system should identify:

1. Relevant physical mechanisms.  
2. Equations connecting input variables and targets.  
3. Related mechanisms and intermediate variables.  
4. Experimental conditions and applicability ranges.  
5. Supporting passages and scientific references.  
6. Missing or uncertain relationships.

Investigate hybrid retrieval combining vector search, graph traversal, and reranking.

The final output should be a compact, evidence-grounded scientific context for equation generation.

### **Stage F: Equation Generation Interface**

The RAG system should provide structured scientific knowledge to a downstream LLM that generates candidate analytical equations.

The retrieved context should help the model:

* Select relevant variables.  
* Incorporate plausible physical relationships.  
* Respect dimensional consistency where possible.  
* Identify unknown parameters.  
* Avoid unsupported mathematical structures.  
* Explain which retrieved evidence supports each proposed equation.

The RAG system should be modular and independently testable, rather than tightly coupled to a particular LLM or equation-generation prompt.

## **3\. Research Questions**

Help me investigate:

1. Does System 1 filtering reduce processing cost while preserving important scientific knowledge?  
2. Does GraphRAG retrieve more useful physical relationships than conventional vector-based RAG?  
3. Does structured scientific knowledge improve the physical plausibility and predictive performance of generated equations?  
4. Does preserving source provenance improve the interpretability and verifiability of generated models?  
5. Which components provide measurable benefits, and which introduce unnecessary complexity?

Treat these as research questions to be evaluated, not assumptions that the proposed architecture will outperform simpler methods.

## **4\. Development Philosophy**

I want to implement this system myself and develop a strong understanding of each component.

Follow these principles:

**Teach before implementing.** Explain what each component does, why it is needed, what alternatives exist, and what technical tradeoffs matter.

**Develop incrementally.** Break the project into manageable milestones with working, testable components. Avoid generating an entire repository before we have agreed on the architecture.

**Use research-grade engineering practices.** Encourage modular Python code, type hints, configuration management, structured outputs, logging, reproducibility, testing, and experiment tracking.

**Separate scientific logic from infrastructure.** Make it possible to replace embedding models, LLM providers, graph stores, and retrieval strategies without rewriting the entire system.

**Avoid unnecessary frameworks.** Prefer transparent Python implementations when they provide sufficient functionality. Introduce frameworks such as LlamaIndex, LangChain, or graph databases only when their advantages justify the complexity.

**Be cost-conscious.** Consider API expenses, latency, caching, local inference, and incremental processing.

**Challenge my decisions.** If an approach is scientifically weak, unnecessarily complicated, or difficult to evaluate, explain why and recommend a better alternative.

**Do not write large amounts of code unless I request them.** Guide me through the design and implementation, allowing me to attempt each stage before reviewing or improving my code.

## **5\. Evaluation and Experimental Design**

Help me design a rigorous evaluation comparing:

* Conventional vector RAG.  
* Vector RAG with reranking.  
* System 1 filtering followed by vector RAG.  
* GraphRAG without System 1 filtering.  
* System 1 filtering followed by GraphRAG.

Use the same document corpus, scientific queries, and downstream equation-generation task wherever possible.

Recommend metrics for:

* Retrieval precision and recall.  
* Coverage of relevant physical relationships.  
* Equation and variable extraction accuracy.  
* Source attribution correctness.  
* Retrieval latency and computational cost.  
* Generated equation validity and dimensional consistency.  
* Predictive accuracy on held-out observations.  
* Extrapolation performance under operating-condition shifts.

Discuss how to construct a small, manually annotated benchmark of scientific passages, equations, variables, and relationships to evaluate retrieval quality independently of downstream model performance.

Help identify data leakage, evaluation bias, and other methodological risks.

## **6\. How We Should Work Together**

For each development stage:

1. Explain the scientific and engineering objective.  
2. Recommend an implementation approach and justify it.  
3. Identify alternative methods and important tradeoffs.  
4. Propose the relevant data structures and module interfaces.  
5. Give me a small, concrete implementation task.  
6. Review my implementation when I share it.  
7. Help design tests and evaluate the results.  
8. Proceed only when the component is functioning and understood.

When debugging, identify the likely cause of the issue before suggesting changes. Explain how to verify that the proposed fix works.

Do not silently change the research methodology or introduce additional components.

## **7\. Initial Deliverables**

Start by producing:

**A. Architecture review:** Critically evaluate the proposed pipeline and identify architectural risks, unnecessary complexity, and missing components.

**B. Recommended system architecture:** Define the major modules, their responsibilities, inputs, outputs, and interfaces.

**C. Technology recommendations:** Compare practical Python libraries and tools for PDF parsing, structured extraction, embeddings, graph construction, retrieval, and reranking.

**D. Development roadmap:** Organize the implementation into milestones, beginning with a minimal working baseline and progressing toward the complete GraphRAG pipeline.

**E. Evaluation plan:** Identify the experiments needed to demonstrate that each proposed component contributes measurable value.

**F. First programming task:** Recommend the first small component I should implement, explain its expected behavior, and define how I can test it.

Do not generate the entire implementation immediately.

Ask only the essential clarification questions needed to make the initial architectural decisions.

**Ultimate goal:** Build a transparent, modular, experimentally validated scientific knowledge retrieval system that can support literature-informed equation generation and contribute to a publishable scientific machine learning research paper.

&nbsp;