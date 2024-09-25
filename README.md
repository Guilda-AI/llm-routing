# LLM Routing Strategies

This project demonstrates various strategies for routing user queries to appropriate Language Models (LLMs). Each approach offers different levels of sophistication and flexibility, impacting the system's efficiency, accuracy, and fault tolerance.

## 1. Basic LLM Router (1_llm_router.py)

### Routing Strategy:
- Uses GPT-4o-mini as a dedicated routing model
- Employs a simple rule-based approach with predefined categories

### Impact:
- Straightforward implementation, easy to understand and maintain
- Limited flexibility; may struggle with ambiguous queries
- Cost-effective for basic categorization tasks
- Potential for misrouting complex or edge-case queries

## 2. Advanced LLM Router (2_advanced_routing.py)

### Routing Strategy:
- Two-stage routing process:
  1. Model selection using GPT-4o-mini
  2. System prompt selection for the chosen model
- Dynamic adaptation of system prompts based on query content

### Impact:
- More nuanced routing decisions, better handling of complex queries
- Improved response quality through tailored system prompts
- Higher computational cost due to multiple API calls
  - Each routing decision and system prompt selection requires a separate API call, increasing overall latency and potential costs
- Enhanced flexibility in model behavior across different query types

## 3. Multi-Agent Customer Support System (main.py) [NEEDS TO FIX MEMORY]
### Routing Strategy:
- Uses LangChain, OpenRouter and LangGraph for a workflow-based approach
- Implements specialized agents for different departments:
  - Routing Agent (GPT-4o-mini)
  - IT Department Agent (Claude-3.5-sonnet)
  - Architecture Department Agent (GPT-4o)
  - General Department Agent (GPT-4o-mini)
- Maintains conversation state using MemorySaver

### Impact:
- Highly modular and extensible system
- Leverages specialized models for different tasks (e.g., Claude for coding queries)
- Enables complex multi-step workflows and context retention
- Increased system complexity and potential for higher latency
- Improved handling of domain-specific queries

## 4. Fault-Tolerant LLM Router (4_fault_tolerance.py) [TO BE DONE]

### Routing Strategy:
- Primary routing using GPT-4o
- Fallback mechanism with alternative models
- Retry logic for failed API calls

### Impact:
- Enhanced system reliability and uptime
- Graceful degradation in case of model or API failures
- Potential for increased response times due to retries and fallbacks
- Balances between optimal model selection and system availability

## Comparison of Approaches

1. **Complexity vs. Flexibility**:
   - Basic Router: Low complexity, limited flexibility
   - Advanced Router: Moderate complexity, high flexibility
   - Multi-Agent System: High complexity, highest flexibility
   - Fault-Tolerant Router: Moderate complexity, high reliability

2. **Performance Optimization**:
   - Basic Router: Minimal optimization, fixed routing
   - Advanced Router: Optimized for query-specific responses
   - Multi-Agent System: Optimized for domain-specific handling
   - Fault-Tolerant Router: Optimized for system reliability

3. **Cost Considerations**:
   - Basic Router: Most cost-effective
   - Advanced Router: Higher cost due to multiple API calls
   - Multi-Agent System: Potentially highest cost, depending on query complexity
   - Fault-Tolerant Router: Moderate to high cost, balanced with reliability

4. **Use Case Suitability**:
   - Basic Router: Simple applications with clear query categories
   - Advanced Router: Applications requiring nuanced understanding of queries
   - Multi-Agent System: Complex support systems or multi-domain applications
   - Fault-Tolerant Router: Mission-critical applications requiring high availability

## Conclusion

The choice of routing strategy significantly impacts the system's behavior, performance, and reliability. While simpler strategies like the Basic Router offer ease of implementation and cost-effectiveness, more advanced approaches like the Multi-Agent System provide sophisticated handling of complex queries at the cost of increased complexity. The Fault-Tolerant Router strikes a balance between optimal performance and system reliability, making it suitable for critical applications.

Developers should consider their specific use case, performance requirements, and budget constraints when selecting or adapting these routing strategies.

## Running the Project with Poetry

This project uses Poetry for dependency management and packaging. Follow these steps to run the project:

1. Ensure you have Poetry installed. If not, install it by following the instructions at [https://python-poetry.org/docs/#installation](https://python-poetry.org/docs/#installation)

2. Clone the repository and navigate to the project directory:
   ```
   git clone <repository-url>
   cd llm-routing
   ```

3. Install the project dependencies:
   ```
   poetry install --no-root
   ```

4. Activate the virtual environment:
   ```
   poetry shell
   ```

5. Run the scripts, e.g.:
   ```
   python src/openrouter-langchain-agents/main.py
   ```

7. To exit the program, type 'exit' when prompted for a query.

Note: Ensure you have set up the necessary API keys for the language models used in the project. These should be stored in a `.env` file in the project root directory.