"""
Multi-Agent Customer Support System with LLM Routing

This script implements a multi-agent system for handling customer support queries
related to software engineering. It uses LangChain and LangGraph to create a
workflow that routes and processes queries through different specialized agents.

Key components:
1. ChatOpenRouter: A custom ChatOpenAI class for using OpenRouter API with LangChain.
2. AgentState: A Pydantic model representing the state of a query in the system.
3. Specialized agents:
   - Routing Agent: Determines which department should handle the query.
   - IT Department Agent: Handles coding and technical queries.
   - Architecture Department Agent: Addresses software architecture questions.
   - General Department Agent: Manages general inquiries.
4. StateGraph: Defines the workflow for processing queries.

LLM Routing Feature:
The system implements LLM routing to optimize query handling:
- Different language models are used for different tasks based on their strengths.
- OpenRouter API is integrated with LangChain to access various AI models:
  * GPT-4o-mini: Used for initial query routing and general inquiries.
  * Claude-3.5-sonnet: Employed for IT-related queries, leveraging its coding expertise.
  * GPT-4o: Utilized for architecture questions, benefiting from its advanced reasoning.
- The routing logic ensures that each query is processed by the most suitable model,
  balancing performance and cost-effectiveness.
- This approach allows for flexibility in model selection and easy updates as new
  models become available or as performance characteristics change.

Workflow:
1. User inputs a query.
2. The Routing Agent (GPT-4o-mini) determines the appropriate department.
3. The query is sent to the corresponding department agent:
   - IT queries: Handled by Claude-3.5-sonnet
   - Architecture queries: Processed by GPT-4o
   - General queries: Managed by GPT-4o-mini
4. The department agent generates a response.
5. The response is returned to the user.

Stateful Conversation:
- The system maintains a session ID for each conversation.
- MemorySaver is used to checkpoint the conversation state.
- This allows for potential future enhancements like context retention across queries.

Environment variables:
- OPENROUTER_API_KEY: API key for OpenRouter (must be set in .env file)

Usage:
Run the script and interact with the command-line interface to submit queries.
Type 'exit' to end the session.
"""

## REDO THE COMMENT BLOCK AND ADD PROPER STATEFUL PARTS ON AGENTS

import os
from typing import TypedDict, Dict, Any, Optional, Annotated
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI ## Must be the comminuty one. _openai throws error
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver 
from termcolor import colored
import uuid
from langchain_core.messages import HumanMessage


load_dotenv()

# Set up memory management (in-memory)
memory = MemorySaver()

class ChatOpenRouter(ChatOpenAI):
    openai_api_base: str
    openai_api_key: str
    model_name: str

    def __init__(self,
                 model_name: str,
                 openai_api_key: Optional[str] = None,
                 openai_api_base: str = "https://openrouter.ai/api/v1",
                 **kwargs):
        openai_api_key = openai_api_key or os.getenv('OPENROUTER_API_KEY')
        super().__init__(openai_api_base=openai_api_base,
                         openai_api_key=openai_api_key,
                         model_name=model_name,
                         **kwargs)

# Initialize OpenRouter clients
llm_router = ChatOpenRouter(model_name="openai/gpt-4o-mini", temperature=0)
llm_it = ChatOpenRouter(model_name="anthropic/claude-3.5-sonnet", temperature=0)
llm_architecture = ChatOpenRouter(model_name="openai/gpt-4o", temperature=0)
llm_general = ChatOpenRouter(model_name="openai/gpt-4o-mini", temperature=0)

# Define the state pydantic model for agents system
# class AgentState(TypedDict):
#     department: str = ""
#     messages: Annotated[list[AnyMessage], add_messages]

from langgraph.graph import MessagesState # pre-built 'messages' key, equivalent to the above
class AgentState(MessagesState):
    department: str = ""


## Department Router Agent
### create agent and chain
def route_query(state: Dict[str, Any]):
    print(colored(f"\nRouting Agent: Processing query - '{state['messages'][-1]}'", "blue"))
    router_prompt = ChatPromptTemplate.from_template(
        "Route the following query to the appropriate department: {query}\n"
        "Departments: IT, Architecture, General\n"
        "IT: For coding-related questions, use of frameworks and coding best practices.\n"
        "Architecture: For solutions architecture decisions and tools.\n"
        "General: For general inquiries or queries that don't fit the other categories.\n"
        "Output ONLY ONE of these department names."
    )
    chain = router_prompt | llm_router | StrOutputParser()
    department = chain.invoke({"query": state['messages'][-1]})
    print(colored(f"Routing Agent: Query routed to {department}", "green"))
    return {"department": department}

## IT Department Agent
### create agent and chain  
def handle_it_query(state: Dict[str, Any]):
    print(colored(f"\nIT Department: Handling query - '{state['messages'][-1]}'", "blue"))
    prompt = ChatPromptTemplate.from_template(
        "You are an expert in software development and coding. Answer the following query related to coding or technical issues: {query}"
    )
    chain = prompt | llm_it | StrOutputParser()
    response = chain.invoke({"query": state['messages'][-1]})
    print(colored("IT Department: Response generated", "green"))
    return {"messages": response}

## Architecture Department Agent
### create agent and chain
def handle_architecture_query(state: AgentState):
    print(colored(f"\nArchitecture Department: Handling query - '{state['messages'][-1]}'", "blue"))
    prompt = ChatPromptTemplate.from_template(
        "You are a software architecture specialist. Answer the following query related to software architecture decisions or best practices: {query}"
    )
    chain = prompt | llm_architecture | StrOutputParser()
    response = chain.invoke({"query": state['messages'][-1]})
    print(colored("Architecture Department: Response generated", "green"))
    return {"messages": response}

## General Department Agent
### create agent and chain
def handle_general_query(state: AgentState):
    print(colored(f"\nGeneral Department: Handling query - '{state['messages'][-1]}'", "blue"))
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful customer service assistant for a software engineering company. "
        "Answer the following general query: {query}"
        #"If the query is not related to software engineering or to information about the conversation and user, say 'I'm sorry, I can't help with that.'"
    )
    chain = prompt | llm_general | StrOutputParser()
    response = chain.invoke({"query": state['messages'][-1]})
    print(colored("General Department: Response generated", "green"))
    return {"messages": response}


# Define the LangGraph workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("route", route_query)
workflow.add_node("IT", handle_it_query)
workflow.add_node("Architecture", handle_architecture_query)
workflow.add_node("General", handle_general_query)

# Set the conditional entry point
workflow.set_entry_point("route")

# Add conditional edges
workflow.add_conditional_edges(
    "route",
    lambda x: x['department'],  # Access department as an attribute of AgentState
    {
        "IT": "IT",
        "Architecture": "Architecture",
        "General": "General"
    }
)

# Add edges from department nodes to END
workflow.add_edge("IT", END)
workflow.add_edge("Architecture", END)
workflow.add_edge("General", END)


# Compile the graph with the checkpointer
graph = workflow.compile(checkpointer=memory)

def main():
    print("Welcome to Customer Support!")
    
    session_id = str(uuid.uuid4())
    print(f"Session ID: {session_id}")

    while True:
        query = input("How can I assist you today? (Type 'exit' to end): ")
        if query.lower() == 'exit':
            break

        # Use the session_id as the thread_id
        config = {"configurable": {"thread_id": session_id}}
        
        result = graph.invoke(
            {
                "messages": [HumanMessage(content=query)],
                "department": "",
            },
            config=config
        )
        
        if isinstance(result, dict) and "response" in result:
            print(f"\nCustomer Support Assistant: {result['response']}")
        else:
            print(f"\nDebug - Unexpected result: {result}")
        print(f"Session ID: {session_id}")
        print()

    print("\nThank you for using Customer Support. Have a great day!")

if __name__ == "__main__":
    main()