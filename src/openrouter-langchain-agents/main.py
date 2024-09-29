"""
Multi-Agent Customer Support System with LLM Routing and Stateful Conversations

This script implements a multi-agent system for handling customer support queries
related to software engineering. It uses LangChain and LangGraph to create a
workflow that routes and processes queries through different specialized agents.

Key components:
1. ChatOpenRouter: A custom ChatOpenAI class for using OpenRouter API with LangChain.
2. AgentState: A Pydantic model representing the state of a query in the system,
   including the conversation history.
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

Workflow:
1. User inputs a query.
2. The Routing Agent (GPT-4o-mini) determines the appropriate department.
3. The query is sent to the corresponding department agent:
   - IT queries: Handled by Claude-3.5-sonnet
   - Architecture queries: Processed by GPT-4o
   - General queries: Managed by GPT-4o-mini
4. The department agent generates a response.
5. The response is returned to the user.

Stateful Conversation: [NOT WORKING YET]
- The system maintains a session ID for each conversation.
- MemorySaver is used to checkpoint the conversation state.
- AgentState includes a 'messages' field to store the conversation history.
- Each agent function receives and updates the state, allowing for context retention.

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
from langchain_community.chat_models import ChatOpenAI ## Must be the community one, the _openai throws error
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver 
from termcolor import colored
import uuid
from langchain_core.messages import HumanMessage, SystemMessage , RemoveMessage
# MessagesState has pre-built 'messages' key with add_messages reducer, equivalent to the above
from langgraph.graph import MessagesState

load_dotenv()

#### Custom ChatOpenRouter class to use OpenRouter API with LangChain
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
llm_summary = ChatOpenRouter(model_name="openai/gpt-4o-mini", temperature=0)
llm_call_model = ChatOpenRouter(model_name="openai/gpt-4o-mini", temperature=0)

# Define the state pydantic model for agents system
# class AgentState(TypedDict):
#     department: str = ""
#     messages: Annotated[list[AnyMessage], add_messages]

from langgraph.graph import MessagesState # pre-built 'messages' key with add_messages reducer, equivalent to the above
class AgentState(MessagesState):
    department: str
    summary: str

##### Handling memory and summarization
## Logic to summarize conversation and add it to the system message
def call_model(state: AgentState):
    
    # Get summary if it exists
    summary = state.get("summary", "")

    # If there is summary, add it too syst message
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    
    else:
        messages = state["messages"]
    
    #response = llm_call_model.invoke(messages)
    return {"messages": messages}

def summarize_conversation(state: AgentState):   
    # get existing summary, if any
    summary = state.get("summary", "")

    # create 'summarization prompt'
    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
        
    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to messages history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm_summary.invoke(messages)
    
    # Delete all but the 5 most recent messages -> token optimization strategy
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-5]]
    return {"summary": response.content, "messages": delete_messages}

def should_summarize(state: AgentState):
    
    """Return the next node to execute."""
    
    messages = state["messages"]
    
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"
    
    # Otherwise go to route agent
    return "route"


##### Department agents
## Department Router Agent
### create agent and chain
def route_query(state: AgentState):
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
    department = chain.invoke({"query": state['messages']})
    print(colored(f"Routing Agent: Query routed to {department}", "green"))
    return {"department": department}

## IT Department Agent
### create agent and chain  
def handle_it_query(state: AgentState):
    print(colored(f"\nIT Department: Handling query - '{state['messages'][-1]}'", "blue"))
    prompt = ChatPromptTemplate.from_template(
        "You are an expert in software development and coding. Answer the following query related to coding or technical issues: {query}"
    )
    chain = prompt | llm_it | StrOutputParser()
    response = chain.invoke({"query": state['messages']})
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
    response = chain.invoke({"query": state['messages']})
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
    response = chain.invoke({"query": state['messages']})
    print(colored("General Department: Response generated", "green"))
    return {"messages": response}


# Define the LangGraph workflow state
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("call_model", call_model)
workflow.add_node("summarize_conversation", summarize_conversation)
workflow.add_node("route", route_query)
workflow.add_node("IT", handle_it_query)
workflow.add_node("Architecture", handle_architecture_query)
workflow.add_node("General", handle_general_query)

# Set the entry point
workflow.set_entry_point("call_model")

workflow.add_conditional_edges(
    "call_model",
    should_summarize,
    {
        "summarize_conversation": "summarize_conversation",
        "route": "route"
    }
)
workflow.add_edge("summarize_conversation", "route")

# Add edges from department nodes to END
workflow.add_edge("IT", END)
workflow.add_edge("Architecture", END)
workflow.add_edge("General", END)

# conditional edges - departments
workflow.add_conditional_edges(
    "route",
    lambda x: x['department'],  # Access department as an attribute of AgentState
    {
        "IT": "IT",
        "Architecture": "Architecture",
        "General": "General"
    }
)


# Set up memory management (in-memory)
memory = MemorySaver()
# Compile the graph with the checkpointer
graph = workflow.compile(checkpointer=memory)

# Define the path for the  graph image file
img_path = 'src/openrouter-langchain-agents/img'
img_file = os.path.join(img_path, 'department_workflow_graph.png')

# Check if the image file already exists
if not os.path.exists(img_file):
    # Create the 'img' directory if it doesn't exist
    os.makedirs(img_path, exist_ok=True)

    # Generate the Mermaid PNG and save it
    mermaid_png = graph.get_graph().draw_mermaid_png()

    # Save the PNG to a file
    with open(img_file, 'wb') as f:
        f.write(mermaid_png)

    print(f"Graph image saved as '{img_file}'")
else:
    print(f"Graph image already exists at '{img_file}'")

def main():
    print("Welcome to Customer Support!")
    
    session_id = str(uuid.uuid4())
    print(f"Session ID: {session_id}")

    while True:

        query = input("Enter your query (Type 'exit' to end): ")
        if query.lower() == 'exit':
            break
        
        config = {"configurable": {"thread_id": session_id}}
        input_message = HumanMessage(content=query)
        output = graph.invoke({"messages": [input_message]}, config=config)
        
        # Print the last message in the output
        if output['messages']:
            last_message = output['messages'][-1]
            print(colored(f"\nCustomer Support Assistant: {last_message}", "yellow"))
        
        # DEBUG !!!
        # print(graph.get_state(config).values.get("summary",""))        
        # print(f"Session ID: {session_id}")
        print()

    print("\nThank you for using Customer Support. Have a great day!")

if __name__ == "__main__":
    main()