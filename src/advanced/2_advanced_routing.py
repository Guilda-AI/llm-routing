"""
Advanced LLM Router

This script implements an advanced Language Model (LLM) routing system using OpenRouter.
It dynamically selects the most appropriate model and system prompt based on user input.

Key components:
1. model_router: Determines the best model for a given user query (5-second timeout).
2. system_prompt_router: Selects the most suitable system prompt for the chosen model (5-second timeout).
3. openrouter_completion: Generates the response using the selected model and system prompt (45-second timeout).

The script uses GPT-4o-mini for routing decisions and supports three models:
- GPT-4o for complex, non-programming queries
- Claude 3.5 Sonnet for programming-related queries
- LLaMA 3 8B for simple, conversational queries

Features:
- Dynamic model selection based on query content
- Adaptive system prompts for each model, to set the best behavior accordingly.
- Real-time streaming of responses
- Error handling and timeouts for API calls

Usage:
Run the script and input your queries when prompted. The system will automatically
route your query to the appropriate model, select a fitting system prompt, and 
generate a response. Type 'exit' to quit the program.

Requirements:
- OpenAI Python library
- python-dotenv
- termcolor
- timeout-decorator

Note: Ensure that OPENROUTER_API_KEY is set in your environment variables or .env file.
"""

import json
import os
from openai import OpenAI
from termcolor import colored
from dotenv import load_dotenv
from timeout_decorator import timeout

load_dotenv()

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

@timeout(45)  
def openrouter_completion(model_name, user_query, system_message=None):
    """
    Generate a completion using OpenRouter API for the specified model.

    This function sends a request to the OpenRouter API to generate a completion
    based on the given model, user query, and optional system message. 
    
    It streams the response and prints it in real-time.

    Args:
        model_name (str): The name of the model to use for completion.
        user_query (str): The user's input query.
        system_message (str, optional): A system message to guide the model's behavior.

    Returns:
        str: The complete assistant's response, or None if an error occurs.

    Note:
        This function uses streaming to display the response in real-time.
        It has a timeout of 45 seconds.
    """
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_query})
    
    try:
        completion = openrouter_client.chat.completions.create(
            model=model_name,
            messages=messages, 
            stream=True,
        )

        model_response = ""
        print(colored("Assistant response:", "green"))
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                chunk_content = chunk.choices[0].delta.content
                model_response += chunk_content
                print(chunk_content, end="", flush=True)
        return model_response
    except Exception as e:
        print(colored(f"Error in openrouter_completion: {str(e)}", "red"))
        return None

@timeout(5)
def model_router(user_input):
    """
    Route the user input to the appropriate model based on the query type.

    This function uses GPT-4o-mini to determine the most suitable model for handling
    the user's input. It sends a request to the OpenAI API with a system message
    that instructs the model to act as a router and choose between three options:
    GPT-4o, Claude 3.5 Sonnet, or LLaMA 3 8B.

    Args:
        user_input (str): The user's input query.

    Returns:
        str: The name of the chosen model/route, or None if an error occurs.

    Note:
        The function expects a JSON response from the API and extracts the 'route'
        key from it. If any error occurs during the process, it will be caught
        and reported. It has a timeout of 5 seconds.
    """
    try:
        response = openrouter_client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={ "type": "json_object" },
            temperature=0,
            messages=[
                {"role": "system", "content": """You are a helpful router designed to output JSON. You decide which route user input belongs to based on the type of the user query.
                 Return your response as follows:
                 {
                 "route": "route_name", can be one of the following:
                 - "openai/gpt-4o" (the o at the end must be present) - for more complex, in-depth questions and queries. This route doesn't accept programming related queries.
                 - "anthropic/claude-3.5-sonnet" for all code and programming related queries. 
                 - "meta-llama/llama-3-8b-instruct:free" for other simple daily conversation like regular speak
                 }"""},
                {"role": "user", "content": user_input}
            ]
        )
        return json.loads(response.choices[0].message.content)["route"]
    except json.JSONDecodeError as e:
        print(colored(f"Error decoding JSON response: {str(e)}", "red"))
        return None
    except KeyError as e:
        print(colored(f"Error accessing 'route' key in response: {str(e)}", "red"))
        return None
    except Exception as e:
        print(colored(f"An unexpected error occurred in model_router: {str(e)}", "red"))
        return None

@timeout(5)
def system_prompt_router(model_name, user_input):
    """
    Route the user input to the appropriate system message based on the model and query type.

    Args:
        model_name (str): The name of the model to route for.
        user_input (str): The user's input query.

    Returns:
        str: The chosen system message, or None if an error occurs.

    Note:
        This function has a timeout of 5 seconds.
    """
    system_messages = {
        "openai/gpt-4o": [
            "You are a masterful English literary expert, well-versed in all forms of literature and literary analysis.",
            "You are an advanced scientific researcher with expertise across multiple disciplines.",
            "You are a skilled philosopher and ethicist, capable of deep analysis on complex topics."
        ],
        "anthropic/claude-3.5-sonnet": [
            "You are an expert software engineer with deep knowledge of multiple programming languages and best practices.",
            "You are a data scientist specializing in machine learning and statistical analysis.",
            "You are a full-stack web developer with expertise in modern web technologies and frameworks."
        ],
        "meta-llama/llama-3-8b-instruct:free": [
            "You are a friendly conversational AI, adept at casual chit-chat and providing basic information.",
            "You are a helpful virtual assistant, ready to assist with day-to-day queries and tasks.",
            "You are an empathetic listener, providing supportive responses in everyday conversations."
        ]
    }

    if model_name not in system_messages:
        return "You are a helpful AI assistant."

    try:
        response = openrouter_client.chat.completions.create(
            model="gpt-4o",
            response_format={ "type": "json_object" },
            temperature=0,
            messages=[
                {"role": "system", "content": f"""You are a secondary router for the {model_name} model. Based on the user input, choose the most appropriate system message. 
                Return your response as a JSON object as follows:
                {{
                "system_message": "chosen_system_message"
                }}
                Choose from these options:
                {json.dumps(system_messages[model_name])}
                """},
                {"role": "user", "content": user_input}
            ]
        )
        return json.loads(response.choices[0].message.content)["system_message"]
    except Exception as e:
        print(colored(f"Error in system_prompt_router: {str(e)}", "red"))
        return None

def main():
    while True:
        user_input = input(colored("Please enter your question (or type 'exit' to quit): ", "cyan"))
        if user_input.lower() == 'exit':
            break

        model = model_router(user_input)
        print(colored(f"Routing to {model}", "yellow"))

        system_message = system_prompt_router(model, user_input)
        print(colored(f"Using system message: {system_message}", "magenta"))

        response = openrouter_completion(model, user_input, system_message)
        print()

if __name__ == "__main__":
    main()