"""
Basic LLM Router

This script implements a simple Language Model (LLM) routing system.
It uses OpenRouter to direct user queries to different LLM models
based on the contextual nature of the input.

Key features:
- Model routing using GPT-4o-mini through OpenRouter
- Support for multiple models: GPT-4o, Claude 3.5 Sonnet, and LLaMA 3 8B
- Interactive CLI for user queries
- Error handling and graceful error recovery

OpenRouter:
OpenRouter serves as a unified API gateway for accessing various AI models.
It allows the script to interact with multiple LLM providers (OpenAI, Anthropic, Meta)
through a single interface, simplifying the process of model selection and API calls.

The script demonstrates a basic approach to LLM routing, serving as a foundation
for more complex routing strategies. It includes error handling to manage potential
issues with API calls, JSON parsing, and unexpected model responses.

Usage:
The script runs an interactive loop where users can input questions. The system
determines the most appropriate model for each query and returns the model's response.
Users can exit the program by typing 'exit'.

Note: Ensure that the OPENROUTER_API_KEY is set in the environment variables or a .env file.
"""

from openai import OpenAI
import json
import os
from dotenv import load_dotenv

load_dotenv()

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

def openrouter_completion(model_name, user_query, system_message=None):
    """
    Get a completion from OpenRouter for the specified model.

    Args:
        model_name (str): Name of the model to use
        user_query (str): User's input query
        system_message (str, optional): System message for the model.
        If not provided, the model will use the default behavior.

    Returns:
        str: Model's response content
    """
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_query})
    try:
        completion = openrouter_client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error in openrouter_completion: {str(e)}")
        return None

def router(user_input):
    """
    Route the user input to the appropriate model based on the query type.

    This function uses OpenRouter with the GPT-4o-mini model to determine the
    appropriate route for the user's query. It sends the user input to the model
    and expects a JSON response specifying which model should handle the query.

    Args:
        user_input (str): The user's input query.

    Returns:
        dict: A dictionary containing the route information, or None if an error occurs.
        The dictionary has the following structure:
        {
            "route": "route_name"
        }
        where "route_name" is one of:
        - "openai/gpt-4o" for complex queries (excluding code)
        - "anthropic/claude-3.5-sonnet" for code-related queries
        - "meta-llama/llama-3-8b-instruct:free" for simple conversations
    """
    try:
        response = openrouter_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            response_format={ "type": "json_object" },
            temperature=0,
            messages=[
                {"role": "system", "content": """You are a helpful router designed to output JSON. You decide which route user input belongs to based on the type of the user query.
                 Return your response as follows:
                 {
                 "route": "route_name", can be one of the following:
                 - "openai/gpt-4o" (the o at the end must be present) - for more complex, in-depth questions and queries except code
                 - "anthropic/claude-3.5-sonnet" for all code queries. 
                 - "meta-llama/llama-3-8b-instruct:free" for other simple daily conversation like regular speak
                 }"""},
                {"role": "user", "content": user_input}
            ]
        )
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        print("Error: Invalid JSON response from the router.")
        return None
    except Exception as e:
        print(f"Error in router: {str(e)}")
        return None

def main():
    """
    Main function to run the LLM router.

    This function implements an interactive loop where the user can input questions
    and receive responses from different language models based on the query type.
    The function uses the router to determine the appropriate model and then calls
    the openrouter_completion function to get the response.

    The loop continues until the user types 'exit' to quit the program.
    """
    print("Welcome to the LLM Router! Type 'exit' at any time to quit the program.")
    while True:
        user_input = input("Please enter your question: ")
        if user_input.lower() == 'exit':
            print("Thank you for using the LLM Router. Goodbye!")
            break

        try:
            route_response = router(user_input)
            if route_response is None:
                print("Error: Unable to determine the appropriate model. Please try again.")
                continue

            model = route_response.get("route")
            if not model:
                print("Error: Invalid routing response. Please try again.")
                continue

            print(f"Routing to {model}")

            response = None
            if model == "openai/gpt-4o":
                ## Example setting behavior with a system message:
                #system_message = "You are a masterful English literary expert." #just
                #response = openrouter_completion(model, user_input, system_message)
                response = openrouter_completion(model, user_input)
            elif model in ["anthropic/claude-3.5-sonnet", "meta-llama/llama-3-8b-instruct:free"]:
                response = openrouter_completion(model, user_input)
            else:
                print(f"Error: Unsupported model '{model}'. Please try again.")
                continue

            if response:
                print(response)
            else:
                print("Error: Unable to get a response from the model. Please try again.")

        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}. Please try again.")

if __name__ == "__main__":
    main()
