"""
LLM Fault-Tolerant Router

This script implements a fault-tolerant Language Model (LLM) routing system.
It uses a primary router to direct queries to appropriate models and includes
a fallback mechanism to ensure continuous operation even if primary models fail.
The system leverages OpenAI's API and OpenRouter for accessing various LLM models.

Key features:
- Primary and fallback model configurations
- Fault-tolerant completion mechanism
- Streaming responses for better user experience
- Retry mechanism for failed model attempts
"""

import json
import os
import time
from openai import OpenAI
from termcolor import colored

# Initialize clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Model configurations
MODEL_CONFIGS = {
    "primary": [
        {"name": "openai/gpt-4o", "system_message": "You are a highly capable AI assistant."},
        {"name": "anthropic/claude-3.5-sonnet", "system_message": "You are an AI assistant specializing in detailed explanations."},
        {"name": "meta-llama/llama-3-8b-instruct:free", "system_message": "You are a friendly AI for casual conversations."}
    ],
    "fallback": [
        {"name": "openai/gpt-3.5-turbo", "system_message": "You are a helpful AI assistant."},
        {"name": "anthropic/claude-2", "system_message": "You are an AI assistant ready to help with various tasks."},
        {"name": "google/palm-2-chat-bison", "system_message": "You are a versatile AI assistant."}
    ]
}

def openrouter_completion(model_name, user_query, system_message, max_retries=3):
    """
    Attempt to get a completion from OpenRouter with retry mechanism.
    
    Args:
        model_name (str): Name of the model to use
        user_query (str): User's input query
        system_message (str): System message for the model
        max_retries (int): Maximum number of retry attempts
    
    Returns:
        str: Assistant's response or None if all attempts fail
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_query}
    ]
    
    for attempt in range(max_retries):
        try:
            completion = openrouter_client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True
            )

            assistant_response = ""
            print(colored(f"Assistant ({model_name}):", "green"))
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    chunk_content = chunk.choices[0].delta.content
                    assistant_response += chunk_content
                    print(chunk_content, end="", flush=True)
            print()  # Add a newline after the response
            return assistant_response
        except Exception as e:
            print(colored(f"Error with {model_name}: {str(e)}", "red"))
            if attempt < max_retries - 1:
                print(colored(f"Retrying... (Attempt {attempt + 2}/{max_retries})", "yellow"))
                time.sleep(2)  # Wait for 2 seconds before retrying
            else:
                print(colored(f"All attempts failed for {model_name}", "red"))
                return None

def primary_router(user_input):
    """
    Route the user input to the appropriate primary model.
    
    Args:
        user_input (str): User's input query
    
    Returns:
        str: Name of the selected model or None if routing fails
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": """You are a helpful router designed to output JSON. You decide which route user input belongs to based on the type of the user query.
                 Return your response as follows:
                 {
                 "route": "route_name", can be one of the following:
                 - "openai/gpt-4o" - for more complex, in-depth questions and queries.
                 - "anthropic/claude-3.5-sonnet" for all code and programming related queries. 
                 - "meta-llama/llama-3-8b-instruct:free" for other simple daily conversation like regular speak
                 }"""},
                {"role": "user", "content": user_input}
            ]
        )
        return json.loads(response.choices[0].message.content)["route"]
    except Exception as e:
        print(colored(f"Error in primary routing: {str(e)}", "red"))
        return None

def fault_tolerant_completion(user_input):
    """
    Attempt to get a completion using primary models, falling back to secondary models if necessary.
    
    Args:
        user_input (str): User's input query
    
    Returns:
        str: Assistant's response or an error message if all models fail
    """
    primary_model = primary_router(user_input)
    
    if primary_model:
        print(colored(f"Attempting primary model: {primary_model}", "cyan"))
        for model in MODEL_CONFIGS["primary"]:
            if model["name"] == primary_model:
                response = openrouter_completion(model["name"], user_input, model["system_message"])
                if response:
                    return response
    
    print(colored("Primary models failed. Trying fallback models...", "yellow"))
    for model in MODEL_CONFIGS["fallback"]:
        response = openrouter_completion(model["name"], user_input, model["system_message"])
        if response:
            return response
    
    return colored("All models failed. Please try again later.", "red")

def main():
    """
    Main function to run the fault-tolerant LLM router.
    """
    print(colored("Welcome to the Fault-Tolerant LLM Router!", "cyan"))
    while True:
        user_input = input(colored("Please enter your question (or type 'exit' to quit): ", "cyan"))
        if user_input.lower() == 'exit':
            print(colored("Thank you for using the Fault-Tolerant LLM Router. Goodbye!", "cyan"))
            break

        response = fault_tolerant_completion(user_input)
        print(colored("Is there anything else I can help you with?", "green"))
        print()  # Add a blank line for better readability

if __name__ == "__main__":
    main()