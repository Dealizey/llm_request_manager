"""
LLM Response Server

This module provides a FastAPI server that simulates LLM responses by retrieving
pre-stored conversations from a database and streaming them with realistic timing.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
import time
import json
import re
import tiktoken
from database import ConversationDB

# Initialize FastAPI app and database connection
app = FastAPI(title="LLM Response Server")
db = ConversationDB("math.db")

# Define model output speeds (tokens per second)
MODEL_OUTPUT_SPEEDS = {
    "slow": 15, # Add "slow" to your model name to enable slow mode
    "o1": 40,
    "deepseek-chat": 38,
    "gpt-4o": 86,
    "o3-mini": 105,
    "deepseek-ai/DeepSeek-R1": 38,
    "TA/deepseek-ai/DeepSeek-R1": 76,
    "claude-3-5-sonnet": 55,
    "claude-3-7-sonnet-20250219": 55,
    "gemini-2.0-flash-thinking-exp-01-21": 200,
    "gemini-2.0-flash-001": 170,
    "gpt-4.5-preview": 14,
    "claude-3-opus": 30,
    "ark-deepseek-r1-250120": 28.5
}

# Default output speed if model not found in the dictionary
DEFAULT_OUTPUT_SPEED = 40

# Initialize tiktoken encoder
ENCODER = tiktoken.get_encoding("cl100k_base")  # Default encoder for most models


@app.post('/chat/completions')
async def chat_completions(request: Request):
    """
    Handle chat completion requests and stream responses.
    
    Args:
        request (Request): The incoming FastAPI request
        
    Returns:
        StreamingResponse: A streaming response that simulates an LLM output
        
    Raises:
        HTTPException: If the requested model and prompt combination is not found
    """
    # Parse request data
    data = await request.json()
    prompt = data['messages'][0]['content']
    model_name = data['model']
    
    # Extract optional parameters
    debug_mode = data.get('debug_mode', False)
    smoothing_factor = data.get('smoothing_factor', 0.7)  # 0-1, higher = smoother timing

    # Find matching conversation in database
    response_text = await _find_matching_response(prompt, model_name)
    
    # Create and return streaming response
    return StreamingResponse(
        _generate_streaming_response(
            response_text, 
            model_name, 
            debug_mode=debug_mode,
            smoothing_factor=smoothing_factor
        ),
        media_type='text/event-stream'
    )


async def _find_matching_response(prompt, model_name):
    """
    Find a matching response for the given prompt and model in the database.
    
    Args:
        prompt (str): The user's prompt
        model_name (str): The requested model name
        
    Returns:
        str: The matching response text
        
    Raises:
        HTTPException: If no matching conversation is found
    """
    # Query the database for the prompt
    conversations = db.search_conversations(query=prompt)
    
    # Filter conversations by model name
    matching_conversations = [
        conv for conv in conversations if (conv['model_name'] in model_name)
    ]

    if not matching_conversations:
        raise HTTPException(
            status_code=404, 
            detail="Model and prompt not found in database"
        )

    # Return the response from the first matching conversation
    return matching_conversations[0]['response']


async def _generate_streaming_response(
    response_text, 
    model_name, 
    debug_mode=False, 
    smoothing_factor=0.7
):
    """
    Generate a streaming response that simulates an LLM output with dynamic timing.
    
    Args:
        response_text (str): The full response text to stream
        model_name (str): The model name to determine output speed
        debug_mode (bool): Whether to log timing information
        smoothing_factor (float): 0-1, higher values produce smoother timing adjustments
        
    Yields:
        str: Chunks of the response in SSE format
    """
    # Get the output speed for the model (tokens per second)
    output_speed = MODEL_OUTPUT_SPEEDS.get(model_name, DEFAULT_OUTPUT_SPEED)
    if "slow" in model_name:
        output_speed = MODEL_OUTPUT_SPEEDS["slow"]
    
    # Initialize timing variables
    start_time = time.time()
    tokens_sent = 0
    tokens = ENCODER.encode(response_text)
    total_tokens = len(tokens)
    last_delay = 0  # Track the last delay for smoothing
    
    if debug_mode:
        print(f"Starting response generation for model: {model_name}")
        print(f"Target output speed: {output_speed} tokens/second")
        print(f"Estimated total tokens: {total_tokens}")
    
    # Process the text token by token
    token_buffer = []
    i = 0
    
    while i < len(tokens):
        # Calculate expected progress based on elapsed time
        elapsed_time = time.time() - start_time
        expected_tokens = elapsed_time * output_speed
        
        # Calculate timing adjustment
        if tokens_sent > expected_tokens:
            # We're ahead of schedule, slow down
            tokens_ahead = tokens_sent - expected_tokens
            raw_delay = tokens_ahead / output_speed
            
            # Apply smoothing to prevent abrupt changes in timing
            dynamic_delay = (smoothing_factor * last_delay) + ((1 - smoothing_factor) * raw_delay)
            
            # Apply delay with a minimum to prevent too short sleeps
            actual_delay = max(dynamic_delay, 0)
            time.sleep(actual_delay)
            last_delay = actual_delay
            
            if debug_mode and tokens_sent % 10 == 0:
                print(f"Ahead by {tokens_ahead:.1f} tokens, sleeping for {actual_delay:.3f}s")
                
        elif tokens_sent < expected_tokens - 2:  # Small buffer to prevent oscillation
            # We're behind schedule, try to catch up by processing more tokens before next delay
            # This is handled implicitly by not adding delays
            if debug_mode and tokens_sent % 10 == 0:
                tokens_behind = expected_tokens - tokens_sent
                print(f"Behind by {tokens_behind:.1f} tokens, catching up...")
        
        # Get the next token
        if i < len(tokens):
            token_id = tokens[i]
            i += 1
            
            # Add token to buffer
            token_buffer.append(token_id)
            
            # Try to decode the buffer
            try:
                chunk_text = ENCODER.decode(token_buffer, "strict")
                yield _format_sse_message(chunk_text)
                tokens_sent += len(token_buffer)
                token_buffer = []
            except UnicodeDecodeError:
                continue

    
    # Send any remaining tokens
    if token_buffer:
        try:
            final_chunk = ENCODER.decode(token_buffer)
            yield _format_sse_message(final_chunk)
            tokens_sent += len(token_buffer)
        except Exception as e:
            if debug_mode:
                print(f"Error decoding final tokens: {e}")
    
    # Log progress for debugging
    if debug_mode and tokens_sent % 20 == 0:
        print(f"Progress: {tokens_sent}/{total_tokens} tokens ({tokens_sent/total_tokens*100:.1f}%), " 
              f"Time: {elapsed_time:.2f}s, Expected: {expected_tokens:.1f} tokens")
    
    # Log final statistics
    if debug_mode:
        final_time = time.time() - start_time
        print(f"\nResponse complete:")
        print(f"Total tokens: {tokens_sent}")
        print(f"Total time: {final_time:.2f}s")
        print(f"Average speed: {tokens_sent/final_time:.1f} tokens/second")


def _get_token_count(text):
    """
    Get the exact number of tokens in the text using tiktoken.
    
    Args:
        text (str): The text to count tokens for
        
    Returns:
        int: Exact token count
    """
    return len(ENCODER.encode(text))


def _format_sse_message(content):
    """
    Format content as a Server-Sent Events (SSE) message.
    
    Args:
        content (str): The content to include in the message
        
    Returns:
        str: Formatted SSE message
    """
    data = {
        'choices': [
            {
                'delta': {
                    'content': content
                }
            }
        ]
    }
    return f"data: {json.dumps(data)}\n\n"
