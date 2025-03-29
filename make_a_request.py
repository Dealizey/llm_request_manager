import requests
import time
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import proxies, api_base, api_key
from config import silicon_flow_api_base, silicon_flow_api_key
from config import ali_api_key, ali_api_base
from llm_client import LLMClient


def handle_errors(func):
    """Decorator for error handling and retries"""
    def wrapper(*args, **kwargs):
        retries = kwargs.pop('retries', 3)
        
        for attempt in range(retries):
            try:
                result = func(*args, **kwargs)
                return result if result is not None else True  # Return True for success if None
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed for {args[0]}: {str(e)}")
                if attempt == retries - 1:
                    print(f"Max retries reached for {args[0]}")
                    return False
                time.sleep(min(2 ** attempt, 5))  # Exponential backoff
            except Exception as e:
                print(f"Unexpected error for {args[0]}: {str(e)}")
                return False
    return wrapper

@handle_errors
def process_model_request(model_name, prompt, stream=False, max_tokens=20000, 
                         thinking=None, reasoning_effort=None, db_path="math.db"):
    """Function to be executed by each thread to process a model request"""
    print(f"\n{'='*50}\nStarting request to model: {model_name}\n{'='*50}")
    
    start_time = time.time()
    
    # Thread-specific print lock to prevent output interleaving
    print_lock = threading.Lock()
    # Handle Silicon Flow API models
    if (model_name == "Qwen/QwQ-32B" or 
        ("deepseek-ai" in model_name and "TA" not in model_name)):
        client = LLMClient(silicon_flow_api_base, silicon_flow_api_key, {}, db_path)
    
    # Handle Ali API models
    elif model_name in ("qwq-32b", "qwq-plus-2025-03-05", "deepseek-r1"):
        client = LLMClient(ali_api_base, ali_api_key, {}, db_path)
    
    # Default case for all other models
    else:
        client = LLMClient(api_base, api_key, proxies, db_path)
    
    # Make the request
    with print_lock:
        result = client.make_request(
            prompt=prompt,
            model_name=model_name,
            stream=stream,
            max_tokens=max_tokens,
            thinking=thinking,
            reasoning_effort=reasoning_effort
        )
    
    execution_time = time.time() - start_time
    print(f"\n{'='*50}\nCompleted request to model: {model_name}")
    print(f"Execution time: {execution_time:.2f} seconds\n{'='*50}")
    
    # Debug print to verify return value
    print(f"Returning success status for {model_name}")
    return True  # Explicitly return True for success

def batch_process_models(models, prompt, max_workers=None, **kwargs):
    """Process models in batches with optimized thread pool"""
    # Calculate optimal workers if not specified
    if max_workers is None:
        max_workers = min(10, len(models))  # Cap at 10 workers
    
    results = {}
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_model_request, model, prompt, **kwargs): model 
            for model in models
        }
        
        for future in as_completed(futures):
            model = futures[future]
            try:
                results[model] = future.result()
            except requests.exceptions.RequestException as e:
                print(f"Request failed for {model}: {str(e)}")
                results[model] = False
            except Exception as e:
                print(f"Unexpected error for {model}: {str(e)}")
                results[model] = False
    
    # Print summary statistics
    successful = sum(1 for r in results.values() if r)
    total_time = time.time() - start_time
    
    print(f"\n{'='*50}")
    print("Batch Processing Summary:")
    print(f"- Total models: {len(models)}")
    print(f"- Successful: {successful}")
    print(f"- Failed: {len(models) - successful}")
    print(f"- Total time: {total_time:.2f} seconds")
    
    print("\nModel Results:")
    for model, success in results.items():
        print(f"- {model}: {'✓' if success else '✗'}")
    
    print('='*50)
    
    return results

# Example usage
if __name__ == "__main__":
    
    MODEL_GROUPS = {
        "thinking_models": [
            "o3-mini",
            "gemini-2.0-flash-thinking-exp-01-21",
            "o1-mini",
            "qwq-32b",
            "qwq-plus-2025-03-05"
        ],
        "regular_models": [
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet",
            "gemini-2.0-flash",
            "deepseek-ai/DeepSeek-V3",
            "gpt-4.5-preview",
            "gpt-4o",
            "TA/Qwen/Qwen2.5-72B-Instruct-Turbo"
        ]
    }
    
    # 配置选项
    model_group = "thinking_models"  # 可更改为 "regular_models" 或其他组
    models = MODEL_GROUPS.get(model_group, [])
    
    # 根据模型组自动设置参数
    if model_group == "thinking_models":
        thinking = {
            "type": "enabled",
            "budget_tokens": 10000
        }
        reasoning_effort = "medium"
    else:
        thinking = None
        reasoning_effort = None
        
    db_path = "math.db"

    # 示例提示
    prompt = r"""2. $“ \frac 1a> \frac 1b> 0$”是“$2^a>2^b$”的
A. 充分不必要条件
B. 必要不充分条件
C. 充分必要条件
D. 既不充分也不必要条件"""


    random.shuffle(models)
    
    # Process models with controlled concurrency
    results = batch_process_models(
        models=models,
        prompt=prompt,
        max_workers=5,  # Optimal concurrency level
        stream=True,
        max_tokens=20000,
        thinking=thinking,
        reasoning_effort=reasoning_effort,
        db_path=db_path
    )
    
    # Print summary statistics
    successful = sum(1 for r in results.values() if r)
    print(f"\nSummary: {successful}/{len(models)} requests succeeded")
    print("Detailed results:")
    for model, success in results.items():
        print(f"{model}: {'Success' if success else 'Failed'}")
