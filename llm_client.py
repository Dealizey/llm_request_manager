import requests
import json
import sseclient
import time
from database import ConversationDB
import threading

# Global database lock to ensure thread safety for database operations
db_lock = threading.Lock()
class LLMClient:
    def __init__(self, api_base_url, api_key_str, proxies_set, db_path="conversations.db"):
        # API configurations
        self.api_base = api_base_url
        self.api_key = api_key_str
        self.proxies = proxies_set
        
        # Default model settings
        self.openai_compatible_models = ["o1", "deepseek-chat", "gpt-4o", "o3-mini", "deepseek-ai/DeepSeek-R1", "TA/deepseek-ai/DeepSeek-R1"]
        self.claude_models = ["claude-3-5-sonnet", "claude-3-7-sonnet-20250219", "claude"]
        
        # Anthropic API settings
        # self.anthropic_version = "2023-06-01"
        
        # Initialize database connection with thread safety
        self.db = ConversationDB(db_path)
        self.db_lock = db_lock  # Use the global lock for database operations
    
    def is_claude_model(self, model_name):
        """Check if the model is a Claude model"""
        return any(claude_model in model_name for claude_model in self.claude_models)
    
    def get_headers(self, model_name):
        """Get appropriate headers based on model type"""
        if self.is_claude_model(model_name):
            return {
                "x-API-key": self.api_key,
                # "anthropic-version": self.anthropic_version,
                "content-type": "application/json"
            }
        else:
            return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            }
    
    def make_request(self, prompt, model_name="o1", stream=True, max_tokens=20000, thinking=None, reasoning_effort=None):
        """Make a request to the LLM API"""
        print(f"Prompt: {prompt}")
        print(f"Model: {model_name}")
        
        if self.is_claude_model(model_name):
            return self._make_claude_request(prompt, model_name, stream, max_tokens, thinking)
        else:
            return self._make_openai_request(prompt, model_name, stream, reasoning_effort)
    
    def _make_openai_request(self, prompt, model_name, stream=True, reasoning_effort=None):
        """Make a request to OpenAI-compatible API"""
        # Record start time
        start_time = time.time()
        data = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "stream": stream
        }
        
        # Add reasoning_effort parameter if provided
        if reasoning_effort in ["low", "medium", "high"]:
            data["reasoning_effort"] = reasoning_effort
            print(f"Reasoning effort: {reasoning_effort}")
        
        if stream:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=self.get_headers(model_name),
                data=json.dumps(data),
                proxies=self.proxies,
                stream=True,
            )

            # 初始化变量
            accumulated_text = ""  # 用于累积完整响应内容
            total_tokens = 0       # 总令牌数
            completion_tokens = 0  # 完成令牌数

            # 逐行处理流式响应
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data:"):
                        try:
                            json_data = json.loads(decoded_line[5:])  # 移除 "data: " 前缀
                        except json.decoder.JSONDecodeError:
                            break
                        
                        # 检查令牌使用信息（通常在流的最后一块）
                        if "usage" in json_data and json_data["usage"] is not None:
                            completion_tokens = json_data["usage"].get("completion_tokens", 0)
                            total_tokens = json_data["usage"].get("total_tokens", 0)
                        
                        # 检查并处理内容
                        if "choices" in json_data and json_data["choices"]:
                            content = json_data["choices"][0]["delta"].get("content", "")
                            if content:  # 当内容非空时打印并累积
                                print(content, end="", flush=True)
                                accumulated_text += content
                        
            # Print token statistics if available
            if completion_tokens > 0:
                print("\n\n--- Token Statistics ---")
                print(f"Completion tokens: {completion_tokens}")
                print(f"Total tokens: {total_tokens}")
                
            # Save conversation to database with thread safety
            metadata = {
                "stream": stream,
                "reasoning_effort": reasoning_effort
            }
            
            # Use thread lock for database operations
            with self.db_lock:
                conversation_id = self.db.save_conversation(
                    model_name=model_name,
                    prompt=prompt,
                    response=accumulated_text,
                    metadata=metadata
                )
                
                # Calculate execution time
                execution_time = time.time() - start_time
                print(f"Execution time: {execution_time:.2f} seconds")
                
                # Save token usage if available
                if completion_tokens > 0:
                    self.db.save_token_usage(
                        conversation_id=conversation_id,
                        input_tokens=total_tokens - completion_tokens,
                        output_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        execution_time=execution_time
                    )
        else:
            del data["stream"]
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=self.get_headers(model_name),
                data=json.dumps(data),
                proxies=self.proxies,
            )
            if response.status_code != 200:
                print(model_name)
                print(response.text)
            response.raise_for_status()
            response_json = response.json()
            completion = response_json["choices"][0]["message"]["content"]
            print(completion)
            
            # Print token statistics
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0
            reasoning_tokens = None
            accepted_prediction_tokens = None
            rejected_prediction_tokens = None
            
            if "usage" in response_json:
                usage = response_json["usage"]
                print("\n--- Token Statistics ---")
                input_tokens = usage.get('prompt_tokens', 0)
                output_tokens = usage.get('completion_tokens', 0)
                total_tokens = usage.get('total_tokens', 0)
                print(f"Prompt tokens: {input_tokens}")
                print(f"Completion tokens: {output_tokens}")
                print(f"Total tokens: {total_tokens}")
                
                # Print detailed completion token stats if available
                if "completion_tokens_details" in usage:
                    details = usage["completion_tokens_details"]
                    reasoning_tokens = details.get('reasoning_tokens')
                    accepted_prediction_tokens = details.get('accepted_prediction_tokens')
                    rejected_prediction_tokens = details.get('rejected_prediction_tokens')
                    
                    print("Completion tokens details:")
                    print(f"  Reasoning tokens: {reasoning_tokens if reasoning_tokens is not None else 'N/A'}")
                    print(f"  Accepted prediction tokens: {accepted_prediction_tokens if accepted_prediction_tokens is not None else 'N/A'}")
                    print(f"  Rejected prediction tokens: {rejected_prediction_tokens if rejected_prediction_tokens is not None else 'N/A'}")
            
            # Calculate execution time
            execution_time = time.time() - start_time
            print(f"Execution time: {execution_time:.2f} seconds")
            
            # Save conversation to database with thread safety
            metadata = {
                "stream": stream,
                "reasoning_effort": reasoning_effort
            }
            
            # Use thread lock for database operations
            with self.db_lock:
                conversation_id = self.db.save_conversation(
                    model_name=model_name,
                    prompt=prompt,
                    response=completion,
                    metadata=metadata
                )
                
                # Save token usage if available
                if total_tokens > 0:
                    self.db.save_token_usage(
                        conversation_id=conversation_id,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                        reasoning_tokens=reasoning_tokens,
                        accepted_prediction_tokens=accepted_prediction_tokens,
                        rejected_prediction_tokens=rejected_prediction_tokens,
                        execution_time=execution_time
                    )
    
    def _make_claude_request(self, prompt, model_name, stream=True, max_tokens=20000, thinking=None, reasoning_effort=None):
        """Make a request to Claude API"""
        # Record start time
        start_time = time.time()
        url = f"{self.api_base}/messages"
        headers = self.get_headers(model_name)
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        data = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        # Add thinking if provided
        if thinking:
            data["thinking"] = thinking
        
        response = requests.post(url, headers=headers, json=data, proxies=self.proxies, stream=stream)
        
        if not (response.status_code == 200):
            print(f"Error: status_code:{response.status_code}, {response.text}")
            return -1
        
        if stream:
            # Use sseclient to handle SSE stream
            client = sseclient.SSEClient(response)
            
            # Variable to accumulate text
            accumulated_text = ""
            input_tokens = 0
            output_tokens = 0
            
            # Process SSE events
            for event in client.events():
                if event.event == "ping":
                    continue  # Ignore ping events
                
                if event.event == "error":
                    print(f"Error: {event.data}")
                    break
                
                try:
                    data = json.loads(event.data)
                    event_type = data.get("type")
                    
                    # Handle text increments
                    if event_type == "content_block_delta":
                        delta = data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text_chunk = delta.get("text", "")
                            accumulated_text += text_chunk
                            print(text_chunk, end="", flush=True)  # Output text increment immediately
                    
                    # Handle message end
                    elif event_type == "message_stop":
                        print("\n\n--- Response completed ---")
                    
                    # Extract token usage if available
                    if "usage" in data:
                        input_tokens = data["usage"].get("input_tokens", 0)
                        output_tokens = data["usage"].get("output_tokens", 0)
                
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON: {event.data}")
                except Exception as e:
                    print(f"Error processing event: {e}")
            
            print("\n\nComplete response:")
            print(accumulated_text)
            
            # Print token statistics if available
            if input_tokens > 0 or output_tokens > 0:
                print("\n--- Token Statistics ---")
                print(f"Input tokens: {input_tokens}")
                print(f"Output tokens: {output_tokens}")
                print(f"Total tokens: {input_tokens + output_tokens}")
                
            # Save conversation to database with thread safety
            metadata = {
                "stream": stream,
                "max_tokens": max_tokens,
                "thinking": thinking
            }
            
            # Use thread lock for database operations
            with self.db_lock:
                conversation_id = self.db.save_conversation(
                    model_name=model_name,
                    prompt=prompt,
                    response=accumulated_text,
                    metadata=metadata
                )
                
                # Calculate execution time
                execution_time = time.time() - start_time
                print(f"Execution time: {execution_time:.2f} seconds")
                
                # Save token usage if available
                if input_tokens > 0 or output_tokens > 0:
                    self.db.save_token_usage(
                        conversation_id=conversation_id,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        execution_time=execution_time
                    )
        else:
            # Non-streaming request
            response_json = response.json()
            content = response_json.get("content", [])
            response_text = ""
            for block in content:
                if block.get("type") == "text":
                    text = block.get("text", "")
                    response_text += text
                    print(text)
            
            # Print token statistics if available
            input_tokens = 0
            output_tokens = 0
            if "usage" in response_json:
                usage = response_json["usage"]
                print("\n--- Token Statistics ---")
                input_tokens = usage.get('input_tokens', 0)
                output_tokens = usage.get('output_tokens', 0)
                print(f"Input tokens: {input_tokens}")
                print(f"Output tokens: {output_tokens}")
                if "input_tokens" in usage and "output_tokens" in usage:
                    total = usage["input_tokens"] + usage["output_tokens"]
                    print(f"Total tokens: {total}")
                    
            # Save conversation to database with thread safety
            metadata = {
                "stream": stream,
                "max_tokens": max_tokens,
                # "reasoning_effort": reasoning_effort  # Claude API不需要此参数
            }
            
            # Use thread lock for database operations
            with self.db_lock:
                conversation_id = self.db.save_conversation(
                    model_name=model_name,
                    prompt=prompt,
                    response=response_text,
                    metadata=metadata
                )
                
                # Calculate execution time
                execution_time = time.time() - start_time
                print(f"Execution time: {execution_time:.2f} seconds")
                
                # Save token usage if available
                if input_tokens > 0 or output_tokens > 0:
                    self.db.save_token_usage(
                        conversation_id=conversation_id,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        execution_time=execution_time
                    )
