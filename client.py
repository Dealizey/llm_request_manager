"""
Simple LLM Request Client

This module provides a client for making streaming requests to a local LLM server.
It handles Server-Sent Events (SSE) and displays the streamed response.
"""

import requests
import json


class ClientTest:
    """Client for making requests to the LLM server."""
    
    def __init__(self, server_url="http://localhost:8000"):
        """
        Initialize the LLM client.
        
        Args:
            server_url (str): Base URL of the LLM server
        """
        self.server_url = server_url
        self.headers = {"Content-Type": "application/json"}
    
    def send_request(self, prompt, model="test"):
        """
        Send a request to the LLM server and process the streaming response.
        
        Args:
            prompt (str): The user's prompt/question
            model (str): The model to use for the request
            
        Returns:
            str: The complete response text
        """
        url = f"{self.server_url}/chat/completions"
        request_data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        try:
            # Make streaming request
            response = requests.post(
                url, 
                headers=self.headers, 
                json=request_data, 
                stream=True
            )
            response.raise_for_status()
            
            # Process the streaming response
            return self._process_stream(response)
            
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
    
    def _process_stream(self, response):
        """
        Process a streaming response from the server.
        
        Args:
            response: The streaming response object
            
        Returns:
            str: The complete accumulated response
        """
        accumulated_text = ""
        
        for line in response.iter_lines():
            if not line:
                continue
                
            decoded_line = line.decode('utf-8')
            if not decoded_line.startswith("data:"):
                continue
                
            try:
                # Parse the JSON data (remove "data: " prefix)
                json_data = json.loads(decoded_line[5:])
                
                if "choices" in json_data and json_data["choices"]:
                    content = json_data["choices"][0]["delta"].get("content", "")
                    if content:
                        accumulated_text += content
                        print(content, end="", flush=True)
                        
            except json.decoder.JSONDecodeError:
                print(f"Error decoding JSON: {decoded_line}")
        
        print()  # Add a newline at the end
        return accumulated_text


def main():
    """Main function to demonstrate the LLM client."""
    client = ClientTest()
    prompt = r"""在直三棱柱ABC-A_1B_1C_1中，D，E分别为AA_1，AC的中点，且AB=BC=1，AA_1=AC=\sqrt{3}
求二面角B-CD-A_1的正弦值"""
    # model = "gemini-2.0-flash-001"
    # model = "claude-3-7-sonnet-20250219"
    model = "gemini-2.0-flash-thinking-exp-01-21"
    # model = "o1"
    # model = "claude-3-5-sonnet"
    # model = "claude-3-opus"
    # model = "ark-deepseek-r1-250120 slow"
    
    print(f"Sending prompt: {prompt}")
    client.send_request(prompt, model)


if __name__ == "__main__":
    main()
