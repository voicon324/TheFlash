from typing import Any, List, Optional, Dict
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field
import time
import logging
import requests
from config import api_config, pipeline_config
from math_tool import math_tool

logger = logging.getLogger(__name__)

class VNPTLLM(LLM):
    """VNPT AI LLM wrapper for LangChain."""
    
    model_name: str = Field(default_factory=lambda: api_config.get_llm_model(False))
    use_large: bool = False
    use_oss: bool = False
    api_url: str = Field(default_factory=lambda: api_config.get_llm_url(False))
    headers: Dict[str, str] = Field(default_factory=lambda: api_config.get_headers('small'))
    stop_sequences: List[str] = Field(default_factory=list)
    
    def __init__(self, use_large: bool = False, use_oss: bool = False, stop: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.use_large = use_large
        self.use_oss = use_oss
        self.api_url = api_config.get_llm_url(use_large, use_oss)
        self.model_name = api_config.get_llm_model(use_large, use_oss)
        
        # Get appropriate headers
        if use_oss:
            self.headers = api_config.get_headers('oss')
        else:
            self.headers = api_config.get_headers('large' if use_large else 'small')
        self.stop_sequences = stop or []

    @property
    def _llm_type(self) -> str:
        return "vnpt_llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given prompt."""
        
        # Construct message format expected by API
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        base_wait = 5
        attempt = 0
        
        while True:
            attempt += 1
            
            # Basic rate limiting (simple sleep)
            # In a real scenario, we might want a shared token bucket, 
            # but for now we rely on the retry logic to handle rate limits.
            
            # Merge stop sequences from init and call
            final_stop = list(self.stop_sequences)
            if stop:
                final_stop.extend(stop)
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": kwargs.get("temperature", pipeline_config.TEMPERATURE),
                "max_completion_tokens": kwargs.get("max_tokens", pipeline_config.MAX_TOKENS),
                "seed": kwargs.get("seed", pipeline_config.SEED),
            }
            
            # Add stop sequences if any
            if final_stop:
                payload["stop"] = final_stop
            
            try:
                response = requests.post(
                    self.api_url, 
                    headers=self.headers, 
                    json=payload, 
                    timeout=300
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        content = data["choices"][0]["message"]["content"]
                        # Process math if needed
                        content = math_tool.process_markdown(content)
                        return content
                
                # Handle errors
                is_soft_error = False
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if "error" in data:
                            logger.warning(f"API returned error in 200 OK: {data['error']}")
                            is_soft_error = True
                            return ""
                    except:
                        pass

                if response.status_code in [401, 429, 500, 502, 503, 504] or is_soft_error:
                    wait_time = min(120, base_wait * (2 ** (min(attempt, 6) - 1)))
                    logger.warning(f"API attempt {attempt} failed ({response.status_code} - Soft Error: {is_soft_error}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                logger.error(f"API request failed: {response.status_code} {response.text}")
                raise ValueError(f"API request failed with status {response.status_code}")
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Network error on attempt {attempt}: {e}")
                wait_time = min(120, base_wait * (2 ** (min(attempt, 6) - 1)))
                time.sleep(wait_time)
                continue
            except KeyboardInterrupt:
                print("\nInterrupted by user during API call")
                raise
