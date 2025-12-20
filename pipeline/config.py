# VNPT AI Pipeline Configuration
import os
import json
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from pipeline directory
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)


@dataclass
class APIConfig:
    """API Configuration for VNPT AI Services"""
    BASE_URL: str = "https://api.idg.vnpt.vn"
    
    # Endpoints
    LLM_SMALL_ENDPOINT: str = "/data-service/v1/chat/completions/vnptai-hackathon-small"
    LLM_LARGE_ENDPOINT: str = "/data-service/v1/chat/completions/vnptai-hackathon-large"
    EMBEDDING_ENDPOINT: str = "/data-service/vnptai-hackathon-embedding"
    
    # OSS 20B Model Configuration (via ngrok)
    OSS_BASE_URL: str = "https://f380fde127e6.ngrok-free.app"  # Update this with your ngrok URL
    OSS_ENDPOINT: str = "/v1/chat/completions"
    OSS_MODEL: str = "gpt-oss:20b"
    
    # Model names
    LLM_SMALL_MODEL: str = "vnptai_hackathon_small"
    LLM_LARGE_MODEL: str = "vnptai_hackathon_large"
    EMBEDDING_MODEL: str = "vnptai_hackathon_embedding"
    
    # API Credentials - Default fallback
    DEFAULT_AUTH: str = os.getenv("VNPT_AUTHORIZATION", "")
    DEFAULT_TOKEN_ID: str = os.getenv("VNPT_TOKEN_ID", "")
    DEFAULT_TOKEN_KEY: str = os.getenv("VNPT_TOKEN_KEY", "")
    
    # Specific credentials
    SMALL_AUTH: str = ""
    SMALL_TOKEN_ID: str = ""
    SMALL_TOKEN_KEY: str = ""
    
    LARGE_AUTH: str = ""
    LARGE_TOKEN_ID: str = ""
    LARGE_TOKEN_KEY: str = ""
    
    EMBEDDING_AUTH: str = ""
    EMBEDDING_TOKEN_ID: str = ""
    EMBEDDING_TOKEN_KEY: str = ""
    
    # Helper for backward compatibility
    @property
    def AUTHORIZATION(self): return self.DEFAULT_AUTH
    @property
    def TOKEN_ID(self): return self.DEFAULT_TOKEN_ID
    @property
    def TOKEN_KEY(self): return self.DEFAULT_TOKEN_KEY

    def __post_init__(self):
        # Initialize specific credentials with defaults
        self.SMALL_AUTH = self.DEFAULT_AUTH
        self.SMALL_TOKEN_ID = self.DEFAULT_TOKEN_ID
        self.SMALL_TOKEN_KEY = self.DEFAULT_TOKEN_KEY
        
        self.LARGE_AUTH = self.DEFAULT_AUTH
        self.LARGE_TOKEN_ID = self.DEFAULT_TOKEN_ID
        self.LARGE_TOKEN_KEY = self.DEFAULT_TOKEN_KEY
        
        self.EMBEDDING_AUTH = self.DEFAULT_AUTH
        self.EMBEDDING_TOKEN_ID = self.DEFAULT_TOKEN_ID
        self.EMBEDDING_TOKEN_KEY = self.DEFAULT_TOKEN_KEY
        
        # Try to load from api-keys.json
        self._load_from_json()

    def _load_from_json(self):
        # Look for api-keys.json in parent directory or current directory
        paths_to_check = [
            Path(__file__).parent.parent / "api-keys.json",
            Path("api-keys.json")
        ]
        
        json_path = None
        for p in paths_to_check:
            if p.exists():
                json_path = p
                break
        
        if not json_path:
            return

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                api_name = item.get("llmApiName", "")
                auth = item.get("authorization", "")
                
                # Strip Bearer if present
                if auth.startswith("Bearer "):
                    auth = auth[7:].strip()
                
                token_id = item.get("tokenId", "")
                token_key = item.get("tokenKey", "")
                
                if api_name == "LLM small":
                    self.SMALL_AUTH = auth
                    self.SMALL_TOKEN_ID = token_id
                    self.SMALL_TOKEN_KEY = token_key
                elif api_name == "LLM large":
                    self.LARGE_AUTH = auth
                    self.LARGE_TOKEN_ID = token_id
                    self.LARGE_TOKEN_KEY = token_key
                elif "embedings" in api_name or "embeddings" in api_name:
                    self.EMBEDDING_AUTH = auth
                    self.EMBEDDING_TOKEN_ID = token_id
                    self.EMBEDDING_TOKEN_KEY = token_key
                    
            print(f"Loaded credentials from {json_path}")
            
        except Exception as e:
            print(f"Error loading api-keys.json: {e}")

    def get_headers(self, model_type: str = "small") -> dict:
        """Get authentication headers for API requests"""
        # OSS model doesn't need authentication, but needs ngrok header
        if model_type == "oss":
            return {
                "Content-Type": "application/json",
                "ngrok-skip-browser-warning": "true"
            }
        
        if model_type == "small":
            auth, tid, tkey = self.SMALL_AUTH, self.SMALL_TOKEN_ID, self.SMALL_TOKEN_KEY
        elif model_type == "large":
            auth, tid, tkey = self.LARGE_AUTH, self.LARGE_TOKEN_ID, self.LARGE_TOKEN_KEY
        elif model_type == "embedding":
            auth, tid, tkey = self.EMBEDDING_AUTH, self.EMBEDDING_TOKEN_ID, self.EMBEDDING_TOKEN_KEY
        else:
            auth, tid, tkey = self.DEFAULT_AUTH, self.DEFAULT_TOKEN_ID, self.DEFAULT_TOKEN_KEY
            
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {auth}",
            "Token-id": tid,
            "Token-key": tkey,
        }
    
    def get_llm_url(self, use_large: bool = False, use_oss: bool = False) -> str:
        """Get LLM endpoint URL"""
        if use_oss:
            return f"{self.OSS_BASE_URL}{self.OSS_ENDPOINT}"
        endpoint = self.LLM_LARGE_ENDPOINT if use_large else self.LLM_SMALL_ENDPOINT
        return f"{self.BASE_URL}{endpoint}"
    
    def get_embedding_url(self) -> str:
        """Get embedding endpoint URL"""
        return f"{self.BASE_URL}{self.EMBEDDING_ENDPOINT}"
    
    def get_llm_model(self, use_large: bool = False, use_oss: bool = False) -> str:
        """Get LLM model name"""
        if use_oss:
            return self.OSS_MODEL
        return self.LLM_LARGE_MODEL if use_large else self.LLM_SMALL_MODEL


@dataclass
class PipelineConfig:
    """Pipeline Configuration"""
    # Data paths - support both local and Docker environments
    # Docker: /code/private_test.json
    # Local: /home/hkduy/workplace/VNPT_AI/AInicorns_TheBuilder_public/data
    DATA_DIR: str = os.getenv("DATA_DIR", "/home/hkduy/workplace/VNPT_AI/AInicorns_TheBuilder_public/data")
    VAL_FILE: str = "val.json"
    TEST_FILE: str = os.getenv("TEST_FILE", "test.json")
    
    # Output paths
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "/home/hkduy/workplace/VNPT_AI/pipeline/outputs")
    EMBEDDINGS_FILE: str = "embeddings.pkl"
    RESULTS_FILE: str = "results.json"
    
    # Pipeline settings
    BATCH_SIZE: int = 10
    TOP_K: int = 3  # Top-k contexts for retrieval
    MAX_TOKENS: int = 2048
    MAX_INPUT_TOKENS: int = 30000
    TEMPERATURE: float = 0.0
    SEED: int = 42
    
    # Rate limiting (seconds between requests)
    LLM_RATE_LIMIT: float = 3.0
    EMBEDDING_RATE_LIMIT: float = 0.2
    
    # ReAct Agent settings (optional, default off)
    USE_REACT_AGENT: bool = False
    REACT_MAX_STEPS: int = 20
    REACT_TEMPERATURE: float = 0.1
    
    # Chain of Thought settings (optional, default off)
    USE_CHAIN_OF_THOUGHT: bool = False
    
    # RAG Settings
    ENABLE_RAG: bool = False
    
    # Auto-enable CoT for questions with context (improves accuracy)
    AUTO_COT_FOR_CONTEXT: bool = True

    # Context Refinement Settings
    REFINE_CONTEXT: bool = False
    REFINE_CONTEXT_THRESHOLD: int = 1500  # Chars. Triggers refinement if context is longer than this
    REFINE_CHUNK_SIZE: int = 1200
    REFINE_CHUNK_OVERLAP: int = 100
    REFINE_TOP_K: int = 5
    
    # Self-Correction Settings



# Global config instances
api_config = APIConfig()
pipeline_config = PipelineConfig()
