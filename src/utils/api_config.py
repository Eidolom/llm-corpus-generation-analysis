"""
API Configuration Loader
Handles loading API keys from .env file and environment variables.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in Python path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def load_api_key(key_name: str = "GOOGLE_API_KEY", env_file: str = ".env") -> str:
    """
    Load API key from .env file or environment variables.
    
    Priority:
    1. .env file in project root
    2. Environment variable
    3. Raises error if not found
    
    Args:
        key_name: Name of the environment variable (default: GOOGLE_API_KEY)
        env_file: Path to .env file (default: .env in project root)
    
    Returns:
        API key string
        
    Raises:
        RuntimeError: If API key is not found
    """
    # Try to load from .env file first
    if load_dotenv:
        env_path = Path(env_file)
        if env_path.exists():
            load_dotenv(env_path, override=False)
    
    # Get from environment
    api_key = os.getenv(key_name)
    
    if not api_key or api_key.strip() == "":
        raise RuntimeError(
            f"Missing {key_name} configuration.\n"
            f"Please create a .env file in the project root with:\n"
            f"{key_name}=your_actual_api_key_here\n"
            f"\nAlternatively, set the environment variable:\n"
            f"PowerShell: $env:{key_name} = 'your_key'\n"
            f"Bash: export {key_name}='your_key'"
        )
    
    return api_key.strip()
