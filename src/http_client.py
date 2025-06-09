import requests
import time
from typing import Any

class LeanHTTPClient:
    def __init__(self, url: str):
        """Initialize with server URL"""
        self.base_url = url.rstrip('/')
        self.verify_url = f"{self.base_url}/verify"
        self.status_url = f"{self.base_url}/status"
        self.reinitialize_url = f"{self.base_url}/reinitialize"
        
    def get_status(self) -> tuple[bool, dict[str, Any]]:
        """
        Get REPL status from server
        Returns: (success, result)
        """
        try:
            # Check REPL status
            response = requests.get(
                self.status_url,
                timeout=5
            )
            
            if response.status_code != 200:
                error_msg = f"Server error: {response.status_code}"
                return False, {"error": error_msg}
                
            status = response.json()
            return True, status
            
        except Exception as e:
            error_msg = f"Error checking status: {str(e)}"
            return False, {"error": error_msg}
    
    def check_theorem(self, theorem: str, timeout: float=20) -> tuple[bool, dict[str, Any]]:
        """
        Send a theorem to the server and get the response
        
        Args:
            theorem: The theorem to check
            timeout: Timeout in seconds, None for no timeout
            
        Returns: (success, result)
        - If successful: (True, response_json)
        - If error: (False, {"error": error_message})
        """
        try:
            # Check theorem with specified timeout
            response = requests.post(
                self.verify_url,
                json={
                    'theorem': theorem,
                    'timeout': timeout
                },
                timeout=timeout + 5  # Add a small buffer to the client timeout
            )
            
            if response.status_code != 200:
                error_msg = f"Server error: {response.status_code}"
                return False, {"error": error_msg}
                
            result = response.json()
            
            # Check if there was an error with the REPL
            if "error" in result:
                # Check if REPL needs reinitialization
                return False, result
                
            # Theorem check completed successfully
            return True, result
            
        except requests.Timeout:
            error_msg = "Request timed out"
            return False, {"error": error_msg}
        except requests.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            return False, {"error": error_msg}
        except Exception as e:
            error_msg = str(e)
            return False, {"error": error_msg}
    
    def reinitialize_repl(self) -> tuple[bool, dict[str, Any]]:
        """
        Explicitly request the server to reinitialize the REPL
        
        Returns: (success, result)
        - If successful: (True, response_json)
        - If error: (False, {"error": error_message})
        """
        try:
            # Request REPL reinitialization
            response = requests.post(
                self.reinitialize_url,
                timeout=60  # Initialization can take time
            )
            
            if response.status_code != 200:
                error_msg = f"Server error during reinitialization: {response.status_code}"
                return False, {"error": error_msg}
                
            result = response.json()
            if result["status"] == "success":
                return True, result
            else:
                return False, result
                
        except requests.Timeout:
            error_msg = "Reinitialization request timed out"
            return False, {"error": error_msg}
        except Exception as e:
            error_msg = f"Error during reinitialization: {str(e)}"
            return False, {"error": error_msg}
    
    def wait_for_repl_ready(self, max_attempts: int = 10, delay: int = 3) -> bool:
        """
        Wait for the REPL to become ready, checking status at regular intervals
        
        Args:
            max_attempts: Maximum number of status checks
            delay: Seconds to wait between checks
            
        Returns:
            bool: True if REPL became ready, False if it didn't within the allocated attempts
        """
        for attempt in range(1, max_attempts + 1):
            success, status = self.get_status()
            
            if success and status.get("ready", False):
                return True
                
            if attempt < max_attempts:
                time.sleep(delay)
        return False
    
    def save_file_on_server(self, filename: str, content: Any) -> tuple[bool, dict]:
        """
        Sends content to be saved as a file on the server.
        The content can be a string or a dict/list that can be converted to JSON.
        Does not interact with the Lean REPL.
        """
        try:
            # The URL would point to our new /save_file endpoint
            save_file_url = f"{self.base_url}/save_file"

            # NO CHANGE HERE! `requests` handles serializing the dictionary
            # passed in the 'content' field automatically.
            response = requests.post(
                save_file_url,
                json={'filename': filename, 'content': content},
                timeout=15 
            )

            if response.status_code != 200:
                return False, {"error": f"Server error: {response.status_code}", "details": response.text}

            return True, response.json()

        except Exception as e:
            return False, {"error": f"Request failed: {str(e)}"}