"""
Security Hardening: Obfuscated Execution and Self-Destruct Mechanism
HWID (Hardware ID) validation with self-destruct on unauthorized execution
"""

import hashlib
import platform
import subprocess
import sys
import os
import shutil
from typing import Optional
from pathlib import Path

from hean.config import settings
from hean.logging import get_logger

logger = get_logger(__name__)


class SecurityHardening:
    """
    Security Hardening: Obfuscated execution and self-destruct mechanism.
    Validates HWID and wipes sensitive data if unauthorized.
    """
    
    def __init__(self, authorized_hwids: Optional[list[str]] = None):
        """
        Initialize security hardening.
        
        Args:
            authorized_hwids: List of authorized hardware IDs. If None, uses settings.
        """
        self.authorized_hwids = authorized_hwids or getattr(settings, 'authorized_hwids', [])
        self.current_hwid = self._get_hardware_id()
        self.api_keys_path = Path.home() / '.hean' / 'api_keys'
        self.shared_memory_path = Path('/dev/shm/hean_shared_memory')
        
    def _get_hardware_id(self) -> str:
        """
        Generate Hardware ID (HWID) from system characteristics.
        
        Returns:
            SHA256 hash of system identifiers
        """
        try:
            # Collect system identifiers
            identifiers = []
            
            # CPU ID
            try:
                if platform.system() == 'Linux':
                    with open('/proc/cpuinfo', 'r') as f:
                        for line in f:
                            if 'Serial' in line or 'Model' in line:
                                identifiers.append(line.strip())
                elif platform.system() == 'Darwin':  # macOS
                    result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        identifiers.append(result.stdout.strip())
            except Exception:
                pass
            
            # MAC address (first network interface)
            try:
                if platform.system() == 'Linux':
                    result = subprocess.run(['cat', '/sys/class/net/eth0/address'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        identifiers.append(result.stdout.strip())
                elif platform.system() == 'Darwin':
                    result = subprocess.run(['ifconfig', 'en0'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        for line in result.stdout.split('\n'):
                            if 'ether' in line:
                                identifiers.append(line.strip())
                                break
            except Exception:
                pass
            
            # Machine hostname
            identifiers.append(platform.node())
            
            # Platform info
            identifiers.append(f"{platform.system()}-{platform.machine()}")
            
            # Combine and hash
            combined = '|'.join(identifiers)
            hwid = hashlib.sha256(combined.encode()).hexdigest()
            
            logger.debug(f"Generated HWID: {hwid[:16]}...")
            return hwid
            
        except Exception as e:
            logger.error(f"Error generating HWID: {e}")
            # Fallback: use hostname only
            return hashlib.sha256(platform.node().encode()).hexdigest()
    
    def validate_hwid(self) -> bool:
        """
        Validate current HWID against authorized list.
        
        Returns:
            True if authorized, False otherwise
        """
        if not self.authorized_hwids:
            # If no authorized HWIDs set, allow execution (development mode)
            logger.warning("No authorized HWIDs configured. Allowing execution (dev mode).")
            return True
        
        if self.current_hwid in self.authorized_hwids:
            logger.info(f"HWID validation passed: {self.current_hwid[:16]}...")
            return True
        
        logger.error(f"UNAUTHORIZED HWID DETECTED: {self.current_hwid[:16]}...")
        return False
    
    def self_destruct(self) -> None:
        """
        Self-destruct mechanism: Wipe API keys and clear shared memory.
        Called when unauthorized HWID is detected.
        """
        logger.critical("SELF-DESTRUCT INITIATED: Unauthorized execution detected")
        
        try:
            # Wipe API keys file
            if self.api_keys_path.exists():
                logger.critical(f"Wiping API keys: {self.api_keys_path}")
                # Overwrite with random data before deletion
                with open(self.api_keys_path, 'wb') as f:
                    f.write(os.urandom(1024))  # Overwrite with random bytes
                os.remove(self.api_keys_path)
            
            # Clear shared memory
            if self.shared_memory_path.exists():
                logger.critical(f"Clearing shared memory: {self.shared_memory_path}")
                try:
                    if self.shared_memory_path.is_file():
                        os.remove(self.shared_memory_path)
                    else:
                        shutil.rmtree(self.shared_memory_path)
                except Exception as e:
                    logger.error(f"Error clearing shared memory: {e}")
            
            # Clear environment variables
            sensitive_vars = ['BYBIT_API_KEY', 'BYBIT_SECRET', 'OPENAI_API_KEY', 
                            'ANTHROPIC_API_KEY', 'HEAN_API_KEY']
            for var in sensitive_vars:
                if var in os.environ:
                    logger.critical(f"Clearing environment variable: {var}")
                    os.environ.pop(var, None)
            
            # Clear config directory
            config_dir = Path.home() / '.hean'
            if config_dir.exists():
                logger.critical(f"Clearing config directory: {config_dir}")
                try:
                    # Only remove sensitive files, not the entire directory
                    for file in config_dir.glob('*.key'):
                        file.unlink()
                    for file in config_dir.glob('*.secret'):
                        file.unlink()
                    for file in config_dir.glob('api_keys*'):
                        file.unlink()
                except Exception as e:
                    logger.error(f"Error clearing config directory: {e}")
            
            logger.critical("Self-destruct completed. Sensitive data wiped.")
            
        except Exception as e:
            logger.error(f"Error during self-destruct: {e}")
    
    def initialize(self) -> bool:
        """
        Initialize security hardening: Validate HWID and self-destruct if unauthorized.
        
        Returns:
            True if authorized and initialization successful, False otherwise
        """
        if not self.validate_hwid():
            logger.critical("UNAUTHORIZED EXECUTION DETECTED")
            self.self_destruct()
            sys.exit(1)  # Exit immediately after self-destruct
        
        logger.info("Security hardening initialized successfully")
        return True
    
    def get_hwid(self) -> str:
        """
        Get current hardware ID (for configuration purposes).
        
        Returns:
            Current HWID
        """
        return self.current_hwid


# Global security instance
_security: Optional[SecurityHardening] = None


def initialize_security(authorized_hwids: Optional[list[str]] = None) -> bool:
    """
    Initialize security hardening globally.
    
    Args:
        authorized_hwids: List of authorized hardware IDs
        
    Returns:
        True if authorized, False otherwise (will exit if unauthorized)
    """
    global _security
    _security = SecurityHardening(authorized_hwids=authorized_hwids)
    return _security.initialize()


def get_current_hwid() -> Optional[str]:
    """
    Get current hardware ID.
    
    Returns:
        Current HWID, or None if not initialized
    """
    if _security is None:
        return None
    return _security.get_hwid()