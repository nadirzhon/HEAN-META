#!/usr/bin/env python3
"""Test Gemini agent generation."""

import os
import sys

# Set API key
os.environ["GEMINI_API_KEY"] = "AIzaSyB0s3fs3esS0CMWyUBlVH8DfkRst-Z-CSQ"

from hean.agent_generation.generator import AgentGenerator

print("Testing Gemini agent generation...")
print(f"GEMINI_API_KEY set: {bool(os.getenv('GEMINI_API_KEY'))}")

try:
    generator = AgentGenerator()
    
    if generator.llm_client is None:
        print("❌ No LLM client configured")
        sys.exit(1)
    
    client_type = type(generator.llm_client).__name__
    client_module = type(generator.llm_client).__module__
    is_gemini = "generativeai" in client_module or "genai" in str(client_module)
    
    print(f"✓ LLM Client: {client_type}")
    print(f"✓ Module: {client_module}")
    print(f"✓ Is Gemini: {is_gemini}")
    
    if is_gemini:
        print("\n✓ Gemini API successfully initialized!")
        print("✓ Agent generation system is ready to use Gemini as catalyst")
    else:
        print(f"\n⚠ Using {client_type} instead of Gemini")
    
    sys.exit(0)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
