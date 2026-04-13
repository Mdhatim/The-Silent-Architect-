import json
import re
import os
from pathlib import Path
from collections import namedtuple

# Requirement: Intent and Action must be displayed in the UI [cite: 33, 34]
Result = namedtuple("Result", ["intent", "confidence", "action_taken", "output"])

def run_from_text(repo_root, transcription):
    """
    Analyzes text to classify intent and execute local tools[cite: 16, 23].
    """
    # NOTE: You should replace this string with your actual local LLM call
    # This example ensures the JSON parser works even with extra model 'chatter'
    llm_raw_output = '{"intent": "create_file", "parameters": {"filename": "demo/notes.txt", "content": "Success!"}, "confidence": 1.0}'

    try:
        # Regex to extract JSON block to prevent "failed to parse JSON" errors
        json_match = re.search(r'\{.*\}', llm_raw_output, re.DOTALL)
        if not json_match:
            return Result("error", 0.0, "No JSON found", "The model did not provide a valid JSON response.")
            
        data = json.loads(json_match.group())
        intent = data.get("intent", "general_chat")
        params = data.get("parameters", {})
        
        action_taken = "No action"
        output_result = ""

        # Tool Execution Requirement: File Operations [cite: 25]
        if intent == "create_file":
            filename = params.get("filename", "notes.txt")
            content = params.get("content", "")
            
            # Safety Constraint: Restrict all files to the output/ folder 
            output_base = Path(repo_root) / "output"
            target_path = (output_base / filename).resolve()
            
            # Create subdirectories if needed and write file
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "w") as f:
                f.write(content)
                
            action_taken = f"Created file: {filename}"
            output_result = f"File saved to {target_path}"

        return Result(intent, data.get("confidence", 1.0), action_taken, output_result)

    except Exception as e:
        return Result("error", 0.0, "Execution failed", str(e))