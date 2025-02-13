# models.py
import logging
import os
import json
import datetime
import traceback
import sqlite3
import requests
from abc import ABC, abstractmethod
from openai import OpenAI
from llama_cpp import Llama
from typing import Optional, Dict, Any
import warnings
import re
import tempfile
import subprocess

logger = logging.getLogger(__name__)

class BaseAIModel(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class OpenAIModel(BaseAIModel):
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content

class DeepseekLocalModel(BaseAIModel):
    def __init__(self, model_path: str):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,        # Can keep larger context
            n_threads=int(os.getenv('THREADS', '4')),
            n_gpu_layers=0,    # Still CPU-only
            use_mmap=True,     # Memory mapping for efficiency
            verbose=False
        )

    def generate(self, prompt: str) -> str:
        response = self.llm(
            prompt,
            max_tokens=1024,   # Longer responses
            temperature=0.2,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.2
        )
        return response["choices"][0]["text"]

class ClaudeModel(BaseAIModel):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1/messages"

    def generate(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        data = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(self.base_url, headers=headers, json=data)
        return response.json()["content"][0]["text"]

class LlamaModel(BaseAIModel):
    def __init__(self, model_path: str):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=20
        )

    def generate(self, prompt: str) -> str:
        response = self.llm(prompt, max_tokens=1000)
        return response["choices"][0]["text"]

class MemoryManager:
    def __init__(self):
        try:
            self.conn = sqlite3.connect('memory.db')
            self.conn.row_factory = sqlite3.Row
            self._init_db()
            logger.info("Database connection established")
        except Exception as e:
            logger.critical(f"Database connection failed: {str(e)}", exc_info=True)
            raise

    def get_memories(self, limit: int = 10):
        try:
            cursor = self.conn.execute('''SELECT * FROM memories ORDER BY id DESC LIMIT ?''', (limit,))
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Error fetching memories: {str(e)}")
            return []

    def _init_db(self):
            self.conn.execute('''CREATE TABLE IF NOT EXISTS memories
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                task TEXT,
                solution TEXT,
                success BOOLEAN,
                verification_result TEXT,
                reflections TEXT)''')
            self.conn.commit()

    def store_memory(self, memory: dict):
        try:
            # Ensure proper JSON serialization
            memory.setdefault('verification_result', {})
            memory.setdefault('reflections', [])

            self.conn.execute('''INSERT INTO memories
                (timestamp, task, solution, success, verification_result, reflections)
                VALUES (?, ?, ?, ?, ?, ?)''',
                (
                    memory['timestamp'],
                    memory['task'],
                    memory['solution'],
                    memory['success'],
                    json.dumps(memory['verification_result']),
                    json.dumps(memory['reflections'])
                ))
            self.conn.commit()
            logger.debug(f"Memory stored: {memory['task'][:20]}...")
        except Exception as e:
            logger.error(f"Memory storage failed: {str(e)}", exc_info=True)
            raise

    def get_memories(self, limit: int = 10):
        cursor = self.conn.execute('''SELECT * FROM memories ORDER BY id DESC LIMIT ?''', (limit,))
        memories = []
        for row in cursor.fetchall():
            memory = dict(row)
            # Convert JSON strings back to objects
            memory['verification_result'] = json.loads(memory.get('verification_result', '{}'))
            memory['reflections'] = json.loads(memory.get('reflections', '[]'))
            memories.append(memory)
        return memories

class SoftwareEngineerAgent:
    def __del__(self):
        if hasattr(self, 'memory_manager'):
            self.memory_manager.conn.close()

    def __init__(self, model: BaseAIModel):
        self.current_step = "Initializing"
        self.model = model
        self.memory_manager = MemoryManager()
        self.reflection_interval = 3
        logger.info("Agent initialized")

    def process_task(self, problem: str) -> dict:
        self.current_step = "Analyzing requirements"
        memory_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "task": problem,
            "success": False,
            "solution": "",
            "verification_result": {},
            "reflections": []
        }

        try:
            self.current_step = "Generating solution"
            logger.debug(f"Starting task processing: {problem[:40]}...")
            solution = self.generate_solution(problem)
            memory_entry["solution"] = solution

            verification_result = self.verify_solution(problem, solution)
            memory_entry["verification_result"] = verification_result
            memory_entry["success"] = verification_result["success"]

            self.memory_manager.store_memory(memory_entry)

            if len(self.memory_manager.get_memories()) % self.reflection_interval == 0:
                reflections = self.reflect()
                memory_entry["reflections"] = reflections

            logger.info(f"Task processed successfully: {memory_entry['success']}")
            return memory_entry

        except Exception as e:
            logger.error(f"Task processing failed: {str(e)}", exc_info=True)
            memory_entry["error"] = str(e)
            self.memory_manager.store_memory(memory_entry)
            return memory_entry

    def generate_solution(self, problem: str) -> str:
        prompt = f"""You are a senior software engineer. Solve this problem:

        {problem}

        Provide only the code solution with minimal explanations"""
        return self.model.generate(prompt)


    def verify_solution(self, problem: str, solution: str, language: str = "python") -> dict:
        """
        Verifies the generated solution by:
        1. Asking the model to generate 3 test cases.
        2. Executing the solution code in the appropriate runtime.

        For each language, the assumptions are:
        - **Python:** The solution code defines at least one callable (the first found) that implements the solution.
        - **PHP/Laravel:** The solution code defines a function named `solutionFunction`.
        - **Node-based (javascript, vue, react, nuxt, next, node, nodejs):** The solution code exports a function named `solutionFunction`.
        """
        # Build a test case prompt (same for all languages)
        test_prompt = f"""Given this problem:
            {problem}
            And this solution:
            {solution}
            Generate exactly 3 test cases with inputs and expected outputs.
            Return only a valid JSON array of exactly 3 objects, each with the following format:
            {{
                "inputs": [list of input values],
                "expected": expected_output_value
            }}"""

        results = {"success": False, "tests": []}

        try:
            # Generate and clean test cases response
            test_response = self.model.generate(test_prompt).strip()
            if test_response.startswith("```"):
                test_response = re.sub(r'^```(?:python|php|javascript|vue|react|nuxt|next)?', '', test_response)
                test_response = re.sub(r'```$', '', test_response)
            test_response = test_response.strip()
            # Try to extract a JSON array using regex
            match = re.search(r'\[.*\]', test_response, re.DOTALL)
            json_text = match.group(0) if match else test_response
            test_cases = json.loads(json_text)
            if not isinstance(test_cases, list) or len(test_cases) != 3:
                raise ValueError("Invalid test cases format: Expected a list of 3 test cases.")

            # Clean the solution code (remove markdown fences and extra whitespace)
            cleaned_solution = solution.strip()
            if cleaned_solution.startswith("```"):
                cleaned_solution = re.sub(r'^```(?:python|php|javascript|vue|react|nuxt|next)?', '', cleaned_solution)
                cleaned_solution = re.sub(r'```$', '', cleaned_solution)
            cleaned_solution = cleaned_solution.strip()

            all_success = True

            # Process each test case
            for idx, test in enumerate(test_cases, 1):
                test_result = {
                    "test_case": idx,
                    "success": False,
                    "error": None,
                    "input": test.get('inputs'),
                    "expected": test.get('expected'),
                    "received": None
                }

                # Ensure inputs is a list
                inputs = test.get('inputs')
                if not isinstance(inputs, list):
                    inputs = [inputs]

                lang = language.lower()
                if lang == "python":
                    # Execute in a controlled namespace
                    namespace = {"__name__": "solution_module"}
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        exec(cleaned_solution, namespace)
                    # Retrieve the first callable (excluding built-ins)
                    func = next((obj for name, obj in namespace.items() if callable(obj) and name != '__builtins__'), None)
                    if not func:
                        raise ValueError("No valid function found in Python solution")
                    try:
                        result = func(*inputs)
                        test_result["received"] = result
                        test_result["success"] = (result == test.get('expected'))
                    except Exception as e:
                        test_result["error"] = str(e)

                elif lang in ("php", "laravel"):
                    # Write PHP solution to a temporary file.
                    # Expect a function named solutionFunction.
                    with tempfile.NamedTemporaryFile(suffix=".php", delete=False) as tmp:
                        tmp.write(cleaned_solution.encode("utf-8"))
                        tmp_filename = tmp.name
                    try:
                        input_json = json.dumps(inputs)
                        cmd = [
                            'php', '-r',
                            f'require "{tmp_filename}"; echo json_encode(solutionFunction({input_json}));'
                        ]
                        output = subprocess.check_output(cmd, universal_newlines=True)
                        result = json.loads(output) if output.strip() else None
                        test_result["received"] = result
                        test_result["success"] = (result == test.get('expected'))
                    except Exception as e:
                        test_result["error"] = str(e)
                    finally:
                        os.remove(tmp_filename)

                elif lang in ("node", "nodejs", "javascript", "vue", "react", "nuxt", "next"):
                    # Write JS solution to a temporary file.
                    # Expect an exported function named solutionFunction.
                    with tempfile.NamedTemporaryFile(suffix=".js", delete=False) as tmp:
                        tmp.write(cleaned_solution.encode("utf-8"))
                        tmp_filename = tmp.name
                    try:
                        input_json = json.dumps(inputs)
                        cmd = [
                            'node', '-e',
                            f'const sol = require("{tmp_filename}"); console.log(JSON.stringify(sol.solutionFunction({input_json})));'
                        ]
                        output = subprocess.check_output(cmd, universal_newlines=True)
                        result = json.loads(output) if output.strip() else None
                        test_result["received"] = result
                        test_result["success"] = (result == test.get('expected'))
                    except Exception as e:
                        test_result["error"] = str(e)
                    finally:
                        os.remove(tmp_filename)
                else:
                    raise ValueError("Unsupported language")

                if not test_result["success"]:
                    all_success = False
                results["tests"].append(test_result)

            results["success"] = all_success

        except json.JSONDecodeError as e:
            results["error"] = f"Invalid JSON format: {str(e)}"
        except Exception as e:
            results["error"] = str(e)

        return results


    def reflect(self) -> list:
        reflections = []
        memories = self.memory_manager.get_memories(self.reflection_interval)

        try:
            success_rate = sum(1 for m in memories if m['success']) / len(memories)
            reflections.append(f"Recent success rate: {success_rate*100:.1f}%")
        except ZeroDivisionError:
            reflections.append("No recent memories for reflection")

        # Error analysis with safe access
        common_errors = {}
        for memory in memories:
            if not memory.get('success'):
                error = memory.get('verification_result', {}).get('error')
                error_msg = error or 'Unknown error' if error else 'Unknown error'
                common_errors[error_msg] = common_errors.get(error_msg, 0) + 1

        if common_errors:
            reflections.append("Most common errors:")
            for error, count in sorted(common_errors.items(), key=lambda x: x[1], reverse=True)[:3]:
                reflections.append(f"- {error} ({count} occurrences)")

        return reflections

    def get_progress(self):
        return {"step": self.current_step, "complete": False}


def create_model(model_type: str) -> BaseAIModel:
    try:
        logger.info(f"Creating model: {model_type}")
        if model_type == "OpenAI":
            if not (api_key := os.getenv('OPENAI_API_KEY')):
                raise ValueError("OPENAI_API_KEY not set in .env")
            return OpenAIModel(api_key)

        elif model_type == "Deepseek":
            model_path = os.getenv('DEEPSEEK_MODEL_PATH')
            if not model_path:
                raise ValueError("DEEPSEEK_MODEL_PATH not set in .env")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Deepseek model not found at: {model_path}")
            return DeepseekLocalModel(model_path)

        elif model_type == "Claude":
            if not (api_key := os.getenv('CLAUDE_API_KEY')):
                raise ValueError("CLAUDE_API_KEY not set in .env")
            return ClaudeModel(api_key)

        elif model_type == "Llama":
            model_path = os.getenv('LLAMA_MODEL_PATH')
            if not model_path:
                raise ValueError("LLAMA_MODEL_PATH not set in .env")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Llama model not found at: {model_path}")
            return LlamaModel(model_path)

        raise ValueError("Invalid model type")
    except Exception as e:
        logger.error(f"Model creation failed: {str(e)}", exc_info=True)
        raise