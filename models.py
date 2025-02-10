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
from typing import Optional

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
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content

class DeepseekLocalModel(BaseAIModel):
    def __init__(self, model_path: str):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=6,
            n_gpu_layers=33
        )

    def generate(self, prompt: str) -> str:
        response = self.llm(prompt, max_tokens=1000)
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
            self._init_db()
            logger.info("Database connection established")
        except Exception as e:
            logger.critical(f"Database connection failed: {str(e)}", exc_info=True)
            raise

    def _init_db(self):
            self.conn.execute('''CREATE TABLE IF NOT EXISTS memories
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                task TEXT,
                solution TEXT,
                success BOOLEAN,
                verification_result TEXT,
                reflections TEXT)''')

    def store_memory(self, memory: dict):
        try:
            self.conn.execute('''INSERT INTO memories
                (timestamp, task, solution, success, verification_result, reflections)
                VALUES (?, ?, ?, ?, ?, ?)''',
                (memory['timestamp'],
                memory['task'],
                memory['solution'],
                memory['success'],
                json.dumps(memory.get('verification_result', {})),
                json.dumps(memory.get('reflections', []))))
            self.conn.commit()
            logger.debug(f"Memory stored: {memory['task'][:20]}...")
        except Exception as e:
            logger.error(f"Memory storage failed: {str(e)}", exc_info=True)
            raise

    def get_memories(self, limit: int = 10):
        cursor = self.conn.execute('''SELECT * FROM memories ORDER BY id DESC LIMIT ?''', (limit,))
        return [dict(row) for row in cursor.fetchall()]

class SoftwareEngineerAgent:
    def __init__(self, model: BaseAIModel):
        self.model = model
        self.memory_manager = MemoryManager()
        self.reflection_interval = 3
        logger.info("Agent initialized")

    def process_task(self, problem: str) -> dict:
        memory_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "task": problem,
            "success": False,
            "solution": "",
            "verification_result": {},
            "reflections": []
        }

        try:
            logger.debug(f"Starting task processing: {problem[:30]}...")
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

        Provide only the code solution without explanations, enclosed in triple backticks."""
        return self.model.generate(prompt)

    def verify_solution(self, problem: str, solution: str) -> dict:
        test_prompt = f"""Given this problem:
        {problem}
        And this solution:
        {solution}
        Generate 3 test cases with inputs and expected outputs in JSON format."""

        try:
            test_cases = json.loads(self.model.generate(test_prompt))
            namespace = {}
            exec(solution, namespace)

            results = {"success": True, "tests": []}
            for test in test_cases:
                try:
                    func_args = test['inputs']
                    expected = test['expected']
                    result = namespace[list(namespace.keys())[-1]](*func_args)
                    results["tests"].append({
                        "input": func_args,
                        "expected": expected,
                        "received": result,
                        "success": result == expected
                    })
                    if result != expected:
                        results["success"] = False
                except Exception as e:
                    results["tests"].append({
                        "error": str(e),
                        "success": False
                    })
                    results["success"] = False
            return results
        except Exception as e:
            return {"success": False, "error": str(e)}

    def reflect(self) -> list:
        reflections = []
        memories = self.memory_manager.get_memories(self.reflection_interval)

        success_rate = sum(1 for m in memories if m['success']) / len(memories)
        reflections.append(f"Recent success rate: {success_rate*100:.1f}%")

        common_errors = {}
        for memory in memories:
            if not memory['success']:
                error = memory.get('verification_result', {}).get('error', 'Unknown error')
                common_errors[error] = common_errors.get(error, 0) + 1

        if common_errors:
            reflections.append("Most common errors:")
            for error, count in sorted(common_errors.items(), key=lambda x: x[1], reverse=True)[:3]:
                reflections.append(f"- {error} ({count} occurrences)")

        return reflections

def create_model(model_type: str) -> BaseAIModel:
    try:
        logger.info(f"Creating model: {model_type}")
        if model_type == "OpenAI":
            return OpenAIModel(os.getenv('OPENAI_API_KEY'))
        elif model_type == "Deepseek":
            return DeepseekLocalModel(os.getenv('DEEPSEEK_MODEL_PATH'))
        elif model_type == "Claude":
            return ClaudeModel(os.getenv('CLAUDE_API_KEY'))
        elif model_type == "Llama":
            return LlamaModel(os.getenv('LLAMA_MODEL_PATH'))
        raise ValueError("Invalid model type")
    except Exception as e:
        logger.error(f"Model creation failed: {str(e)}", exc_info=True)
        raise