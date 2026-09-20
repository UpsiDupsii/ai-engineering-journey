# services/evaluation.py
import json
import requests
from typing import Dict, Any
from core.config import settings

def evaluate_rag_output(query: str, context_text: str, generated_answer: str) -> Dict[str, Any]:
    """
    Topic 24: LLM-as-a-Judge Evaluation (Ragas concept).
    Evaluates Faithfulness and Answer Relevancy on a scale of 0-10.
    """
    prompt = f"""You are an objective AI evaluator grading a Retrieval-Augmented Generation (RAG) system. 
Please evaluate the generated answer based on the user's question and the retrieved context.

You must score the system on two metrics from 0 to 10:
1. Faithfulness (0-10): Is the generated answer entirely based on the provided context? (Score 0 if it hallucinates outside info, 10 if completely faithful).
2. Relevancy (0-10): Does the generated answer directly and effectively answer the user's question? (Score 0 if completely off-topic, 10 if perfectly relevant).

Provide a brief feedback sentence explaining the scores.

Output your evaluation STRICTLY as a JSON object with the following keys: "faithfulness", "relevancy", "feedback".
Do not include markdown blocks, just the raw JSON.

---
USER QUESTION: {query}
---
RETRIEVED CONTEXT: 
{context_text}
---
GENERATED ANSWER: 
{generated_answer}
---
JSON EVALUATION:"""

    url = f"{settings.OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result_str = response.json().get("response", "{}")
        eval_dict = json.loads(result_str)
        
        return {
            "faithfulness": int(eval_dict.get("faithfulness", 0)),
            "relevancy": int(eval_dict.get("relevancy", 0)),
            "feedback": str(eval_dict.get("feedback", "No feedback provided."))
        }
    except Exception as e:
        return {
            "faithfulness": 0,
            "relevancy": 0,
            "feedback": f"Evaluation failed due to LLM error: {str(e)}"
        }