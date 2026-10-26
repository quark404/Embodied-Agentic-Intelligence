"""OpenAI-compatible text action endpoint for locally served open models.

This adapter is deliberately separate from the simulator and does not imply that
a specific Qwen checkpoint or VLM server has been validated.
"""

import json
import os
from urllib.request import Request, urlopen


class ChatActionPolicy:
    def __init__(self, endpoint: str, model: str, *, api_key_env: str = "MODEL_API_KEY",
                 transport=None):
        self.endpoint = endpoint.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key_env = api_key_env
        self.transport = transport or self._post

    def _post(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = Request(self.endpoint, data=json.dumps(payload).encode(), headers=headers)
        with urlopen(request, timeout=60) as response:
            return json.load(response)

    def act(self, observation: dict, history: list[dict]) -> dict:
        prompt = {
            "observation": observation,
            "recent_history": history[-4:],
            "actions": ["lookup_code", "unlock", "grasp", "place"],
        }
        response = self.transport({
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "Return one JSON object with type and required fields. "
                 "Use only listed actions. Do not infer an unlock code absent from observations or receipts."},
                {"role": "user", "content": json.dumps(prompt)},
            ],
        })
        action = json.loads(response["choices"][0]["message"]["content"])
        if not isinstance(action, dict) or action.get("type") not in prompt["actions"]:
            raise ValueError("Model returned an invalid action")
        required = {"unlock": "code", "grasp": "object", "place": "destination"}
        field = required.get(action["type"])
        if field and not isinstance(action.get(field), str):
            raise ValueError(f"Model action requires string field {field}")
        return action
