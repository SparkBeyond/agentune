import json


def extract_json_from_response(response: str) -> dict:
    """Extract JSON from LLM response."""
    def _raise_no_json_error() -> None:
        raise ValueError('No JSON found in response')
    
    start = response.find('{')
    end = response.rfind('}') + 1
    if start != -1 and end > start:
        json_str = response[start:end]
    else:
        _raise_no_json_error()
    
    return json.loads(json_str)
