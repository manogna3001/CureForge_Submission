import json
import time
from datetime import datetime, timezone
import os

METRICS_PATH = ".cache/metrics/metrics.json"

def update_metrics(i):
    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "agent_created_total": [{"labels": {}, "value": 1}],
            "tool_calls_total": [{"labels": {"tool": "pubmed_search"}, "value": i}],
            "llm_calls_total": [{"labels": {}, "value": i * 2}],
            "phase_transitions_total": [{"labels": {}, "value": i // 5}]
        }
    }
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(data, f, indent=2)

print("Starting agent simulation...")
for i in range(1, 100):
    update_metrics(i)
    time.sleep(5)
