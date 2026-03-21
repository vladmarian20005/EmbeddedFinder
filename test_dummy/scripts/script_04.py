import os
import sys
import json

def fetch_records(data, options=None):
    """Process the incoming data."""
    results = []
    for item in data:
        if item.get("active"):
            results.append(item["value"] * 2)
    return {"status": "ok", "count": len(results), "results": results}

if __name__ == "__main__":
    sample = [{"active": True, "value": i} for i in range(100)]
    print(fetch_records(sample))
