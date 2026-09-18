"""Locust load test for Sahayak AI v2_advanced — semantic-cache + supervisor.

Targets:  POST /api/v2/chat/orchestrate  (ChatResponse JSON, X-Cache-Status header)
SLA:      HIT  p95 < 80ms, p99 < 120ms
          MISS p95 < 1500ms
          errors < 0.1%, zero OOM/restarts over 10 min

Run (headless):
  locust -f tests/load/locustfile.py --headless -u 50 -r 5 --run-time 10m \
         --host http://localhost:8000 --html tests/load/locust_report.html
Run (web UI):
  locust -f tests/load/locustfile.py --host http://localhost:8000

Only external dependency: `locust`. Dataset lives in tests/load/queries.json.
"""
import datetime
import json
import random
import uuid
from pathlib import Path

from locust import HttpUser, between, task

_DATA = json.loads((Path(__file__).parent / "queries.json").read_text(encoding="utf-8"))

TIER_A = _DATA["tier_a"]["clusters"]
TIER_B_TEMPLATES = _DATA["tier_b"]["templates"]
TIER_C = _DATA["tier_c"]["queries"]

CHAT_URL = "/api/v2/chat/orchestrate"

# Tier A clusters are worded to avoid the cache's crisis/personal bypass tokens
# so a paraphrase of a seeded canonical is eligible for a HIT.
CANONICAL_VARIATIONS = [c["q"] for c in TIER_A] + [v for c in TIER_A for v in c["v"]]


def fresh_cold_query() -> str:
    template = random.choice(TIER_B_TEMPLATES)
    return (
        template
        .replace("{uuid}", str(uuid.uuid4()))
        .replace("{date}", datetime.date.today().isoformat())
    )


class SahayakUser(HttpUser):
    """50 VUs, spawn rate 5/s, think time 0.5-2.0s. 60/25/15 hit/miss/bypass mix."""

    wait_time = between(0.5, 2.0)

    def _post(self, message: str, expect: str):
        payload = {"message": message, "user_id": f"load_{uuid.uuid4().hex[:12]}", "tier": "free"}
        with self.client.post(
            CHAT_URL, json=payload, headers={"Content-Type": "application/json"},
            name=CHAT_URL, catch_response=True,
        ) as resp:
            header_status = (resp.headers.get("X-Cache-Status") or "MISS").upper()

            # Tag the request so the Web UI / CSV report splits latencies per
            # cache outcome: /chat [HIT] | /chat [MISS] | /chat [BYPASS]
            marker = "BYPASS" if expect == "bypass" else header_status
            resp.request_meta["name"] = f"{CHAT_URL} [{marker}]"

            if resp.status_code != 200:
                resp.failure(f"non-200 status: {resp.status_code}")
                return
            try:
                body = resp.json()
            except ValueError as exc:
                resp.failure(f"invalid JSON body: {exc}")
                return
            for key in ("routed_agent", "response", "cached"):
                if key not in body:
                    resp.failure(f"missing {key!r} in ChatResponse")
                    return
            # Safety invariant: a bypass query must never be served from cache.
            if expect == "bypass" and header_status == "HIT":
                resp.failure("CRITICAL: safety/bypass query returned X-Cache-Status=HIT")
            # Seeding expects: first Tier-A request is naturally a MISS, later ones
            # must be HITs. Non-HIT here is not a hard failure (N+1 seeding a
            # fresh cluster), it just lowers the HIT-rate metric.

    @task(60)
    def cache_hit_cluster(self):
        cluster = random.choice(TIER_A)
        query = random.choice([cluster["q"]] + cluster["v"])
        self._post(query, expect="hit")

    @task(25)
    def cache_miss_unique(self):
        self._post(fresh_cold_query(), expect="miss")

    @task(15)
    def cache_bypass_safety(self):
        self._post(random.choice(TIER_C), expect="bypass")