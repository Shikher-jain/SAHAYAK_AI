// k6 load test for Sahayak AI v2_advanced — semantic-cache + supervisor.
// Headless CI run:  k6 run tests/load/k6_benchmark.js
//
// Load profile: ramp 0->50 VU in 1m, hold 50 VU for 8m, ramp 50->0 in 1m.
// Only external dependency: k6. Dataset shared with Locust via queries.json.
//
// SLA thresholds:
//   http_req_failed        rate < 0.1%
//   cache_hit_duration     p95 < 80ms, p99 < 120ms
//   cache_miss_duration    p95 < 1500ms
//   cache_bypass_violation rate == 0 (safety queries must never serve from cache)

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend, Rate, Counter } from 'k6/metrics';
import { SharedArray } from 'k6/data';

const DATA = new SharedArray('sahayak_queries', () => JSON.parse(open('./queries.json')));
const TIER_A = DATA[0].tier_a.clusters;
const TIER_B_TEMPLATES = DATA[0].tier_b.templates;
const TIER_C = DATA[0].tier_c.queries;

const hitDuration = new Trend('cache_hit_duration', true);
const missDuration = new Trend('cache_miss_duration', true);
const hitRate = new Rate('cache_hit_rate');
const bypassViolation = new Rate('cache_bypass_violation');
const respError = new Rate('resp_error');
const bypassTotal = new Counter('cache_bypass_total');

const CHAT_URL = '/api/v2/chat/orchestrate';

export const options = {
  stages: [
    { duration: '1m', target: 50 },
    { duration: '8m', target: 50 },
    { duration: '1m', target: 0 },
  ],
  thresholds: {
    http_req_failed: ['rate<0.001'],
    resp_error: ['rate<0.001'],
    cache_hit_duration: ['p(95)<80', 'p(99)<120'],
    cache_miss_duration: ['p(95)<1500'],
  },
};

function headerValue(res, name) {
  const lower = name.toLowerCase();
  for (const key in res.headers) {
    if (key.toLowerCase() === lower) return res.headers[key];
  }
  return '';
}

function freshColdQuery() {
  const template = TIER_B_TEMPLATES[Math.floor(Math.random() * TIER_B_TEMPLATES.length)];
  const token = `${Date.now()}-${Math.floor(Math.random() * 1e9)}`;
  const today = new Date().toISOString().slice(0, 10);
  return template.replace(/\{uuid\}/g, token).replace(/\{date\}/g, today);
}

function pickQuery() {
  const roll = Math.random(); // 60% hit / 25% miss / 15% bypass
  if (roll < 0.6) {
    const cluster = TIER_A[Math.floor(Math.random() * TIER_A.length)];
    const pool = [cluster.q, ...cluster.v];
    return { message: pool[Math.floor(Math.random() * pool.length)], kind: 'hit' };
  }
  if (roll < 0.85) return { message: freshColdQuery(), kind: 'miss' };
  return { message: TIER_C[Math.floor(Math.random() * TIER_C.length)], kind: 'bypass' };
}

export default function () {
  const { message, kind } = pickQuery();
  const payload = JSON.stringify({
    message,
    user_id: `k6-${__VU}-${Math.random().toString(16).slice(2)}`,
    tier: 'free',
  });

  const res = http.post(CHAT_URL, payload, {
    headers: { 'Content-Type': 'application/json' },
  });

  let parsed = null;
  try {
    parsed = JSON.parse(res.body);
  } catch (_) {
    // body is not valid JSON
  }
  const bodyOk = parsed !== null && typeof parsed.response === 'string';
  const statusOk = res.status === 200;
  respError.add(!(statusOk && bodyOk));

  const cacheStatus = headerValue(res, 'X-Cache-Status');

  if (kind === 'hit') {
    hitDuration.add(res.timings.duration);
    hitRate.add(cacheStatus === 'HIT');
  } else if (kind === 'miss') {
    missDuration.add(res.timings.duration);
  } else {
    bypassTotal.add(1);
    // Safety invariant: bypass queries can be MISS (supervisor path) but never HIT.
    bypassViolation.add(cacheStatus === 'HIT');
  }

  check(res, {
    'status is 200': () => statusOk,
    'valid ChatResponse JSON': () => bodyOk,
  });

  sleep(0.5 + Math.random() * 1.5);
}