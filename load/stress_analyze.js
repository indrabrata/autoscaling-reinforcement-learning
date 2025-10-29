import http from "k6/http";
import { check, sleep } from "k6";

export let options = {
  discardResponseBodies: true,

  scenarios: {
    stress_analyze: {
      executor: "ramping-arrival-rate",
      startRate: 50,
      timeUnit: "1s",
      preAllocatedVUs: 500,  
      maxVUs: 10000,          
      stages: [
        { target: 500, duration: "30s" },
        { target: 1500, duration: "30s" },
        { target: 3000, duration: "30s" },
        { target: 5000, duration: "30s" },
        { target: 7000, duration: "30s" },
        { target: 10000, duration: "30s" },
      ],
      exec: "analyzeLoad",
    },
  },
};

const BASE_URL = __ENV.BASE_URL || "http://localhost:8080/api";
const transactionsData = JSON.parse(open("./seed_analyze.json"));

export function analyzeLoad() {
  const payload = JSON.stringify(transactionsData);
  const res = http.post(`${BASE_URL}/analyze`, payload, {
    headers: { "Content-Type": "application/json" },
  });

  check(res, {
    "status is 200": (r) => r.status === 200,
  });

  sleep(0.1);
}
