import http from "k6/http";
import { check, sleep } from "k6";

export let options = {
  discardResponseBodies: true,

  scenarios: {
    stress_orders: {
      executor: "ramping-arrival-rate",
      startRate: 50,
      timeUnit: "1s",
      preAllocatedVUs: 500, 
      maxVUs: 7000,          
      stages: [
        { target: 500, duration: "30s" },    
        { target: 1500, duration: "30s" },
        { target: 3000, duration: "30s" },
        { target: 5000, duration: "30s" },
        { target: 7000, duration: "30s" },   
      ],
      exec: "ordersLoad",
    },
  },
};

const BASE_URL = __ENV.BASE_URL || "http://localhost:8080/api";

export function ordersLoad() {
  const res = http.get(`${BASE_URL}/orders`);
  check(res, { "status is 200": (r) => r.status === 200 });
  sleep(0.1);
}
