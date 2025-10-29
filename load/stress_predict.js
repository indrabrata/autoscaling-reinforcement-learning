import http from "k6/http";
import { check, sleep } from "k6";

export let options = {
  discardResponseBodies: true,

  scenarios: {
    stress_predict: {
      executor: "ramping-arrival-rate",
      startRate: 50,           
      timeUnit: "1s",
      preAllocatedVUs: 500,    
      maxVUs: 5000,             
      stages: [
        { target: 500, duration: "30s" },    
        { target: 1500, duration: "30s" },
        { target: 3000, duration: "30s" },
        { target: 4000, duration: "30s" },
        { target: 5000, duration: "30s" }, 
      ],
      exec: "predictLoad",
    },
  },
};

const BASE_URL = __ENV.BASE_URL || "http://localhost:8080/api";
const imageBin = open("./shoes.jpg", "b");

export function predictLoad() {
  const formData = {
    file: http.file(imageBin, "shoes.jpg", "image/jpeg"),
  };

  const res = http.post(`${BASE_URL}/predict?topk=1`, formData);

  check(res, {
    "status is 200": (r) => r.status === 200,
  });

  sleep(0.1);
}
