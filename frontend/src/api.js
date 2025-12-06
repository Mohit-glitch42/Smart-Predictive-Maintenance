import axios from "axios";

const API_BASE = "http://localhost:8080/api";

export async function predictFailure(payload) {
  // payload is the JetEngineRequest-shaped object from the form
  const response = await axios.post(`${API_BASE}/predict`, payload);
  return response.data;
}
