import axios from "axios";

const API_BASE = "http://localhost:8080/api";

export async function predictFailure(payload) {
  const response = await axios.post(`${API_BASE}/predict`, payload);
  return response.data;
}
