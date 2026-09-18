import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';

const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api/v2',
});

// Auto-inject Correlation ID
apiClient.interceptors.request.use((config) => {
  config.headers['X-Correlation-ID'] = uuidv4();
  return config;
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 429) {
      console.warn(`Rate limited. Retry after: ${error.response.headers['retry-after']}s`);
    }
    return Promise.reject(error);
  }
);

export default apiClient;
