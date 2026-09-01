// Standardized API client for Sahayak AI

export const DEFAULT_BACKEND_URL = (import.meta.env && import.meta.env.VITE_BACKEND_URL) || "https://sahayak-ai-yxl8.onrender.com";

export const getBackendUrl = () => {
  return localStorage.getItem("backend_url") || DEFAULT_BACKEND_URL;
};

export const setBackendUrl = (url) => {
  if (url) {
    localStorage.setItem("backend_url", url.replace(/\/+$/, ""));
  } else {
    localStorage.removeItem("backend_url");
  }
};

/**
 * Universal backend call wrapper with error handling, authentication, and headers
 */
export const callBackend = async (method, path, data = null, customHeaders = {}) => {
  const token = localStorage.getItem("auth_token");
  const apiKey = localStorage.getItem("api_key");
  
  const headers = { 
    ...(!data || !(data instanceof FormData) ? { "Content-Type": "application/json" } : {}),
    ...(token ? { "Authorization": `Bearer ${token}` } : {}),
    ...(apiKey ? { "X-API-Key": apiKey } : {}),
    ...customHeaders
  };

  const config = { 
    method: method.toUpperCase(), 
    headers 
  };

  if (data) {
    config.body = data instanceof FormData ? data : JSON.stringify(data);
  }

  const baseUrl = getBackendUrl();
  const cleanPath = path.startsWith("/") ? path : `/${path}`;

  try {
    const response = await fetch(`${baseUrl}${cleanPath}`, config);
    
    // Handle 204 No Content
    if (response.status === 204) {
      return { ok: true, data: null, status: 204, error: null };
    }

    let payload;
    const contentType = response.headers.get("content-type");
    if (contentType && contentType.includes("application/json")) {
      payload = await response.json().catch(() => null);
    } else {
      payload = await response.text().catch(() => null);
    }

    if (!response.ok) {
      let errorMessage = `HTTP ${response.status}: Request failed`;
      if (payload) {
        if (typeof payload === 'object') {
          errorMessage = payload.detail || payload.message || payload.error || JSON.stringify(payload);
        } else if (typeof payload === 'string' && payload.trim()) {
          errorMessage = payload;
        }
      }
      return { 
        ok: false, 
        data: payload, 
        status: response.status, 
        error: errorMessage 
      };
    }

    return { ok: true, data: payload, status: response.status, error: null };
  } catch (error) {
    const isNetworkError = error.message?.includes("Failed to fetch") || error.name === "TypeError";
    const userError = isNetworkError 
      ? `Cannot connect to server at ${baseUrl}. Ensure backend is running.`
      : error.message || "Network request failed";
    
    return { 
      ok: false, 
      data: null, 
      status: 0, 
      error: userError 
    };
  }
};
