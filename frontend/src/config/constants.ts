export const API_CONFIG = {
  BASE_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  VERSION: process.env.NEXT_PUBLIC_API_VERSION || 'v1',
  TOKEN_EXPIRE_HOURS: parseInt(process.env.NEXT_PUBLIC_TOKEN_EXPIRE_HOURS || '24', 10),
} as const;

export const AUTH_ENDPOINTS = {
  TOKEN: `${API_CONFIG.BASE_URL}/api/v1/token`,
  REFRESH: `${API_CONFIG.BASE_URL}/api/v1/token/refresh`,
  ME: `${API_CONFIG.BASE_URL}/api/v1/me`,
  REGISTER: `${API_CONFIG.BASE_URL}/api/v1/register`,
} as const;

export const AUTH_STORAGE_KEYS = {
  TOKEN: 'auth_token',
  REFRESH_TOKEN: 'refresh_token',
  TOKEN_TYPE: 'token_type',
  TOKEN_TIMESTAMP: 'token_timestamp',
  REMEMBER_ME: 'remember_me',
  USER: 'user',
} as const;
