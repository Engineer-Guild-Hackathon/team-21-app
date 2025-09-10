import { AUTH_STORAGE_KEYS } from '../config/constants';

interface RequestOptions extends RequestInit {
  requiresAuth?: boolean;
  isFormData?: boolean;
}

/**
 * APIリクエストを送信する共通関数
 * @param url リクエスト先のURL
 * @param options リクエストオプション
 * @returns レスポンス
 */
export async function fetchApi(url: string, options: RequestOptions = {}) {
  const {
    requiresAuth = false,
    isFormData = false,
    headers: customHeaders,
    ...otherOptions
  } = options;

  // デフォルトのヘッダーを設定
  const headers = new Headers(customHeaders);
  if (!headers.has('Content-Type')) {
    headers.set(
      'Content-Type',
      isFormData ? 'application/x-www-form-urlencoded' : 'application/json'
    );
  }

  // 認証が必要な場合、トークンを追加
  if (requiresAuth) {
    const token = localStorage.getItem(AUTH_STORAGE_KEYS.TOKEN);
    const tokenType = localStorage.getItem(AUTH_STORAGE_KEYS.TOKEN_TYPE);
    if (token && tokenType) {
      headers.set('Authorization', `${tokenType} ${token}`);
    }
  }

  // リクエストを送信
  try {
    console.log('Sending request to:', url, {
      method: otherOptions.method,
      headers: Object.fromEntries(headers.entries()),
      body: otherOptions.body,
    });

    const response = await fetch(url, {
      ...otherOptions,
      headers,
      mode: 'cors',
      credentials: 'same-origin',
    });

    console.log('Response:', {
      status: response.status,
      statusText: response.statusText,
      headers: Object.fromEntries(response.headers.entries()),
    });

    return response;
  } catch (error) {
    console.error('Network error:', error);
    throw error;
  }
}

/**
 * URLSearchParamsをJSON形式に変換する
 * @param params URLSearchParams
 * @returns JSON形式のデータ
 */
export function urlParamsToJson(params: URLSearchParams): Record<string, string> {
  const result: Record<string, string> = {};
  params.forEach((value, key) => {
    result[key] = value;
  });
  return result;
}
