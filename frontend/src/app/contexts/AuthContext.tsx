'use client';

import { createContext, ReactNode, useContext, useEffect, useState } from 'react';
import { API_CONFIG, AUTH_ENDPOINTS, AUTH_STORAGE_KEYS } from '../../config/constants';
import { AuthError, AuthErrorCode } from '../../types/error';
import { fetchApi } from '../../utils/api';

export type UserRole = 'student' | 'parent' | 'teacher';

export interface User {
  id: string;
  name: string;
  role: UserRole;
  email: string;
  avatar?: string;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  login: (email: string, password: string, rememberMe?: boolean) => Promise<void>;
  logout: () => void;
  register: (email: string, password: string, role: UserRole, name: string) => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);

  // トークンのリフレッシュを試みる
  const refreshToken = async () => {
    const refreshToken = localStorage.getItem(AUTH_STORAGE_KEYS.REFRESH_TOKEN);
    if (!refreshToken) {
      return false;
    }

    try {
      const response = await fetchApi(AUTH_ENDPOINTS.REFRESH, {
        method: 'POST',
        body: JSON.stringify({ refresh_token: refreshToken }),
      });

      if (!response.ok) {
        return false;
      }

      const { access_token, token_type } = await response.json();
      localStorage.setItem(AUTH_STORAGE_KEYS.TOKEN, access_token);
      localStorage.setItem(AUTH_STORAGE_KEYS.TOKEN_TYPE, token_type);
      localStorage.setItem(AUTH_STORAGE_KEYS.TOKEN_TIMESTAMP, Date.now().toString());

      return true;
    } catch {
      return false;
    }
  };

  useEffect(() => {
    // ローカルストレージからユーザー情報とトークンを復元
    const storedUser = localStorage.getItem(AUTH_STORAGE_KEYS.USER);
    const storedToken = localStorage.getItem(AUTH_STORAGE_KEYS.TOKEN);
    const tokenTimestamp = localStorage.getItem(AUTH_STORAGE_KEYS.TOKEN_TIMESTAMP);
    const tokenType = localStorage.getItem(AUTH_STORAGE_KEYS.TOKEN_TYPE);
    const rememberMe = localStorage.getItem(AUTH_STORAGE_KEYS.REMEMBER_ME) === 'true';

    if (storedUser && storedToken && tokenTimestamp && tokenType) {
      // トークンの有効期限をチェック
      const tokenAge = Date.now() - parseInt(tokenTimestamp);
      if (tokenAge < API_CONFIG.TOKEN_EXPIRE_HOURS * 60 * 60 * 1000) {
        setUser(JSON.parse(storedUser));
      } else if (rememberMe) {
        // トークンが期限切れかつ「ログイン状態を保持」が有効な場合、リフレッシュを試みる
        refreshToken().then(success => {
          if (!success) {
            logout();
          } else {
            setUser(JSON.parse(storedUser));
          }
        });
      } else {
        logout();
      }
    }
  }, []);

  const login = async (email: string, password: string, rememberMe: boolean = false) => {
    try {
      // APIリクエストを実装
      const params = new URLSearchParams({
        username: email,
        password: password,
        grant_type: 'password',
      });

      const response = await fetch(AUTH_ENDPOINTS.TOKEN, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          Accept: 'application/json',
        },
        body: params,
        mode: 'cors',
        credentials: 'include',
      }).catch(error => {
        throw new AuthError('ネットワークエラーが発生しました', AuthErrorCode.NETWORK_ERROR, error);
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const status = response.status;

        if (status === 401) {
          throw new AuthError(
            'メールアドレスまたはパスワードが正しくありません',
            AuthErrorCode.INVALID_CREDENTIALS,
            errorData
          );
        } else if (status === 422) {
          throw new AuthError(
            '入力内容に誤りがあります',
            AuthErrorCode.VALIDATION_ERROR,
            errorData
          );
        } else if (status >= 500) {
          throw new AuthError(
            'サーバーエラーが発生しました',
            AuthErrorCode.SERVER_ERROR,
            errorData
          );
        } else {
          throw new AuthError(
            errorData.detail || 'ログインに失敗しました',
            AuthErrorCode.UNKNOWN,
            errorData
          );
        }
      }

      const { access_token, refresh_token, token_type } = await response.json().catch(error => {
        throw new AuthError('レスポンスの解析に失敗しました', AuthErrorCode.SERVER_ERROR, error);
      });

      // トークンをローカルストレージに保存
      localStorage.setItem(AUTH_STORAGE_KEYS.TOKEN, access_token);
      localStorage.setItem(AUTH_STORAGE_KEYS.TOKEN_TYPE, token_type);
      localStorage.setItem(AUTH_STORAGE_KEYS.TOKEN_TIMESTAMP, Date.now().toString());
      localStorage.setItem(AUTH_STORAGE_KEYS.REMEMBER_ME, rememberMe.toString());

      // リフレッシュトークンは「ログイン状態を保持」が有効な場合のみ保存
      if (rememberMe && refresh_token) {
        localStorage.setItem(AUTH_STORAGE_KEYS.REFRESH_TOKEN, refresh_token);
      }

      // ユーザー情報を取得
      const userResponse = await fetchApi(AUTH_ENDPOINTS.ME, {
        requiresAuth: true,
      }).catch(error => {
        throw new AuthError('ネットワークエラーが発生しました', AuthErrorCode.NETWORK_ERROR, error);
      });

      if (!userResponse.ok) {
        const errorData = await userResponse.json().catch(() => ({}));
        const status = userResponse.status;

        if (status === 401) {
          throw new AuthError(
            'セッションの有効期限が切れました',
            AuthErrorCode.TOKEN_EXPIRED,
            errorData
          );
        } else if (status === 404) {
          throw new AuthError('ユーザーが見つかりません', AuthErrorCode.USER_NOT_FOUND, errorData);
        } else if (status >= 500) {
          throw new AuthError(
            'サーバーエラーが発生しました',
            AuthErrorCode.SERVER_ERROR,
            errorData
          );
        } else {
          throw new AuthError(
            errorData.detail || 'ユーザー情報の取得に失敗しました',
            AuthErrorCode.UNKNOWN,
            errorData
          );
        }
      }

      const userData = await userResponse.json().catch(error => {
        throw new AuthError('レスポンスの解析に失敗しました', AuthErrorCode.SERVER_ERROR, error);
      });
      const loggedInUser: User = {
        id: userData.id.toString(),
        name: userData.full_name,
        role: userData.role as UserRole,
        email: userData.email,
        avatar: userData.avatar_url,
      };

      setUser(loggedInUser);
      localStorage.setItem(AUTH_STORAGE_KEYS.USER, JSON.stringify(loggedInUser));
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    }
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem(AUTH_STORAGE_KEYS.USER);
    localStorage.removeItem(AUTH_STORAGE_KEYS.TOKEN);
    localStorage.removeItem(AUTH_STORAGE_KEYS.REFRESH_TOKEN);
    localStorage.removeItem(AUTH_STORAGE_KEYS.TOKEN_TYPE);
    localStorage.removeItem(AUTH_STORAGE_KEYS.TOKEN_TIMESTAMP);
    localStorage.removeItem(AUTH_STORAGE_KEYS.REMEMBER_ME);
  };

  const register = async (email: string, password: string, role: UserRole, name: string) => {
    try {
      // APIリクエストを実装
      const response = await fetchApi(AUTH_ENDPOINTS.REGISTER, {
        method: 'POST',
        body: JSON.stringify({ email, password, role, name }),
      }).catch(error => {
        throw new AuthError('ネットワークエラーが発生しました', AuthErrorCode.NETWORK_ERROR, error);
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const status = response.status;

        if (status === 409) {
          throw new AuthError(
            'このメールアドレスは既に登録されています',
            AuthErrorCode.VALIDATION_ERROR,
            errorData
          );
        } else if (status === 422) {
          throw new AuthError(
            '入力内容に誤りがあります',
            AuthErrorCode.VALIDATION_ERROR,
            errorData
          );
        } else if (status >= 500) {
          throw new AuthError(
            'サーバーエラーが発生しました',
            AuthErrorCode.SERVER_ERROR,
            errorData
          );
        } else {
          throw new AuthError(
            errorData.detail || '登録に失敗しました',
            AuthErrorCode.UNKNOWN,
            errorData
          );
        }
      }

      const data = await response.json().catch(error => {
        throw new AuthError('レスポンスの解析に失敗しました', AuthErrorCode.SERVER_ERROR, error);
      });
      const newUser: User = {
        id: data.id,
        name: data.name,
        role: data.role,
        email: data.email,
      };

      setUser(newUser);
      localStorage.setItem(AUTH_STORAGE_KEYS.USER, JSON.stringify(newUser));
    } catch (error) {
      console.error('Registration error:', error);
      throw error;
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user,
        login,
        logout,
        register,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
