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
      // デモ用のログイン機能
      const demoUsers = [
        {
          id: '1',
          name: '山田太郎',
          role: 'student' as UserRole,
          email: 'taro@example.com',
          password: 'demo1234',
        },
        {
          id: '2',
          name: '山田花子',
          role: 'parent' as UserRole,
          email: 'hanako@example.com',
          password: 'demo1234',
        },
        {
          id: '3',
          name: '佐藤先生',
          role: 'teacher' as UserRole,
          email: 'sato@example.com',
          password: 'demo1234',
        },
      ];

      // デモユーザーを検索
      const demoUser = demoUsers.find(user => user.email === email && user.password === password);

      if (!demoUser) {
        throw new AuthError(
          'メールアドレスまたはパスワードが正しくありません',
          AuthErrorCode.INVALID_CREDENTIALS
        );
      }

      // デモ用のトークンを生成
      const access_token = `demo_token_${Date.now()}`;
      const token_type = 'bearer';

      // トークンをローカルストレージに保存
      localStorage.setItem(AUTH_STORAGE_KEYS.TOKEN, access_token);
      localStorage.setItem(AUTH_STORAGE_KEYS.TOKEN_TYPE, token_type);
      localStorage.setItem(AUTH_STORAGE_KEYS.TOKEN_TIMESTAMP, Date.now().toString());
      localStorage.setItem(AUTH_STORAGE_KEYS.REMEMBER_ME, rememberMe.toString());

      // デモユーザー情報を設定
      const loggedInUser: User = {
        id: demoUser.id,
        name: demoUser.name,
        role: demoUser.role,
        email: demoUser.email,
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
      // デモ用の新規登録機能
      // 簡単なバリデーション
      if (!email || !password || !name) {
        throw new AuthError('すべての項目を入力してください', AuthErrorCode.VALIDATION_ERROR);
      }

      if (password.length < 6) {
        throw new AuthError(
          'パスワードは6文字以上で入力してください',
          AuthErrorCode.VALIDATION_ERROR
        );
      }

      // デモ用のユーザーIDを生成
      const newUserId = `demo_${Date.now()}`;

      // デモ用のトークンを生成
      const access_token = `demo_token_${Date.now()}`;
      const token_type = 'bearer';

      // トークンをローカルストレージに保存
      localStorage.setItem(AUTH_STORAGE_KEYS.TOKEN, access_token);
      localStorage.setItem(AUTH_STORAGE_KEYS.TOKEN_TYPE, token_type);
      localStorage.setItem(AUTH_STORAGE_KEYS.TOKEN_TIMESTAMP, Date.now().toString());
      localStorage.setItem(AUTH_STORAGE_KEYS.REMEMBER_ME, 'true');

      // デモユーザー情報を設定
      const newUser: User = {
        id: newUserId,
        name: name,
        role: role,
        email: email,
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
