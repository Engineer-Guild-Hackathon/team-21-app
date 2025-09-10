'use client';

import { createContext, ReactNode, useContext, useEffect, useState } from 'react';

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
  login: (email: string, password: string) => Promise<User>;
  logout: () => void;
  register: (email: string, password: string, role: UserRole, name: string) => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    // ローカルストレージからユーザー情報を復元
    const storedUser = localStorage.getItem('user');
    if (storedUser) {
      setUser(JSON.parse(storedUser));
    }
  }, []);

  const login = async (email: string, password: string): Promise<User> => {
    try {
      // APIリクエストを実装
      const response = await fetch('http://localhost:8000/api/auth/token', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
          username: email,
          password: password,
        }),
      });

      if (!response.ok) {
        throw new Error('ログインに失敗しました');
      }

      const { access_token } = await response.json();

      // クッキーにアクセストークンを保存（middlewareが参照）
      // 注意: 本番ではHttpOnly/secure属性をサーバー側で設定すること
      const maxAge = 30 * 60; // 30分（サーバのトークン期限に整合）
      document.cookie = `token=${access_token}; Path=/; Max-Age=${maxAge}`;

      // ユーザー情報を取得
      const userResponse = await fetch('http://localhost:8000/api/users/me', {
        headers: {
          Authorization: `Bearer ${access_token}`,
        },
      });

      if (!userResponse.ok) {
        throw new Error('ユーザー情報の取得に失敗しました');
      }

      const userData = await userResponse.json();
      const loggedInUser: User = {
        id: userData.id.toString(),
        name: userData.full_name,
        role: (userData.role ?? 'student') as UserRole,
        email: userData.email,
        avatar: userData.avatar_url,
      };

      setUser(loggedInUser);
      localStorage.setItem('user', JSON.stringify(loggedInUser));
      return loggedInUser;
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    }
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('user');
    // クッキーの削除（Max-Age=0）
    document.cookie = 'token=; Path=/; Max-Age=0';
  };

  const register = async (email: string, password: string, role: UserRole, name: string) => {
    try {
      // APIリクエストを実装
      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password, role, name }),
      });

      if (!response.ok) {
        throw new Error('登録に失敗しました');
      }

      const data = await response.json();
      const newUser: User = {
        id: data.id,
        name: data.name,
        role: data.role,
        email: data.email,
      };

      setUser(newUser);
      localStorage.setItem('user', JSON.stringify(newUser));
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
