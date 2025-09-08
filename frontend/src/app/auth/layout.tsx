'use client';

import { useRouter } from 'next/navigation';
import { useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';

export default function AuthLayout({ children }: { children: React.ReactNode }) {
  const { user } = useAuth();
  const router = useRouter();

  useEffect(() => {
    // すでにログインしている場合はホームページにリダイレクト
    if (user) {
      router.push('/');
    }
  }, [user, router]);

  return <>{children}</>;
}
