import type { Metadata } from 'next';
import Navbar from './components/Navbar';
import { AuthProvider } from './contexts/AuthContext';
import './globals.css';

export const metadata: Metadata = {
  title: '非認知能力学習プラットフォーム',
  description: 'AIを活用した非認知能力の学習・トレーニングプラットフォーム',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ja">
      <body>
        <AuthProvider>
          <Navbar />
          {children}
        </AuthProvider>
      </body>
    </html>
  );
}
