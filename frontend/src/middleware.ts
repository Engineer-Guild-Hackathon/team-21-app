import type { NextRequest } from 'next/server';
import { NextResponse } from 'next/server';

// 認証が必要なパス
const protectedPaths = [
  '/learning',
  '/progress',
  '/feedback',
  '/dashboard',
  '/analysis',
  '/class',
  '/records',
];

// 認証不要のパス
const publicPaths = ['/auth/login', '/auth/register'];

export function middleware(request: NextRequest) {
  const token = request.cookies.get('token');
  const { pathname } = request.nextUrl;

  // 認証が必要なパスへのアクセスをチェック
  if (protectedPaths.some(path => pathname.startsWith(path))) {
    if (!token) {
      const loginUrl = new URL('/auth/login', request.url);
      loginUrl.searchParams.set('redirect', pathname);
      return NextResponse.redirect(loginUrl);
    }
  }

  // ログイン済みユーザーが認証ページにアクセスした場合は適切なページへ案内
  if (publicPaths.includes(pathname) && token) {
    // デフォルトで学習ページにリダイレクト（ロール別分岐はクライアント側で処理）
    return NextResponse.redirect(new URL('/learning', request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - api (API routes)
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     */
    '/((?!api|_next/static|_next/image|favicon.ico).*)',
  ],
};
