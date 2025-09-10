export class AuthError extends Error {
  constructor(
    message: string,
    public readonly code: AuthErrorCode,
    public readonly originalError?: unknown
  ) {
    super(message);
    this.name = 'AuthError';
  }
}

export enum AuthErrorCode {
  INVALID_CREDENTIALS = 'INVALID_CREDENTIALS',
  NETWORK_ERROR = 'NETWORK_ERROR',
  SERVER_ERROR = 'SERVER_ERROR',
  USER_NOT_FOUND = 'USER_NOT_FOUND',
  VALIDATION_ERROR = 'VALIDATION_ERROR',
  TOKEN_EXPIRED = 'TOKEN_EXPIRED',
  UNKNOWN = 'UNKNOWN',
}

export function getAuthErrorMessage(error: AuthError): string {
  switch (error.code) {
    case AuthErrorCode.INVALID_CREDENTIALS:
      return 'メールアドレスまたはパスワードが正しくありません';
    case AuthErrorCode.NETWORK_ERROR:
      return 'ネットワークエラーが発生しました。インターネット接続を確認してください';
    case AuthErrorCode.SERVER_ERROR:
      return 'サーバーエラーが発生しました。しばらく時間をおいて再度お試しください';
    case AuthErrorCode.USER_NOT_FOUND:
      return 'ユーザーが見つかりません';
    case AuthErrorCode.VALIDATION_ERROR:
      return '入力内容に誤りがあります';
    case AuthErrorCode.TOKEN_EXPIRED:
      return 'セッションの有効期限が切れました。再度ログインしてください';
    default:
      return 'エラーが発生しました。しばらく時間をおいて再度お試しください';
  }
}
