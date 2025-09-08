from typing import Optional, Dict, Any

class MLBaseException(Exception):
    """MLサービスの基本例外クラス"""
    def __init__(
        self,
        message: str,
        error_code: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}

class ModelNotReadyError(MLBaseException):
    """モデルが準備できていない場合の例外"""
    def __init__(
        self,
        model_name: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=f"モデル '{model_name}' の準備ができていません",
            error_code="MODEL_NOT_READY",
            details=details
        )

class InvalidInputError(MLBaseException):
    """入力データが無効な場合の例外"""
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="INVALID_INPUT",
            details=details
        )

class PredictionError(MLBaseException):
    """予測処理中のエラー"""
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="PREDICTION_ERROR",
            details=details
        )

class ResourceExhaustedError(MLBaseException):
    """リソースが不足している場合のエラー"""
    def __init__(
        self,
        resource_type: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=f"リソース '{resource_type}' が不足しています",
            error_code="RESOURCE_EXHAUSTED",
            details=details
        )

class DataQualityError(MLBaseException):
    """データ品質に問題がある場合のエラー"""
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="DATA_QUALITY_ERROR",
            details=details
        )

def handle_ml_error(error: Exception) -> Dict[str, Any]:
    """MLエラーをハンドリングしてAPI応答用の形式に変換"""
    if isinstance(error, MLBaseException):
        return {
            "error": {
                "code": error.error_code,
                "message": str(error),
                "details": error.details
            }
        }
    
    # 未知のエラーの場合
    return {
        "error": {
            "code": "UNKNOWN_ERROR",
            "message": "予期せぬエラーが発生しました",
            "details": {
                "original_error": str(error)
            }
        }
    }
