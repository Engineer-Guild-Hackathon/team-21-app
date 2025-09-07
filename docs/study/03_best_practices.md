# 開発ベストプラクティス

## コーディング規約

### 1. Python

```python
# Good
from typing import List, Optional
from datetime import datetime

class UserService:
    """ユーザー関連のビジネスロジックを提供するサービス"""

    def __init__(self, repository: UserRepository):
        self.repository = repository

    async def get_active_users(self) -> List[User]:
        """アクティブなユーザーの一覧を取得する"""
        return await self.repository.find_by_status(UserStatus.ACTIVE)

# Bad
class userService:
    def __init__(self, repo):
        self.repo = repo

    def getActiveUsers(self):
        return self.repo.find_by_status("active")
```

### 2. TypeScript

```typescript
// Good
interface UserProps {
  id: string;
  name: string;
  email: string;
  createdAt: Date;
}

const UserProfile: React.FC<UserProps> = ({ id, name, email, createdAt }) => {
  return (
    <div>
      <h2>{name}</h2>
      <p>{email}</p>
    </div>
  );
};

// Bad
const UserProfile = (props) => {
  return (
    <div>
      <h2>{props.name}</h2>
      <p>{props.email}</p>
    </div>
  );
};
```

## エラー処理

### 1. 例外処理

```python
# Good
from fastapi import HTTPException

class UserNotFoundError(Exception):
    pass

async def get_user(user_id: str) -> User:
    try:
        user = await repository.find_by_id(user_id)
        if not user:
            raise UserNotFoundError(f"User {user_id} not found")
        return user
    except DatabaseError as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500)
    except UserNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

# Bad
async def get_user(user_id):
    user = await repository.find_by_id(user_id)
    if not user:
        return None
```

### 2. バリデーション

```python
# Good
from pydantic import BaseModel, EmailStr, validator

class CreateUserRequest(BaseModel):
    name: str
    email: EmailStr
    age: int

    @validator('age')
    def validate_age(cls, v):
        if v < 0 or v > 150:
            raise ValueError('Invalid age')
        return v

# Bad
@app.post("/users")
async def create_user(name: str, email: str, age: int):
    if not name or not email:
        raise HTTPException(status_code=400)
```

## パフォーマンス最適化

### 1. データベースアクセス

```python
# Good
from sqlalchemy import select
from sqlalchemy.orm import joinedload

async def get_user_with_posts(user_id: str):
    query = select(User).options(
        joinedload(User.posts)
    ).where(User.id == user_id)

    return await session.execute(query)

# Bad
async def get_user_with_posts(user_id: str):
    user = await session.get(User, user_id)
    posts = await session.query(Post).filter(
        Post.user_id == user_id
    ).all()
```

### 2. キャッシュ

```python
# Good
from functools import lru_cache
from redis import Redis

redis = Redis()

async def get_user_cache(user_id: str) -> Optional[User]:
    # Redisから取得
    cached = await redis.get(f"user:{user_id}")
    if cached:
        return User.parse_raw(cached)

    # DBから取得してキャッシュ
    user = await repository.find_by_id(user_id)
    if user:
        await redis.set(
            f"user:{user_id}",
            user.json(),
            ex=3600
        )
    return user

# Bad
async def get_user(user_id: str):
    return await repository.find_by_id(user_id)
```

## セキュリティ

### 1. 認証

```python
# Good
from jose import jwt
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"])

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict) -> str:
    return jwt.encode(
        data,
        settings.SECRET_KEY,
        algorithm="HS256"
    )

# Bad
def verify_password(password: str, stored_password: str):
    return password == stored_password
```

### 2. 入力検証

```python
# Good
from pydantic import BaseModel, constr

class LoginRequest(BaseModel):
    email: EmailStr
    password: constr(min_length=8, max_length=64)

# Bad
@app.post("/login")
async def login(email: str, password: str):
    if "@" not in email:
        raise HTTPException(status_code=400)
```

## テスト

### 1. ユニットテスト

```python
# Good
import pytest
from unittest.mock import Mock

@pytest.fixture
def user_repository():
    return Mock(UserRepository)

def test_get_user_success(user_repository):
    # Arrange
    user_id = "test-id"
    expected_user = User(id=user_id, name="Test")
    user_repository.find_by_id.return_value = expected_user

    service = UserService(user_repository)

    # Act
    user = await service.get_user(user_id)

    # Assert
    assert user == expected_user
    user_repository.find_by_id.assert_called_once_with(user_id)

# Bad
def test_user():
    service = UserService(repository)
    user = service.get_user("123")
    assert user is not None
```

### 2. 統合テスト

```python
# Good
from fastapi.testclient import TestClient

def test_create_user_integration(client: TestClient):
    # Arrange
    user_data = {
        "name": "Test User",
        "email": "test@example.com",
        "age": 30
    }

    # Act
    response = client.post("/users", json=user_data)

    # Assert
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == user_data["name"]
    assert data["email"] == user_data["email"]

# Bad
def test_api():
    response = client.post("/users", json={})
    assert response.status_code == 201
```

## ロギング

### 1. 構造化ロギング

```python
# Good
import structlog

logger = structlog.get_logger()

async def process_order(order_id: str):
    logger.info(
        "processing_order",
        order_id=order_id,
        timestamp=datetime.now()
    )

    try:
        result = await process_payment(order_id)
        logger.info(
            "payment_processed",
            order_id=order_id,
            amount=result.amount
        )
    except Exception as e:
        logger.error(
            "payment_failed",
            order_id=order_id,
            error=str(e)
        )

# Bad
import logging

logging.info(f"Processing order {order_id}")
```

### 2. メトリクス

```python
# Good
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path
    ).inc()

    with REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).time():
        response = await call_next(request)

    return response

# Bad
@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"Request to {request.url}")
    return await call_next(request)
```

## ドキュメント

### 1. API ドキュメント

```python
# Good
from fastapi import APIRouter, Path, Query

router = APIRouter()

@router.get(
    "/users/{user_id}",
    response_model=UserResponse,
    responses={
        404: {"description": "User not found"},
        500: {"description": "Internal server error"}
    }
)
async def get_user(
    user_id: str = Path(..., description="The ID of the user"),
    include_posts: bool = Query(False, description="Include user's posts")
):
    """
    指定されたIDのユーザー情報を取得する。

    - user_id: ユーザーID
    - include_posts: 投稿情報も含めるかどうか
    """
    pass

# Bad
@router.get("/users/{id}")
async def get_user(id: str, include_posts: bool):
    pass
```

### 2. コードドキュメント

```python
# Good
class UserService:
    """ユーザー関連のビジネスロジックを提供するサービス

    このサービスは以下の機能を提供します：
    - ユーザーの作成/更新/削除
    - ユーザー情報の取得
    - ユーザー認証

    Attributes:
        repository: ユーザーリポジトリ
        auth_service: 認証サービス
    """

    def __init__(
        self,
        repository: UserRepository,
        auth_service: AuthService
    ):
        self.repository = repository
        self.auth_service = auth_service

# Bad
class UserService:
    def __init__(self, repo, auth):
        self.repo = repo
        self.auth = auth
```
