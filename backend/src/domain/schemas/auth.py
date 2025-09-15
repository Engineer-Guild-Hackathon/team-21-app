from typing import Optional

from pydantic import BaseModel, ConfigDict, EmailStr, field_validator


class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None


class UserCreate(UserBase):
    password: str
    password_confirm: str
    role: Optional[str] = "student"
    class_id: Optional[str] = None  # クラスID（文字列）
    terms_accepted: bool = False

    @field_validator("password_confirm")
    @classmethod
    def validate_password_confirm(cls, v, info):
        if "password" in info.data and v != info.data["password"]:
            raise ValueError("パスワードが一致しません")
        return v


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(UserBase):
    id: int
    role: str
    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[str] = None
