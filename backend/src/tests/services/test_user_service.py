from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest

from ...domain.models.user import User
from ...domain.types.user import Email, UserId, UserProfile
from ...services.user.user_service import UserService


@pytest.fixture
def mock_repository():
    return Mock(
        create=AsyncMock(),
        read=AsyncMock(),
        update=AsyncMock(),
        delete=AsyncMock(),
        list=AsyncMock(),
        find_by_email=AsyncMock(),
        find_active_users=AsyncMock(),
        update_last_login=AsyncMock(),
    )


@pytest.fixture
def user_service(mock_repository):
    return UserService(mock_repository)


@pytest.fixture
def sample_user():
    return User(
        id=1,
        email="test@example.com",
        hashed_password="hashed",
        full_name="Test User",
        is_active=True,
        is_verified=False,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


@pytest.mark.asyncio
async def test_create_user(user_service, mock_repository, sample_user):
    # Arrange
    mock_repository.find_by_email.return_value = None
    mock_repository.create.return_value = sample_user

    # Act
    user = await user_service.create_user(
        email=Email("test@example.com"), password="password123", full_name="Test User"
    )

    # Assert
    assert user.email == "test@example.com"
    assert user.full_name == "Test User"
    mock_repository.create.assert_called_once()


@pytest.mark.asyncio
async def test_create_user_duplicate_email(user_service, mock_repository, sample_user):
    # Arrange
    mock_repository.find_by_email.return_value = sample_user

    # Act & Assert
    with pytest.raises(ValueError, match="Email already registered"):
        await user_service.create_user(
            email=Email("test@example.com"),
            password="password123",
            full_name="Test User",
        )


@pytest.mark.asyncio
async def test_get_user(user_service, mock_repository, sample_user):
    # Arrange
    mock_repository.read.return_value = sample_user

    # Act
    user = await user_service.get_user(UserId(1))

    # Assert
    assert user == sample_user
    mock_repository.read.assert_called_once_with(UserId(1))


@pytest.mark.asyncio
async def test_update_profile(user_service, mock_repository, sample_user):
    # Arrange
    mock_repository.read.return_value = sample_user
    mock_repository.update.return_value = sample_user

    profile = UserProfile(
        full_name="Updated Name",
        avatar_url="https://example.com/avatar.jpg",
        bio="Test bio",
    )

    # Act
    updated_user = await user_service.update_profile(UserId(1), profile)

    # Assert
    assert updated_user.full_name == profile.full_name
    assert updated_user.avatar_url == profile.avatar_url
    assert updated_user.bio == profile.bio
    mock_repository.update.assert_called_once()


@pytest.mark.asyncio
async def test_delete_user(user_service, mock_repository):
    # Arrange
    mock_repository.delete.return_value = True

    # Act
    result = await user_service.delete_user(UserId(1))

    # Assert
    assert result is True
    mock_repository.delete.assert_called_once_with(UserId(1))
