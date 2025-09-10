"""add role and verified

Revision ID: 20240306_002
Revises: 20240306_001
Create Date: 2024-03-06 00:00:00.000000

"""

from typing import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20240306_002"
down_revision: str = "20240306_001"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """アップグレード"""
    op.add_column("users", sa.Column("role", sa.String(), nullable=False, server_default="student"))
    op.add_column("users", sa.Column("is_verified", sa.Boolean(), nullable=False, server_default="false"))


def downgrade() -> None:
    """ダウングレード"""
    op.drop_column("users", "is_verified")
    op.drop_column("users", "role")
