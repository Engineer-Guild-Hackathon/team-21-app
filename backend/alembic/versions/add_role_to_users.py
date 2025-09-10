"""add role to users

Revision ID: add_role_to_users
Revises: create_users_table
Create Date: 2024-02-10 12:00:00.000000

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "add_role_to_users"
down_revision = "create_users_table"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # roleカラムの追加
    op.add_column(
        "users",
        sa.Column("role", sa.String(), nullable=False, server_default="student"),
    )


def downgrade() -> None:
    # roleカラムの削除
    op.drop_column("users", "role")
