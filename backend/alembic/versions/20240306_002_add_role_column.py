"""add role column to users

Revision ID: 20240306_002
Revises: 20240306_001
Create Date: 2024-03-06 00:20:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "20240306_002"
down_revision = "05b3400baf1d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("role", sa.String(), nullable=False, server_default="student"),
    )


def downgrade() -> None:
    op.drop_column("users", "role")
