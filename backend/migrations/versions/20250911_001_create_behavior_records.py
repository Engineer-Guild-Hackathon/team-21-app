"""create behavior_records table

Revision ID: 20250911_001_create_behavior_records
Revises: 20240306_001_create_users
Create Date: 2025-09-11
"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "20250911_001"
down_revision = "05b3400baf1d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "behavior_records",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("user_id", sa.Integer, nullable=False, index=True),
        sa.Column("problem_id", sa.Integer, nullable=True, index=True),
        sa.Column("action_type", sa.String(length=64), nullable=False),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "attempt_count", sa.Integer, nullable=False, server_default=sa.text("1")
        ),
        sa.Column("success", sa.Boolean, nullable=True),
        sa.Column("approach_description", sa.Text, nullable=True),
        sa.Column("emotion_state", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )


def downgrade() -> None:
    op.drop_table("behavior_records")
