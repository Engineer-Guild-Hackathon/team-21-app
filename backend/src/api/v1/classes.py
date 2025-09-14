import secrets
import string
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.security import get_current_active_user
from src.domain.models.classroom import Class as ClassModel
from src.domain.models.classroom import LearningProgress
from src.domain.models.user import User
from src.domain.schemas.classroom import (
    ClassCreate,
    ClassResponse,
    LearningProgressCreate,
    LearningProgressResponse,
    StudentProgressResponse,
)
from src.infrastructure.database import get_db

router = APIRouter()


def generate_class_id() -> str:
    """クラスIDを生成（例：ABC-123）"""
    letters = "".join(secrets.choice(string.ascii_uppercase) for _ in range(3))
    numbers = "".join(secrets.choice(string.digits) for _ in range(3))
    return f"{letters}-{numbers}"


@router.post("/", response_model=ClassResponse)
async def create_class(
    class_data: ClassCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """クラスを作成（教師のみ）"""
    if current_user.role != "teacher":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="教師のみがクラスを作成できます",
        )

    # クラスIDを生成
    class_id = generate_class_id()

    # 同じクラスIDが存在しないかチェック
    while True:
        result = await db.execute(
            select(ClassModel).where(ClassModel.class_id == class_id)
        )
        existing_class = result.scalar_one_or_none()
        if not existing_class:
            break
        class_id = generate_class_id()

    # クラスを作成
    db_class = ClassModel(
        class_id=class_id,
        name=class_data.name,
        description=class_data.description,
        teacher_id=current_user.id,
    )
    db.add(db_class)
    await db.commit()
    await db.refresh(db_class)

    return db_class


@router.get("/my-classes", response_model=List[ClassResponse])
async def get_my_classes(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """自分のクラス一覧を取得"""
    if current_user.role == "teacher":
        # 教師の場合：自分が作成したクラス
        result = await db.execute(
            select(ClassModel).where(ClassModel.teacher_id == current_user.id)
        )
        classes = result.scalars().all()
    elif current_user.role == "student":
        # 生徒の場合：所属しているクラス
        result = await db.execute(
            select(ClassModel).where(ClassModel.id == current_user.class_id)
        )
        classes = result.scalars().all()
    else:
        # 保護者の場合：子どものクラス
        # TODO: 保護者と生徒の関係を実装する必要がある
        classes = []

    return classes


@router.get("/{class_id}/students", response_model=List[StudentProgressResponse])
async def get_class_students(
    class_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """クラスの生徒一覧と進捗を取得（教師のみ）"""
    if current_user.role != "teacher":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="教師のみがクラスの生徒一覧を確認できます",
        )

    # クラスを取得
    result = await db.execute(select(ClassModel).where(ClassModel.class_id == class_id))
    class_obj = result.scalar_one_or_none()
    if not class_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="クラスが見つかりません"
        )

    # 教師がそのクラスの作成者かチェック
    if class_obj.teacher_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="このクラスへのアクセス権限がありません",
        )

    # クラスの生徒一覧を取得
    result = await db.execute(select(User).where(User.class_id == class_obj.id))
    students = result.scalars().all()

    # 各生徒の進捗を取得
    student_progress_list = []
    for student in students:
        result = await db.execute(
            select(LearningProgress).where(
                LearningProgress.student_id == student.id,
                LearningProgress.class_id == class_obj.id,
            )
        )
        progress = result.scalar_one_or_none()

        # 進捗が存在しない場合はデフォルト値で作成
        if not progress:
            progress = LearningProgress(student_id=student.id, class_id=class_obj.id)
            db.add(progress)
            await db.commit()
            await db.refresh(progress)

        student_progress_list.append(
            StudentProgressResponse(
                student_id=student.id,
                student_name=student.full_name or student.email,
                student_email=student.email,
                progress=progress,
            )
        )

    return student_progress_list


@router.post("/{class_id}/join", response_model=dict)
async def join_class(
    class_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """クラスに参加（生徒のみ）"""
    if current_user.role != "student":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="生徒のみがクラスに参加できます",
        )

    # クラスを取得
    result = await db.execute(select(ClassModel).where(ClassModel.class_id == class_id))
    class_obj = result.scalar_one_or_none()
    if not class_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="クラスが見つかりません"
        )

    # 既にクラスに所属しているかチェック
    if current_user.class_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="既にクラスに所属しています"
        )

    # クラスに参加
    current_user.class_id = class_obj.id
    await db.commit()

    # 学習進捗レコードを作成
    progress = LearningProgress(student_id=current_user.id, class_id=class_obj.id)
    db.add(progress)
    await db.commit()

    return {"message": "クラスに参加しました", "class_id": class_id}


@router.put("/progress/{student_id}", response_model=LearningProgressResponse)
async def update_learning_progress(
    student_id: int,
    progress_data: LearningProgressCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """学習進捗を更新"""
    # 進捗レコードを取得または作成
    result = await db.execute(
        select(LearningProgress).where(
            LearningProgress.student_id == student_id,
            LearningProgress.class_id == progress_data.class_id,
        )
    )
    progress = result.scalar_one_or_none()

    if not progress:
        # 新規作成
        progress = LearningProgress(
            student_id=student_id,
            class_id=progress_data.class_id,
            grit_score=progress_data.grit_score,
            collaboration_score=progress_data.collaboration_score,
            self_regulation_score=progress_data.self_regulation_score,
            emotional_intelligence_score=progress_data.emotional_intelligence_score,
            quests_completed=progress_data.quests_completed,
            total_learning_time=progress_data.total_learning_time,
            retry_count=progress_data.retry_count,
        )
        db.add(progress)
    else:
        # 更新
        progress.grit_score = progress_data.grit_score
        progress.collaboration_score = progress_data.collaboration_score
        progress.self_regulation_score = progress_data.self_regulation_score
        progress.emotional_intelligence_score = (
            progress_data.emotional_intelligence_score
        )
        progress.quests_completed = progress_data.quests_completed
        progress.total_learning_time = progress_data.total_learning_time
        progress.retry_count = progress_data.retry_count

    await db.commit()
    await db.refresh(progress)
    return progress
