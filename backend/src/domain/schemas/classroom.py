from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class ClassBase(BaseModel):
    name: str
    description: Optional[str] = None


class ClassCreate(ClassBase):
    pass


class ClassUpdate(ClassBase):
    name: Optional[str] = None


class ClassResponse(ClassBase):
    id: int
    class_id: str
    teacher_id: int
    is_active: bool
    model_config = ConfigDict(from_attributes=True)


class ClassWithStudents(ClassResponse):
    students: List["UserResponse"] = []
    learning_progress: List["LearningProgressResponse"] = []


class LearningProgressBase(BaseModel):
    grit_score: float = 0.0
    collaboration_score: float = 0.0
    self_regulation_score: float = 0.0
    emotional_intelligence_score: float = 0.0
    quests_completed: int = 0
    total_learning_time: int = 0
    retry_count: int = 0


class LearningProgressCreate(LearningProgressBase):
    student_id: int
    class_id: int


class LearningProgressUpdate(LearningProgressBase):
    pass


class LearningProgressResponse(LearningProgressBase):
    id: int
    student_id: int
    class_id: int
    model_config = ConfigDict(from_attributes=True)


class StudentProgressResponse(BaseModel):
    student_id: int
    student_name: str
    student_email: str
    progress: LearningProgressResponse
    model_config = ConfigDict(from_attributes=True)


# 循環参照を解決するために後で更新
from .user import UserResponse  # noqa: E402,F401
