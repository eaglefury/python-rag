from dataclasses import dataclass
import enum
from typing import Any, List, Optional


class AccessRole(enum.Enum):
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"


@dataclass
class DocumentMetadata:
    source: str
    tag: List[str]
    access_role: AccessRole = AccessRole.USER


@dataclass
class FileData:
    content: str
    name: str
