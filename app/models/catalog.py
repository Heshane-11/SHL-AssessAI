from pydantic import BaseModel
from typing import Optional, List


class CatalogItem(BaseModel):
    name: str
    url: str
    description: str
    duration: Optional[str] = None
    remote_testing: Optional[str] = None
    adaptive_irt: Optional[str] = None
    test_type: str
    skills_measured: List[str] = []
