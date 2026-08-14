from typing import List, Optional

class DynamicMaskInterface:
    def __init__(self, excluded_object_ids: Optional[List[str]] = None, excluded_classes: Optional[List[str]] = None):
        self.excluded_object_ids = excluded_object_ids or []
        self.excluded_classes = excluded_classes or ["dynamic_person", "moving_vehicle"]

    def should_exclude(self, object_id: Optional[str] = None, object_class: Optional[str] = None) -> bool:
        if object_id and object_id in self.excluded_object_ids:
            return True
        if object_class and object_class in self.excluded_classes:
            return True
        return False
