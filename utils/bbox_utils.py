"""
Bounding Box Utilities - Comprehensive helper functions for bbox manipulation
================================================================================

This module provides clear, well-documented functions for all common bbox operations.
Use this as your reference guide to never get confused about bbox manipulation again!

Author: AI Assistant
Purpose: Educational reference for bbox operations in computer vision
"""

from typing import Optional, Tuple


def validate_bbox(bbox: Tuple[float, float, float, float]) -> bool:
    """
    Validate if a bounding box is properly formatted.

    Args:
        bbox: (x1, y1, x2, y2) tuple

    Returns:
        True if valid, False otherwise

    Example:
        >>> validate_bbox((10, 20, 50, 80))  # Valid
        True
        >>> validate_bbox((50, 20, 10, 80))  # Invalid: x1 > x2
        False
    """
    x1, y1, x2, y2 = bbox
    return x1 < x2 and y1 < y2 and all(coord >= 0 for coord in bbox)


def get_bbox_dimensions(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Get width and height of a bounding box.

    Args:
        bbox: (x1, y1, x2, y2) tuple

    Returns:
        (width, height) tuple

    Example:
        >>> get_bbox_dimensions((10, 20, 50, 80))
        (40, 60)
    """
    x1, y1, x2, y2 = bbox
    return x2 - x1, y2 - y1


def get_bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Get center point of a bounding box.

    Args:
        bbox: (x1, y1, x2, y2) tuple

    Returns:
        (center_x, center_y) tuple

    Example:
        >>> get_bbox_center((10, 20, 50, 80))
        (30.0, 50.0)
    """
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def get_bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    """
    Get area of a bounding box.

    Args:
        bbox: (x1, y1, x2, y2) tuple

    Returns:
        Area as float

    Example:
        >>> get_bbox_area((10, 20, 50, 80))
        2400.0
    """
    width, height = get_bbox_dimensions(bbox)
    return width * height


def is_point_inside_bbox(
    point: Tuple[float, float], bbox: Tuple[float, float, float, float]
) -> bool:
    """
    Check if a point is inside a bounding box.

    Args:
        point: (x, y) tuple
        bbox: (x1, y1, x2, y2) tuple

    Returns:
        True if point is inside bbox

    Example:
        >>> is_point_inside_bbox((30, 50), (10, 20, 50, 80))
        True
        >>> is_point_inside_bbox((60, 50), (10, 20, 50, 80))
        False
    """
    px, py = point
    x1, y1, x2, y2 = bbox
    return x1 <= px <= x2 and y1 <= py <= y2


def do_bboxes_overlap(
    bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]
) -> bool:
    """
    Check if two bounding boxes overlap.

    Args:
        bbox1: (x1, y1, x2, y2) tuple
        bbox2: (x1, y1, x2, y2) tuple

    Returns:
        True if bboxes overlap

    Example:
        >>> do_bboxes_overlap((10, 10, 50, 50), (30, 30, 70, 70))
        True
        >>> do_bboxes_overlap((10, 10, 30, 30), (40, 40, 70, 70))
        False
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Two bboxes don't overlap if one is completely to the left, right, above, or below the other
    return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)


def get_bbox_intersection(
    bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]
) -> Optional[Tuple[float, float, float, float]]:
    """
    Get intersection bounding box of two bboxes.

    Args:
        bbox1: (x1, y1, x2, y2) tuple
        bbox2: (x1, y1, x2, y2) tuple

    Returns:
        Intersection bbox or None if no intersection

    Example:
        >>> get_bbox_intersection((10, 10, 50, 50), (30, 30, 70, 70))
        (30, 30, 50, 50)
    """
    if not do_bboxes_overlap(bbox1, bbox2):
        return None

    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Intersection is the overlap area
    ix1 = max(x1_1, x1_2)
    iy1 = max(y1_1, y1_2)
    ix2 = min(x2_1, x2_2)
    iy2 = min(y2_1, y2_2)

    return (ix1, iy1, ix2, iy2)


def calculate_iou(
    bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]
) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.

    Args:
        bbox1: (x1, y1, x2, y2) tuple
        bbox2: (x1, y1, x2, y2) tuple

    Returns:
        IoU value between 0 and 1

    Example:
        >>> calculate_iou((10, 10, 50, 50), (30, 30, 70, 70))
        0.25  # 25% overlap
    """
    intersection = get_bbox_intersection(bbox1, bbox2)
    if intersection is None:
        return 0.0

    intersection_area = get_bbox_area(intersection)
    area1 = get_bbox_area(bbox1)
    area2 = get_bbox_area(bbox2)
    union_area = area1 + area2 - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0


def expand_bbox(
    bbox: Tuple[float, float, float, float], factor: float = 1.2
) -> Tuple[float, float, float, float]:
    """
    Expand a bounding box by a factor while keeping it centered.

    Args:
        bbox: (x1, y1, x2, y2) tuple
        factor: Expansion factor (1.0 = no change, 1.2 = 20% larger)

    Returns:
        Expanded bbox

    Example:
        >>> expand_bbox((10, 10, 50, 50), 1.2)
        (6.0, 6.0, 54.0, 54.0)
    """
    x1, y1, x2, y2 = bbox
    center_x, center_y = get_bbox_center(bbox)
    width, height = get_bbox_dimensions(bbox)

    new_width = width * factor
    new_height = height * factor

    new_x1 = center_x - new_width / 2
    new_y1 = center_y - new_height / 2
    new_x2 = center_x + new_width / 2
    new_y2 = center_y + new_height / 2

    return (new_x1, new_y1, new_x2, new_y2)


def convert_bbox_format(
    bbox: Tuple[float, float, float, float],
    from_format: str = "xyxy",
    to_format: str = "xywh",
) -> Tuple[float, float, float, float]:
    """
    Convert bbox between different formats.

    Args:
        bbox: Input bbox tuple
        from_format: "xyxy" (x1,y1,x2,y2) or "xywh" (x,y,w,h) or "cxcywh" (center_x,center_y,w,h)
        to_format: "xyxy", "xywh", or "cxcywh"

    Returns:
        Converted bbox tuple

    Example:
        >>> convert_bbox_format((10, 20, 50, 80), "xyxy", "xywh")
        (10, 20, 40, 60)
        >>> convert_bbox_format((10, 20, 40, 60), "xywh", "cxcywh")
        (30, 50, 40, 60)
    """
    if from_format == to_format:
        return bbox

    # Convert to xyxy first
    if from_format == "xyxy":
        x1, y1, x2, y2 = bbox
    elif from_format == "xywh":
        x, y, w, h = bbox
        x1, y1, x2, y2 = x, y, x + w, y + h
    elif from_format == "cxcywh":
        cx, cy, w, h = bbox
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    else:
        raise ValueError(f"Unsupported format: {from_format}")

    # Convert from xyxy to target format
    if to_format == "xyxy":
        return (x1, y1, x2, y2)
    elif to_format == "xywh":
        return (x1, y1, x2 - x1, y2 - y1)
    elif to_format == "cxcywh":
        return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
    else:
        raise ValueError(f"Unsupported format: {to_format}")


def visualize_bbox_info(
    bbox: Tuple[float, float, float, float], label: str = "BBox"
) -> str:
    """
    Create a visual representation of bbox information.

    Args:
        bbox: (x1, y1, x2, y2) tuple
        label: Label for the bbox

    Returns:
        Formatted string with bbox visualization

    Example:
        >>> print(visualize_bbox_info((10, 20, 50, 80), "Player"))
    """
    x1, y1, x2, y2 = bbox
    width, height = get_bbox_dimensions(bbox)
    center_x, center_y = get_bbox_center(bbox)
    area = get_bbox_area(bbox)

    return f"""
{label} Information:
┌───────────────────────────────────────────────┐
│ Top-left: ({x1:4.1f}, {y1:4.1f})              │
│ Top-right: ({x2:4.1f}, {y1:4.1f})             │
│ Bottom-left: ({x1:4.1f}, {y2:4.1f})           │
│ Bottom-right: ({x2:4.1f}, {y2:4.1f})          │
│ Center: ({center_x:4.1f}, {center_y:4.1f})    │
│ Width: {width:4.1f}                           │
│ Height: {height:4.1f}                         │
│ Area: {area:4.1f}                             │
└───────────────────────────────────────────────┘
"""


def measure_distance_between_points(
    point1: Tuple[float, float], point2: Tuple[float, float]
) -> float:
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
    # return np.linalg.norm(np.array(point1) - np.array(point2))


# Example usage and testing
if __name__ == "__main__":
    # Example bboxes
    player_bbox = (10, 20, 50, 80)  # x1, y1, x2, y2
    ball_center = (30, 50)

    print("🎯 BBOX MANIPULATION EXAMPLES")
    print("=" * 50)

    # Basic operations
    print(f"Player bbox: {player_bbox}")
    print(f"Ball center: {ball_center}")
    print(
        f"Is ball inside player bbox? {is_point_inside_bbox(ball_center, player_bbox)}"
    )
    print(f"Player bbox dimensions: {get_bbox_dimensions(player_bbox)}")
    print(f"Player bbox center: {get_bbox_center(player_bbox)}")
    print(f"Player bbox area: {get_bbox_area(player_bbox)}")

    # Visual representation
    print(visualize_bbox_info(player_bbox, "Player"))

    # Format conversions
    print("Format conversions:")
    print(f"xyxy to xywh: {convert_bbox_format(player_bbox, 'xyxy', 'xywh')}")
    print(f"xyxy to cxcywh: {convert_bbox_format(player_bbox, 'xyxy', 'cxcywh')}")

    # Expansion
    expanded = expand_bbox(player_bbox, 1.2)
    print(f"Expanded bbox (1.2x): {expanded}")

    # IoU example
    bbox2 = (30, 30, 70, 70)
    print(
        f"IoU between {player_bbox} and {bbox2}: {calculate_iou(player_bbox, bbox2):.3f}"
    )
