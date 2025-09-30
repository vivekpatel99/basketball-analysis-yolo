#!/usr/bin/env python3
"""
Bounding Box Demo - Interactive examples to master bbox manipulation
====================================================================

Run this script to see practical examples of all bbox operations.
This will help you understand bbox manipulation through real examples.

Usage:
    python examples/bbox_demo.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.bbox_utils import (
    calculate_iou,
    convert_bbox_format,
    do_bboxes_overlap,
    expand_bbox,
    get_bbox_area,
    get_bbox_center,
    get_bbox_dimensions,
    get_bbox_intersection,
    is_point_inside_bbox,
    validate_bbox,
    visualize_bbox_info,
)


def demo_basic_operations():
    """Demonstrate basic bbox operations."""
    print("🎯 BASIC BBOX OPERATIONS")
    print("=" * 60)

    # Example bboxes
    player_bbox = (100, 150, 200, 300)  # x1, y1, x2, y2
    ball_center = (150, 225)

    print(f"Player bbox: {player_bbox}")
    print(f"Ball center: {ball_center}")
    print()

    # Basic calculations
    width, height = get_bbox_dimensions(player_bbox)  # noqa: F405
    center_x, center_y = get_bbox_center(player_bbox)  # noqa: F405
    area = get_bbox_area(player_bbox)  # noqa: F405

    print(f"📐 Dimensions: {width} × {height}")
    print(f"📍 Center: ({center_x}, {center_y})")
    print(f"📊 Area: {area}")
    print()

    # Point inside check
    is_inside = is_point_inside_bbox(ball_center, player_bbox)
    print(f"🎾 Is ball inside player bbox? {is_inside}")
    print()

    # Visual representation
    print(visualize_bbox_info(player_bbox, "Player"))


def demo_overlap_operations():
    """Demonstrate bbox overlap operations."""
    print("\n🔄 BBOX OVERLAP OPERATIONS")
    print("=" * 60)

    # Two overlapping bboxes
    bbox1 = (50, 50, 150, 150)
    bbox2 = (100, 100, 200, 200)

    print(f"Bbox 1: {bbox1}")
    print(f"Bbox 2: {bbox2}")
    print()

    # Check overlap
    overlap = do_bboxes_overlap(bbox1, bbox2)
    print(f"🔄 Do they overlap? {overlap}")

    if overlap:
        intersection = get_bbox_intersection(bbox1, bbox2)
        print(f"📦 Intersection: {intersection}")

        iou = calculate_iou(bbox1, bbox2)
        print(f"📊 IoU: {iou:.3f}")

    print()

    # Non-overlapping example
    bbox3 = (300, 300, 400, 400)
    overlap2 = do_bboxes_overlap(bbox1, bbox3)
    print(f"Bbox 3: {bbox3}")
    print(f"🔄 Do bbox1 and bbox3 overlap? {overlap2}")


def demo_format_conversions():
    """Demonstrate bbox format conversions."""
    print("\n🔄 BBOX FORMAT CONVERSIONS")
    print("=" * 60)

    # Original bbox in xyxy format
    bbox_xyxy = (100, 150, 200, 300)
    print(f"Original (xyxy): {bbox_xyxy}")
    print()

    # Convert to different formats
    bbox_xywh = convert_bbox_format(bbox_xyxy, "xyxy", "xywh")
    bbox_cxcywh = convert_bbox_format(bbox_xyxy, "xyxy", "cxcywh")

    print(f"📦 xywh format: {bbox_xywh}")
    print(f"📦 cxcywh format: {bbox_cxcywh}")
    print()

    # Convert back to verify
    back_to_xyxy = convert_bbox_format(bbox_xywh, "xywh", "xyxy")
    print(f"🔄 Back to xyxy: {back_to_xyxy}")
    print(f"✅ Conversion correct? {bbox_xyxy == back_to_xyxy}")


def demo_bbox_expansion():
    """Demonstrate bbox expansion."""
    print("\n📈 BBOX EXPANSION")
    print("=" * 60)

    original_bbox = (100, 100, 200, 200)
    print(f"Original bbox: {original_bbox}")
    print(visualize_bbox_info(original_bbox, "Original"))

    # Expand by different factors
    for factor in [1.1, 1.2, 1.5, 2.0]:
        expanded = expand_bbox(original_bbox, factor)
        print(f"Expanded by {factor}x: {expanded}")
        print(visualize_bbox_info(expanded, f"Expanded {factor}x"))


def demo_basketball_specific():
    """Demonstrate basketball-specific bbox operations."""
    print("\n🏀 BASKETBALL-SPECIFIC EXAMPLES")
    print("=" * 60)

    # Player bbox
    player_bbox = (100, 150, 200, 300)
    ball_center = (150, 225)

    print(f"Player bbox: {player_bbox}")
    print(f"Ball center: {ball_center}")
    print()

    # Check if ball is in player's possession area
    is_ball_in_player = is_point_inside_bbox(ball_center, player_bbox)
    print(f"🎾 Ball in player possession? {is_ball_in_player}")

    # Calculate distance from ball to player center
    player_center = get_bbox_center(player_bbox)
    distance = (
        (ball_center[0] - player_center[0]) ** 2
        + (ball_center[1] - player_center[1]) ** 2
    ) ** 0.5
    print(f"📏 Distance from ball to player center: {distance:.1f} pixels")

    # Check if ball is in expanded player area (for loose ball detection)
    expanded_player = expand_bbox(player_bbox, 1.5)
    is_ball_near_player = is_point_inside_bbox(ball_center, expanded_player)
    print(f"🎾 Ball near player (1.5x expanded)? {is_ball_near_player}")

    print()
    print("Visual comparison:")
    print(visualize_bbox_info(player_bbox, "Player"))
    print(visualize_bbox_info(expanded_player, "Expanded Player"))


def demo_common_mistakes():
    """Demonstrate common bbox mistakes and how to avoid them."""
    print("\n⚠️  COMMON MISTAKES & HOW TO AVOID THEM")
    print("=" * 60)

    # Mistake 1: Confusing coordinate order
    print("❌ MISTAKE 1: Confusing (x1,y1) with (x2,y2)")
    invalid_bbox = (200, 300, 100, 150)  # x1 > x2, y1 > y2
    print(f"Invalid bbox: {invalid_bbox}")
    print(f"Is valid? {validate_bbox(invalid_bbox)}")

    correct_bbox = (100, 150, 200, 300)  # x1 < x2, y1 < y2
    print(f"Correct bbox: {correct_bbox}")
    print(f"Is valid? {validate_bbox(correct_bbox)}")
    print()

    # Mistake 2: Wrong coordinate system
    print("❌ MISTAKE 2: Forgetting Y increases downward")
    print("In computer vision: (0,0) is TOP-LEFT, Y increases DOWN")
    print("In math graphs: (0,0) is BOTTOM-LEFT, Y increases UP")
    print("Always remember: Y goes DOWN in images!")
    print()

    # Mistake 3: Zero-width/height bboxes
    print("❌ MISTAKE 3: Zero-width or zero-height bboxes")
    zero_width = (100, 100, 100, 200)  # width = 0
    zero_height = (100, 100, 200, 100)  # height = 0
    print(f"Zero width bbox: {zero_width} -> Valid? {validate_bbox(zero_width)}")
    print(f"Zero height bbox: {zero_height} -> Valid? {validate_bbox(zero_height)}")
    print()

    # Mistake 4: Negative coordinates
    print("❌ MISTAKE 4: Negative coordinates")
    negative_bbox = (-10, -20, 100, 200)
    print(f"Negative bbox: {negative_bbox} -> Valid? {validate_bbox(negative_bbox)}")
    print("Note: Some systems allow negative coords, but it's usually an error")


def main():
    """Run all bbox demonstrations."""
    print("🎯 COMPREHENSIVE BBOX MANIPULATION DEMO")
    print("=" * 80)
    print("This demo will help you master bounding box operations!")
    print("=" * 80)

    try:
        demo_basic_operations()
        demo_overlap_operations()
        demo_format_conversions()
        demo_bbox_expansion()
        demo_basketball_specific()
        demo_common_mistakes()

        print("\n🎉 DEMO COMPLETE!")
        print("=" * 80)
        print("You now have a solid understanding of bbox manipulation!")
        print("Key takeaways:")
        print("1. Always remember: (x1,y1) = top-left, (x2,y2) = bottom-right")
        print("2. Y increases downward in computer vision")
        print("3. Always validate: x1 < x2 and y1 < y2")
        print("4. Use the utility functions in utils/bbox_utils.py")
        print("5. Practice with different bbox formats (xyxy, xywh, cxcywh)")

    except Exception as e:
        print(f"❌ Error running demo: {e}")
        print("Make sure you're running from the project root directory")


if __name__ == "__main__":
    main()
