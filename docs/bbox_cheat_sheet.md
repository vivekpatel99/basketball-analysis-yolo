# 🎯 Bounding Box (BBox) Cheat Sheet

_Your Ultimate Reference for Computer Vision BBox Operations_

______________________________________________________________________

## 📋 Table of Contents

- [Quick Reference](#-quick-reference)
- [Coordinate System](#-coordinate-system)
- [BBox Formats](#-bbox-formats)
- [Essential Operations](#-essential-operations)
- [Common Patterns](#-common-patterns)
- [Basketball-Specific Examples](#-basketball-specific-examples)
- [Common Mistakes](#-common-mistakes)
- [Code Examples](#-code-examples)

______________________________________________________________________

## 🚀 Quick Reference

### Basic BBox Format

```python
bbox = (x1, y1, x2, y2)  # Always in this order!
# x1, y1 = TOP-LEFT corner (smaller coordinates)
# x2, y2 = BOTTOM-RIGHT corner (larger coordinates)
# x1 < x2 and y1 < y2 (always!)
```

### Essential Calculations

```python
width = x2 - x1
height = y2 - y1
center_x = (x1 + x2) / 2
center_y = (y1 + y2) / 2
area = width * height
```

### Point Inside Check

```python
point_inside = (x1 <= px <= x2) and (y1 <= py <= y2)
```

______________________________________________________________________

## 🗺️ Coordinate System

### Computer Vision vs Math Graphs

```
Computer Vision (Images)          Math Graphs
┌─────────────────┐ (0,0)        (0,0) ┌─────────────────┐
│                 │                      │                 │
│        Y ↓      │                      │        Y ↑      │
│                 │                      │                 │
│ X →             │                      │ X →             │
└─────────────────┘                      └─────────────────┘
```

### Key Points

- **Origin (0,0)**: TOP-LEFT corner
- **X-axis**: Increases going RIGHT →
- **Y-axis**: Increases going DOWN ↓
- **Never forget**: Y goes DOWN, not UP!

______________________________________________________________________

## 📦 BBox Formats

### 1. XYXY Format (Most Common)

```python
bbox = (x1, y1, x2, y2)
# (100, 150, 200, 300)
#  ↑    ↑    ↑    ↑
#  │    │    │    └─ Bottom-right Y
#  │    │    └────── Bottom-right X
#  │    └─────────── Top-left Y
#  └──────────────── Top-left X
```

### 2. XYWH Format

```python
bbox = (x, y, width, height)
# (100, 150, 100, 150)
#  ↑    ↑    ↑     ↑
#  │    │    │     └─ Height
#  │    │    └─────── Width
#  │    └──────────── Top-left Y
#  └───────────────── Top-left X
```

### 3. CXCYWH Format (Center-based)

```python
bbox = (center_x, center_y, width, height)
# (150, 225, 100, 150)
#  ↑     ↑     ↑     ↑
#  │     │     │     └─ Height
#  │     │     └─────── Width
#  │     └───────────── Center Y
#  └─────────────────── Center X
```

### Format Conversions

```python
# XYXY → XYWH
x, y, w, h = x1, y1, x2-x1, y2-y1

# XYXY → CXCYWH
cx, cy, w, h = (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1

# XYWH → XYXY
x1, y1, x2, y2 = x, y, x+w, y+h

# CXCYWH → XYXY
x1, y1, x2, y2 = cx-w/2, cy-h/2, cx+w/2, cy+h/2
```

______________________________________________________________________

## ⚡ Essential Operations

### 1. Point Inside BBox

```python
def is_point_inside(point, bbox):
    px, py = point
    x1, y1, x2, y2 = bbox
    return x1 <= px <= x2 and y1 <= py <= y2
```

### 2. BBox Overlap Check

```python
def do_bboxes_overlap(bbox1, bbox2):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)
```

### 3. BBox Intersection

```python
def get_intersection(bbox1, bbox2):
    if not do_bboxes_overlap(bbox1, bbox2):
        return None

    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    ix1 = max(x1_1, x1_2)
    iy1 = max(y1_1, y1_2)
    ix2 = min(x2_1, x2_2)
    iy2 = min(y2_1, y2_2)

    return (ix1, iy1, ix2, iy2)
```

### 4. IoU (Intersection over Union)

```python
def calculate_iou(bbox1, bbox2):
    intersection = get_intersection(bbox1, bbox2)
    if intersection is None:
        return 0.0

    intersection_area = get_area(intersection)
    area1 = get_area(bbox1)
    area2 = get_area(bbox2)
    union_area = area1 + area2 - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0
```

### 5. BBox Expansion

```python
def expand_bbox(bbox, factor=1.2):
    x1, y1, x2, y2 = bbox
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    width, height = x2 - x1, y2 - y1

    new_width = width * factor
    new_height = height * factor

    return (center_x - new_width/2, center_y - new_height/2,
            center_x + new_width/2, center_y + new_height/2)
```

______________________________________________________________________

## 🎯 Common Patterns

### 1. Distance Between Points

```python
def distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
```

### 2. Distance from Point to BBox Center

```python
def distance_to_bbox_center(point, bbox):
    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    return distance(point, center)
```

### 3. Closest BBox to Point

```python
def find_closest_bbox(point, bboxes):
    distances = [distance_to_bbox_center(point, bbox) for bbox in bboxes]
    return bboxes[distances.index(min(distances))]
```

### 4. BBox Validation

```python
def validate_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return x1 < x2 and y1 < y2 and all(coord >= 0 for coord in bbox)
```

______________________________________________________________________

## 🏀 Basketball-Specific Examples

### Ball Possession Detection

```python
def is_ball_in_possession(ball_center, player_bbox, threshold=0.5):
    """Check if ball is in player's possession area."""
    return is_point_inside(ball_center, player_bbox)

def is_ball_near_player(ball_center, player_bbox, expansion_factor=1.5):
    """Check if ball is near player (expanded bbox)."""
    expanded_bbox = expand_bbox(player_bbox, expansion_factor)
    return is_point_inside(ball_center, expanded_bbox)
```

### Player-Player Interactions

```python
def players_colliding(player1_bbox, player2_bbox, iou_threshold=0.1):
    """Check if two players are colliding."""
    return calculate_iou(player1_bbox, player2_bbox) > iou_threshold

def get_player_interaction_points(player_bbox, ball_center):
    """Get key points for player-ball interaction analysis."""
    ball_x, ball_y = ball_center
    x1, y1, x2, y2 = player_bbox

    points = []

    # Horizontal intersection points
    if y1 <= ball_y <= y2:
        points.extend([(x1, ball_y), (x2, ball_y)])

    # Vertical intersection points
    if x1 <= ball_x <= x2:
        points.extend([(ball_x, y1), (ball_x, y2)])

    return points
```

### Court Area Detection

```python
def is_in_paint_area(bbox, court_dimensions):
    """Check if bbox is in the paint area."""
    paint_bbox = court_dimensions['paint']  # (x1, y1, x2, y2)
    return do_bboxes_overlap(bbox, paint_bbox)

def is_in_three_point_area(bbox, court_dimensions):
    """Check if bbox is in three-point area."""
    three_point_bbox = court_dimensions['three_point']
    return do_bboxes_overlap(bbox, three_point_bbox)
```

______________________________________________________________________

## ⚠️ Common Mistakes

### 1. Coordinate Order Confusion

```python
# ❌ WRONG - Don't confuse (x1,y1) with (x2,y2)
invalid_bbox = (200, 300, 100, 150)  # x1 > x2, y1 > y2

# ✅ CORRECT - Always x1 < x2 and y1 < y2
valid_bbox = (100, 150, 200, 300)
```

### 2. Y-Axis Direction

```python
# ❌ WRONG - Thinking Y goes up like in math
# In computer vision, Y increases DOWNWARD

# ✅ CORRECT - Remember Y goes DOWN
# (0,0) is TOP-LEFT, Y increases going DOWN
```

### 3. Zero-Dimension BBoxes

```python
# ❌ WRONG - Zero width or height
zero_width = (100, 100, 100, 200)   # width = 0
zero_height = (100, 100, 200, 100)  # height = 0

# ✅ CORRECT - Always have positive dimensions
valid_bbox = (100, 100, 200, 200)   # width = 100, height = 100
```

### 4. Negative Coordinates

```python
# ❌ WRONG - Negative coordinates (usually an error)
negative_bbox = (-10, -20, 100, 200)

# ✅ CORRECT - Non-negative coordinates
valid_bbox = (0, 0, 100, 200)
```

### 5. Format Mixing

```python
# ❌ WRONG - Mixing formats without conversion
bbox1 = (100, 150, 200, 300)  # xyxy format
bbox2 = (100, 150, 100, 150)  # xywh format
# Comparing these directly is wrong!

# ✅ CORRECT - Convert to same format first
bbox1_xywh = convert_to_xywh(bbox1)
# Now compare bbox1_xywh with bbox2
```

______________________________________________________________________

## 💻 Code Examples

### Complete BBox Class

```python
class BoundingBox:
    def __init__(self, x1, y1, x2, y2):
        if not (x1 < x2 and y1 < y2):
            raise ValueError("Invalid bbox: x1 < x2 and y1 < y2 required")
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def center(self):
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self):
        return self.width * self.height

    def contains_point(self, x, y):
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def overlaps(self, other):
        return not (self.x2 < other.x1 or other.x2 < self.x1 or
                   self.y2 < other.y1 or other.y2 < self.y1)

    def intersection(self, other):
        if not self.overlaps(other):
            return None

        return BoundingBox(
            max(self.x1, other.x1),
            max(self.y1, other.y1),
            min(self.x2, other.x2),
            min(self.y2, other.y2)
        )

    def iou(self, other):
        intersection = self.intersection(other)
        if intersection is None:
            return 0.0

        intersection_area = intersection.area
        union_area = self.area + other.area - intersection_area
        return intersection_area / union_area if union_area > 0 else 0.0

    def expand(self, factor=1.2):
        center_x, center_y = self.center
        new_width = self.width * factor
        new_height = self.height * factor

        return BoundingBox(
            center_x - new_width / 2,
            center_y - new_height / 2,
            center_x + new_width / 2,
            center_y + new_height / 2
        )

    def __repr__(self):
        return f"BoundingBox({self.x1}, {self.y1}, {self.x2}, {self.y2})"
```

### Usage Examples

```python
# Create bboxes
player = BoundingBox(100, 150, 200, 300)
ball_center = (150, 225)

# Check if ball is in player's possession
if player.contains_point(*ball_center):
    print("Ball is in player's possession!")

# Calculate distance to player center
distance = ((ball_center[0] - player.center[0])**2 +
            (ball_center[1] - player.center[1])**2)**0.5

# Expand player bbox for loose ball detection
expanded_player = player.expand(1.5)
if expanded_player.contains_point(*ball_center):
    print("Ball is near player!")

# Check player-player collision
player2 = BoundingBox(180, 160, 280, 310)
if player.overlaps(player2):
    iou = player.iou(player2)
    print(f"Players colliding with IoU: {iou:.3f}")
```

______________________________________________________________________

## 🎯 Quick Tips

1. **Always validate bboxes**: `x1 < x2 and y1 < y2`
2. **Remember Y goes DOWN**: Not up like in math graphs
3. **Use consistent formats**: Don't mix xyxy, xywh, cxcywh
4. **Test edge cases**: Zero dimensions, negative coords, etc.
5. **Use utility functions**: Don't reinvent the wheel
6. **Visualize when debugging**: Draw bboxes on images
7. **Consider performance**: Use numpy for batch operations

______________________________________________________________________

## 📚 Additional Resources

- **Your utility file**: `utils/bbox_utils.py`
- **Demo script**: `examples/bbox_demo.py`
- **In-code documentation**: Check your `ball_aquisition_detector.py`

______________________________________________________________________

_Happy coding! 🚀 Remember: Practice makes perfect. The more you work with bboxes, the more intuitive they become._
