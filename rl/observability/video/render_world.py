"""Render full-world Diep snapshots from a followed agent camera."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from rl.observability.video.render_overlay import draw_text

DEFAULT_FRAME_SIZE = (720, 720)
DEFAULT_VIEW_WORLD_SIZE = 1400.0

_COLOR_BY_KIND: dict[str, tuple[int, int, int]] = {
    "agent": (80, 180, 255),
    "shape": (245, 210, 70),
    "crasher": (245, 90, 130),
    "projectile": (255, 110, 80),
    "wall": (120, 120, 120),
}
_AGENT_COLORS = (
    (80, 180, 255),
    (120, 255, 140),
    (255, 140, 80),
    (210, 130, 255),
)


def _position(entity: Mapping[str, Any]) -> tuple[float, float]:
    pos = entity.get("position") or {}
    return float(pos.get("x", 0.0)), float(pos.get("y", 0.0))


def _angle(entity: Mapping[str, Any]) -> float:
    return float((entity.get("position") or {}).get("angle", 0.0))


def _size(entity: Mapping[str, Any]) -> float:
    physics = entity.get("physics") or {}
    return max(2.0, float(physics.get("size", physics.get("width", 10.0))))


def _sides(entity: Mapping[str, Any]) -> int:
    return max(1, int((entity.get("physics") or {}).get("sides", 1)))


def _agent_entity(snapshot: Mapping[str, Any], agent_id: int) -> Mapping[str, Any] | None:
    for entity in snapshot.get("entities", ()) or ():
        if entity.get("kind") == "agent" and int(entity.get("agentIndex", entity.get("id", -1))) == agent_id:
            return entity
    return None


def _world_to_screen(
    x: float,
    y: float,
    *,
    camera: tuple[float, float],
    scale: float,
    width: int,
    height: int,
) -> tuple[int, int]:
    return int(width / 2 + (x - camera[0]) * scale), int(height / 2 + (y - camera[1]) * scale)


def _draw_pixel(frame: np.ndarray, x: int, y: int, color: Sequence[int]) -> None:
    if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
        frame[y, x] = color


def _draw_line(frame: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: Sequence[int]) -> None:
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        _draw_pixel(frame, x0, y0, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _draw_circle(frame: np.ndarray, center: tuple[int, int], radius: int, color: Sequence[int], *, fill: bool = True) -> None:
    cx, cy = center
    r = max(1, int(radius))
    r2 = r * r
    inner = max(0, r - 1) ** 2
    for y in range(cy - r, cy + r + 1):
        for x in range(cx - r, cx + r + 1):
            d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
            if d2 <= r2 and (fill or d2 >= inner):
                _draw_pixel(frame, x, y, color)


def _polygon_points(center: tuple[int, int], radius: int, sides: int, angle: float) -> list[tuple[int, int]]:
    if sides <= 1:
        return []
    return [
        (
            int(center[0] + math.cos(angle + 2 * math.pi * i / sides) * radius),
            int(center[1] + math.sin(angle + 2 * math.pi * i / sides) * radius),
        )
        for i in range(sides)
    ]


def _draw_polygon(frame: np.ndarray, points: Sequence[tuple[int, int]], color: Sequence[int]) -> None:
    if len(points) < 2:
        return
    for start, end in zip(points, (*points[1:], points[0])):
        _draw_line(frame, start, end, color)
    cx = sum(p[0] for p in points) // len(points)
    cy = sum(p[1] for p in points) // len(points)
    _draw_circle(frame, (cx, cy), 2, color, fill=True)


def _draw_health_bar(frame: np.ndarray, center: tuple[int, int], radius: int, entity: Mapping[str, Any]) -> None:
    health = entity.get("health") or {}
    max_health = float(health.get("maxHealth", 0.0) or 0.0)
    if max_health <= 0:
        return
    ratio = max(0.0, min(1.0, float(health.get("health", 0.0)) / max_health))
    width = max(8, int(radius * 2))
    y = center[1] - radius - 7
    x0 = center[0] - width // 2
    for x in range(x0, x0 + width):
        _draw_pixel(frame, x, y, (40, 40, 40))
        _draw_pixel(frame, x, y + 1, (40, 40, 40))
    for x in range(x0, x0 + int(width * ratio)):
        _draw_pixel(frame, x, y, (80, 255, 90))
        _draw_pixel(frame, x, y + 1, (80, 255, 90))


def _draw_arena(frame: np.ndarray, snapshot: Mapping[str, Any], *, camera: tuple[float, float], scale: float) -> None:
    arena = snapshot.get("arena") or {}
    left = float(arena.get("leftX", -1000.0))
    right = float(arena.get("rightX", 1000.0))
    top = float(arena.get("topY", -1000.0))
    bottom = float(arena.get("bottomY", 1000.0))
    width = frame.shape[1]
    height = frame.shape[0]
    corners = [
        _world_to_screen(left, top, camera=camera, scale=scale, width=width, height=height),
        _world_to_screen(right, top, camera=camera, scale=scale, width=width, height=height),
        _world_to_screen(right, bottom, camera=camera, scale=scale, width=width, height=height),
        _world_to_screen(left, bottom, camera=camera, scale=scale, width=width, height=height),
    ]
    for start, end in zip(corners, (*corners[1:], corners[0])):
        _draw_line(frame, start, end, (90, 95, 110))


def _entity_color(entity: Mapping[str, Any], followed_agent_id: int) -> tuple[int, int, int]:
    kind = str(entity.get("kind", ""))
    if kind == "agent":
        agent_index = int(entity.get("agentIndex", entity.get("id", 0)))
        if agent_index == followed_agent_id:
            return (255, 255, 255)
        return _AGENT_COLORS[agent_index % len(_AGENT_COLORS)]
    return _COLOR_BY_KIND.get(kind, (180, 180, 190))


def render_world_frame(
    snapshot: Mapping[str, Any],
    *,
    followed_agent_id: int = 1,
    frame_size: tuple[int, int] = DEFAULT_FRAME_SIZE,
    view_world_size: float = DEFAULT_VIEW_WORLD_SIZE,
) -> np.ndarray:
    """Render a direct gameplay view centered on ``followed_agent_id``."""
    width, height = int(frame_size[0]), int(frame_size[1])
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (18, 20, 28)
    followed = _agent_entity(snapshot, followed_agent_id)
    camera = _position(followed) if followed is not None else (0.0, 0.0)
    scale = min(width, height) / max(float(view_world_size), 1.0)
    _draw_arena(frame, snapshot, camera=camera, scale=scale)

    entities = list(snapshot.get("entities", ()) or ())
    # Draw smaller/background objects first, then agents/projectiles on top.
    entities.sort(key=lambda e: (str(e.get("kind")) == "agent", str(e.get("kind")) == "projectile", _size(e)))
    for entity in entities:
        lifecycle = entity.get("lifecycle") or {}
        if lifecycle.get("removed"):
            continue
        x, y = _position(entity)
        center = _world_to_screen(x, y, camera=camera, scale=scale, width=width, height=height)
        radius = max(2, int(_size(entity) * scale))
        if center[0] < -radius or center[0] > width + radius or center[1] < -radius or center[1] > height + radius:
            continue
        color = _entity_color(entity, followed_agent_id)
        sides = _sides(entity)
        if sides <= 1:
            _draw_circle(frame, center, radius, color, fill=True)
        else:
            _draw_polygon(frame, _polygon_points(center, radius, sides, _angle(entity)), color)
        if entity.get("kind") == "agent":
            # Barrel/heading line makes this feel like the actual tank camera, not an obs grid.
            end = (int(center[0] + math.cos(_angle(entity)) * radius * 1.7), int(center[1] + math.sin(_angle(entity)) * radius * 1.7))
            _draw_line(frame, center, end, (230, 230, 230))
            _draw_health_bar(frame, center, radius, entity)
            label = f'A{int(entity.get("agentIndex", entity.get("id", 0)))}'
            draw_text(frame, center[0] - radius, center[1] + radius + 4, label, color=(255, 255, 255), scale=2)
    draw_text(frame, 8, height - 18, f'CAM A{followed_agent_id}', color=(255, 255, 255), scale=2)
    return frame


__all__ = ["DEFAULT_FRAME_SIZE", "DEFAULT_VIEW_WORLD_SIZE", "render_world_frame"]
