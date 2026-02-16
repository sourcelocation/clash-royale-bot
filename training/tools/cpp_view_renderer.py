from __future__ import annotations

import os
from typing import Any

GRID_W = 18
GRID_H = 32
RIVER_TOP_ROW = 15
RIVER_BOTTOM_ROW = 16
BRIDGE_SIDE_MARGIN = 2.5
BRIDGE_WIDTH = 2.0

CARD_NAMES = {
    0: "IceSpirit",
    1: "Musketeer",
    2: "Cannon",
    3: "Hog",
    4: "Skeletons",
    5: "Fireball",
    6: "IceGolem",
    7: "Log",
}

CARD_LABELS = {
    0: "ISpirit",
    1: "Musk",
    2: "Cannon",
    3: "Hog",
    4: "Skels",
    5: "Fireball",
    6: "IGolem",
    7: "Log",
}

CARD_COLORS = {
    0: (120, 210, 255),
    1: (255, 165, 120),
    2: (170, 170, 170),
    3: (165, 110, 70),
    4: (235, 235, 235),
    5: (255, 135, 80),
    6: (160, 210, 255),
    7: (220, 170, 90),
}

FIREBALL_RADIUS_TILES = 2.5
LOG_RADIUS_TILES = 1.0
LOG_LENGTH_TILES = 5.0


def kv(key: str, value: str, width: int = 14) -> str:
    return f"{key:<{width}} {value}"


def _clamp_rgb(v: float) -> int:
    return max(0, min(255, int(round(v))))


def _team_tint(color: tuple[int, int, int], team: int) -> tuple[int, int, int]:
    r, g, b = color
    if team == 0:
        return (_clamp_rgb(r * 0.90), _clamp_rgb(g * 1.05), _clamp_rgb(b * 1.20))
    return (_clamp_rgb(r * 1.20), _clamp_rgb(g * 0.90), _clamp_rgb(b * 0.90))


def _entity_color(entity: dict[str, Any]) -> tuple[int, int, int]:
    kind = str(entity.get("kind", "unknown"))
    team = int(entity.get("team", 0))
    card_id = int(entity.get("card_id", -1))

    if kind == "tower":
        base = (245, 220, 95)
    elif card_id in CARD_COLORS:
        base = CARD_COLORS[card_id]
    elif kind == "building":
        base = (160, 190, 155)
    elif kind == "troop":
        base = (170, 200, 230)
    else:
        base = (180, 180, 180)
    return _team_tint(base, team)


def _entity_label(entity: dict[str, Any]) -> str:
    kind = str(entity.get("kind", "unknown"))
    card_id = int(entity.get("card_id", -1))
    eid = int(entity.get("id", -1))
    if kind == "tower":
        return f"Tower#{eid}"
    name = CARD_LABELS.get(card_id, CARD_NAMES.get(card_id, "Entity"))
    return f"{name}#{eid}"


def _dim_color(color: tuple[int, int, int], factor: float = 0.35) -> tuple[int, int, int]:
    r, g, b = color
    return (_clamp_rgb(r * factor), _clamp_rgb(g * factor), _clamp_rgb(b * factor))


class ArenaRenderer:
    def __init__(
        self,
        *,
        title: str,
        width: int = 1080,
        height: int = 960,
        headless: bool = False,
    ) -> None:
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        import pygame

        self.pygame = pygame
        self.headless = bool(headless)
        pygame.init()
        flags = pygame.RESIZABLE if not self.headless else 0
        self.screen = pygame.display.set_mode((int(width), int(height)), flags)
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 16)
        self.label_font = pygame.font.SysFont("monospace", 12)
        self.closed = False
        self.last_viewport: tuple[int, int, int, int] = (0, 0, 1, 1)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.pygame.quit()

    def poll_events(self, *, esc_quit: bool = True) -> dict[str, bool]:
        out = {
            "quit": False,
            "toggle_pause": False,
            "reset": False,
            "step_once": False,
            "esc": False,
            "left": False,
            "right": False,
            "up": False,
            "down": False,
            "enter": False,
            "speed_cycle": False,
            "key_downs": [],
            "mouse_left_clicks": [],
        }
        if self.closed:
            out["quit"] = True
            return out
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                out["quit"] = True
            elif event.type == self.pygame.VIDEORESIZE and not self.headless:
                self.screen = self.pygame.display.set_mode((event.w, event.h), self.pygame.RESIZABLE)
            elif event.type == self.pygame.KEYDOWN:
                if event.key == self.pygame.K_ESCAPE:
                    if esc_quit:
                        out["quit"] = True
                    else:
                        out["esc"] = True
                elif event.key == self.pygame.K_SPACE:
                    out["toggle_pause"] = True
                elif event.key == self.pygame.K_r:
                    out["reset"] = True
                elif event.key == self.pygame.K_n:
                    out["step_once"] = True
                elif event.key == self.pygame.K_LEFT:
                    out["left"] = True
                elif event.key == self.pygame.K_RIGHT:
                    out["right"] = True
                elif event.key == self.pygame.K_UP:
                    out["up"] = True
                elif event.key == self.pygame.K_DOWN:
                    out["down"] = True
                elif event.key in (self.pygame.K_RETURN, self.pygame.K_KP_ENTER):
                    out["enter"] = True
                elif event.key == self.pygame.K_t:
                    out["speed_cycle"] = True
                out["key_downs"].append(int(event.key))
            elif event.type == self.pygame.MOUSEBUTTONDOWN and event.button == 1:
                out["mouse_left_clicks"].append((int(event.pos[0]), int(event.pos[1])))
        if out["quit"]:
            self.close()
        return out

    def draw(
        self,
        state: dict[str, Any],
        hud_lines: list[str],
        *,
        fps_limit: int,
        hud_grid: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        if self.closed:
            return {"fps": 0.0, "arena_w": 1.0, "arena_h": 1.0, "aspect": 1.0}

        self.screen.fill((17, 21, 27))
        width, height = self.screen.get_size()
        hud_rect, game_rect = self._layout_panels(width, height)
        hx, hy, hw, hh = hud_rect
        gx, gy, gw, gh = game_rect

        self.pygame.draw.rect(self.screen, (24, 29, 37), self.pygame.Rect(hx, hy, hw, hh))
        self.pygame.draw.line(self.screen, (48, 55, 68), (hx + hw, hy), (hx + hw, hy + hh), 2)

        vx, vy, vw, vh = self._fit_arena_viewport(gx, gy, gw, gh)
        self.last_viewport = (vx, vy, vw, vh)
        viewport = self.last_viewport
        self.pygame.draw.rect(self.screen, (18, 22, 28), self.pygame.Rect(vx, vy, vw, vh))

        for gx in range(GRID_W + 1):
            x = vx + int(gx / GRID_W * vw)
            self.pygame.draw.line(self.screen, (40, 45, 55), (x, vy), (x, vy + vh), 1)
        for gy in range(GRID_H + 1):
            y = vy + int(gy / GRID_H * vh)
            self.pygame.draw.line(self.screen, (40, 45, 55), (vx, y), (vx + vw, y), 1)

        river_top = vy + int((RIVER_TOP_ROW / GRID_H) * vh)
        river_bottom = vy + int(((RIVER_BOTTOM_ROW + 1) / GRID_H) * vh)
        self.pygame.draw.rect(self.screen, (35, 70, 110), self.pygame.Rect(vx, river_top, vw, river_bottom - river_top))
        bridge_y0 = float(RIVER_TOP_ROW)
        bridge_y1 = float(RIVER_BOTTOM_ROW + 1)
        left_x0 = BRIDGE_SIDE_MARGIN
        left_x1 = BRIDGE_SIDE_MARGIN + BRIDGE_WIDTH
        right_x1 = GRID_W - BRIDGE_SIDE_MARGIN
        right_x0 = right_x1 - BRIDGE_WIDTH
        lx0, by0 = self._grid_to_viewport(left_x0, bridge_y0, viewport)
        lx1, by1 = self._grid_to_viewport(left_x1, bridge_y1, viewport)
        rx0, _ = self._grid_to_viewport(right_x0, bridge_y0, viewport)
        rx1, _ = self._grid_to_viewport(right_x1, bridge_y1, viewport)
        self.pygame.draw.rect(self.screen, (185, 170, 120), self.pygame.Rect(lx0, by0, max(1, lx1 - lx0), max(1, by1 - by0)))
        self.pygame.draw.rect(self.screen, (185, 170, 120), self.pygame.Rect(rx0, by0, max(1, rx1 - rx0), max(1, by1 - by0)))

        # Draw spell AOE overlays before entities so troops remain readable on top.
        self._draw_spell_effects(state=state, viewport=viewport)

        entities = state.get("entities", [])
        tile_scale = min(vw / GRID_W, vh / GRID_H)
        for e in entities:
            x, y = self._world_to_viewport(float(e["x"]), float(e["y"]), viewport)
            radius = max(3, int(float(e["radius"]) * min(vw / GRID_W, vh / GRID_H)))
            alive = bool(e["alive"])
            kind = str(e["kind"])
            color = _entity_color(e)
            if not alive:
                color = (90, 90, 90)

            if kind == "troop":
                self.pygame.draw.circle(self.screen, color, (x, y), radius)
            elif kind == "building":
                self.pygame.draw.rect(self.screen, color, self.pygame.Rect(x - radius, y - radius, radius * 2, radius * 2))
            elif kind == "tower":
                half_side = max(4, int(1.0 * tile_scale))
                self.pygame.draw.rect(self.screen, color, self.pygame.Rect(x - half_side, y - half_side, half_side * 2, half_side * 2))
            else:
                self.pygame.draw.rect(
                    self.screen,
                    color,
                    self.pygame.Rect(x - radius - 2, y - radius - 2, (radius + 2) * 2, (radius + 2) * 2),
                    2,
                )

            hp = float(e["hp"])
            max_hp = max(1e-6, float(e["max_hp"]))
            hp_frac = max(0.0, min(1.0, hp / max_hp))
            bar_w = max(8, radius * 2)
            self.pygame.draw.rect(self.screen, (35, 35, 35), self.pygame.Rect(x - bar_w // 2, y - radius - 8, bar_w, 3))
            self.pygame.draw.rect(self.screen, (60, 220, 100), self.pygame.Rect(x - bar_w // 2, y - radius - 8, int(bar_w * hp_frac), 3))

            label_text = _entity_label(e)
            label_surf = self.label_font.render(label_text, True, (240, 242, 246))
            label_rect = label_surf.get_rect(center=(x, y - radius - 16))
            bg_rect = label_rect.inflate(4, 2)
            self.pygame.draw.rect(self.screen, (15, 18, 24), bg_rect, border_radius=3)
            self.screen.blit(label_surf, label_rect)

        y = hy + 10
        for line in hud_lines:
            if line == "":
                y += 6
                continue
            color = (235, 240, 245)
            if line.startswith("[") and line.endswith("]"):
                color = (166, 210, 255)
            surf = self.font.render(line, True, color)
            self.screen.blit(surf, (hx + 10, y))
            y += 18
        if hud_grid is not None:
            self._draw_hud_grid(hx=hx, hy=hy, hw=hw, hh=hh, start_y=y + 4, grid=hud_grid)

        self.pygame.display.flip()
        self.clock.tick(max(1, int(fps_limit)))
        return {
            "fps": float(self.clock.get_fps()),
            "arena_w": float(vw),
            "arena_h": float(vh),
            "aspect": float(vw / max(1, vh)),
        }

    def _draw_spell_effects(
        self,
        *,
        state: dict[str, Any],
        viewport: tuple[int, int, int, int],
    ) -> None:
        vx, vy, vw, vh = viewport
        if vw <= 0 or vh <= 0:
            return
        tile_scale = min(vw / GRID_W, vh / GRID_H)
        overlay = self.pygame.Surface((vw, vh), self.pygame.SRCALPHA)

        fireballs = state.get("fireballs", [])
        for f in fireballs:
            fx = float(f.get("x", 0.0))
            fy = float(f.get("y", 0.0))
            team = int(f.get("team", 0))
            px, py = self._world_to_viewport(fx, fy, viewport)
            lx = px - vx
            ly = py - vy
            radius_px = max(4, int(FIREBALL_RADIUS_TILES * tile_scale))
            if team == 0:
                fill = (255, 125, 80, 72)
                edge = (255, 165, 120, 180)
            else:
                fill = (255, 95, 95, 72)
                edge = (255, 145, 145, 180)
            self.pygame.draw.circle(overlay, fill, (lx, ly), radius_px)
            self.pygame.draw.circle(overlay, edge, (lx, ly), radius_px, 2)

        logs = state.get("logs", [])
        for s in logs:
            sx = float(s.get("x", 0.0))
            sy = float(s.get("y", 0.0))
            team = int(s.get("team", 0))
            dir_y = float(s.get("dir_y", 0.0))
            half_len = 0.5 * LOG_LENGTH_TILES
            ax = sx - half_len
            bx = sx + half_len
            px0, py0 = self._world_to_viewport(ax, sy, viewport)
            px1, py1 = self._world_to_viewport(bx, sy, viewport)
            lx0 = px0 - vx
            ly0 = py0 - vy
            lx1 = px1 - vx
            ly1 = py1 - vy
            half_w_px = max(2, int(LOG_RADIUS_TILES * tile_scale))
            left = int(min(lx0, lx1))
            right = int(max(lx0, lx1))
            top = int(min(ly0, ly1) - half_w_px)
            bottom = int(max(ly0, ly1) + half_w_px)
            rect_w = max(1, right - left)
            rect_h = max(1, bottom - top)
            if team == 0:
                fill = (210, 170, 85, 82)
                edge = (235, 195, 115, 190)
            else:
                fill = (195, 125, 90, 82)
                edge = (220, 155, 120, 190)
            self.pygame.draw.rect(overlay, fill, self.pygame.Rect(left, top, rect_w, rect_h))
            self.pygame.draw.rect(overlay, edge, self.pygame.Rect(left, top, rect_w, rect_h), 2)
            cx = (left + right) // 2
            cy = (top + bottom) // 2
            dy_px = int(max(6, half_w_px * 2) * (1.0 if dir_y >= 0.0 else -1.0))
            self.pygame.draw.line(overlay, edge, (cx, cy), (cx, cy - dy_px), 2)

        self.screen.blit(overlay, (vx, vy))

    def _draw_hud_grid(self, *, hx: int, hy: int, hw: int, hh: int, start_y: int, grid: dict[str, Any]) -> None:
        cells = list(grid.get("cells", []))
        if not cells:
            return
        max_cells = 256
        hidden_cells = max(0, len(cells) - max_cells)
        shown_cells = cells[:max_cells]
        dim_mask = [bool(x) for x in list(grid.get("dim_mask", []))]
        cell_size = max(8, int(grid.get("cell_size", 12)))
        gap = max(2, int(grid.get("gap", 2)))
        requested_cols = max(1, int(grid.get("cols", 1)))
        max_cols_by_count = max(1, len(shown_cells))
        inner_w = max(1, hw - 20)
        max_cols_by_width = max(1, (inner_w + gap) // (cell_size + gap))
        cols = max(1, min(requested_cols, max_cols_by_count, max_cols_by_width))
        rows = (len(shown_cells) + cols - 1) // cols
        title = str(grid.get("title", ""))
        legend = list(grid.get("legend", []))
        palette = dict(grid.get("palette", {}))

        x0 = hx + 10
        y = start_y
        if title:
            surf = self.font.render(title, True, (166, 210, 255))
            self.screen.blit(surf, (x0, y))
            y += 18

        for row in range(rows):
            for col in range(cols):
                idx = row * cols + col
                if idx >= len(shown_cells):
                    break
                code = int(shown_cells[idx])
                color = palette.get(code, (120, 120, 120))
                if idx < len(dim_mask) and dim_mask[idx]:
                    color = _dim_color(color)
                rx = x0 + col * (cell_size + gap)
                ry = y + row * (cell_size + gap)
                self.pygame.draw.rect(self.screen, color, self.pygame.Rect(rx, ry, cell_size, cell_size))
                self.pygame.draw.rect(self.screen, (30, 34, 42), self.pygame.Rect(rx, ry, cell_size, cell_size), 1)
        y += rows * (cell_size + gap) + 8
        if hidden_cells > 0:
            surf = self.label_font.render(f"and {hidden_cells} more...", True, (200, 205, 214))
            self.screen.blit(surf, (x0, y))
            y += 16

        for item in legend:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", ""))
            color = tuple(item.get("color", (120, 120, 120)))
            if len(color) != 3:
                color = (120, 120, 120)
            self.pygame.draw.rect(self.screen, color, self.pygame.Rect(x0, y + 2, 12, 12))
            self.pygame.draw.rect(self.screen, (30, 34, 42), self.pygame.Rect(x0, y + 2, 12, 12), 1)
            surf = self.label_font.render(label, True, (235, 240, 245))
            self.screen.blit(surf, (x0 + 18, y))
            y += 16

    def _world_to_screen(self, x: float, y: float, width: int, height: int) -> tuple[int, int]:
        px = int((x + GRID_W / 2.0) / GRID_W * width)
        py = int((1.0 - (y + GRID_H / 2.0) / GRID_H) * height)
        return px, py

    def _fit_arena_viewport(self, container_x: int, container_y: int, container_w: int, container_h: int) -> tuple[int, int, int, int]:
        target_ratio = GRID_W / GRID_H
        if container_h <= 0 or container_w <= 0:
            return 0, 0, 1, 1
        container_ratio = container_w / container_h
        if container_ratio > target_ratio:
            vh = container_h
            vw = int(vh * target_ratio)
            vx = container_x + (container_w - vw) // 2
            vy = container_y
        else:
            vw = container_w
            vh = int(vw / target_ratio)
            vx = container_x
            vy = container_y + (container_h - vh) // 2
        return vx, vy, max(1, vw), max(1, vh)

    def _world_to_viewport(self, x: float, y: float, viewport: tuple[int, int, int, int]) -> tuple[int, int]:
        vx, vy, vw, vh = viewport
        px, py = self._world_to_screen(x, y, vw, vh)
        return vx + px, vy + py

    def _grid_to_viewport(self, xg: float, yg: float, viewport: tuple[int, int, int, int]) -> tuple[int, int]:
        vx, vy, vw, vh = viewport
        px = vx + int((xg / GRID_W) * vw)
        py = vy + int((yg / GRID_H) * vh)
        return px, py

    def _layout_panels(self, window_w: int, window_h: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
        if window_w <= 0 or window_h <= 0:
            return (0, 0, 1, 1), (0, 0, 1, 1)
        hud_w = int(window_w * 0.34)
        hud_w = max(320, hud_w)
        hud_w = min(hud_w, max(180, window_w - 220))
        hud_rect = (0, 0, hud_w, window_h)
        game_rect = (hud_w, 0, max(1, window_w - hud_w), window_h)
        return hud_rect, game_rect
