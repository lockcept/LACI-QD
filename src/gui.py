"""
This module contains the GUIQuoridor class, which provides a graphical user interface for the Quoridor game.
"""

import tkinter as tk
from board import Board


class GUIQuoridor:
    """
    A graphical user interface for the Quoridor game.
    """

    def __init__(self, game):
        self.game = game
        self.board = None
        self.board_size = game.n
        self.cell_size = 60
        self.wall_thickness = 10
        self.wall_margin = 5
        self.margin = 10
        self.selected_position = None
        self.hovered_position = None
        self.cur_player = 1
        self.is_human_turn = False
        self.root = tk.Tk()
        self.root.title("Quoridor Game")

        width = self.margin * 2 + self.cell_size * self.board_size
        height = self.margin * 2 + self.cell_size * self.board_size
        self.root.geometry(f"{width}x{height}+100+100")
        self.canvas = tk.Canvas(
            self.root,
            width=self.margin * 2 + self.cell_size * self.board_size,
            height=self.margin * 2 + self.cell_size * self.board_size,
            bg="white",
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_hover)

    def draw_board(self):
        """
        Draws the game board on the canvas.
        """
        board = self.board
        self.canvas.delete("all")

        for i in range(self.board_size + 1):
            x0 = self.margin + i * self.cell_size
            y0 = self.margin
            x1 = self.margin + i * self.cell_size
            y1 = self.margin + self.board_size * self.cell_size
            self.canvas.create_line(x0, y0, x1, y1, fill="black", width=1)

            x0 = self.margin
            y0 = self.margin + i * self.cell_size
            x1 = self.margin + self.board_size * self.cell_size
            y1 = self.margin + i * self.cell_size
            self.canvas.create_line(x0, y0, x1, y1, fill="black", width=1)

        # 벽 그리기
        for wall in board.h_walls:
            x, y = wall
            x_start = self.margin + y * self.cell_size + self.wall_margin
            y_start = self.margin + (x + 1) * self.cell_size - self.wall_thickness // 2
            x_end = x_start + self.cell_size * 2 - self.wall_margin * 2
            y_end = y_start + self.wall_thickness
            self.canvas.create_rectangle(
                x_start, y_start, x_end, y_end, fill="brown", outline="brown"
            )

        for wall in board.v_walls:  # 수직 벽
            x, y = wall
            x_start = self.margin + (y + 1) * self.cell_size - self.wall_thickness // 2
            y_start = self.margin + x * self.cell_size + self.wall_margin
            x_end = x_start + self.wall_thickness
            y_end = y_start + self.cell_size * 2 - self.wall_margin * 2
            self.canvas.create_rectangle(
                x_start, y_start, x_end, y_end, fill="brown", outline="brown"
            )

        # 플레이어 말 그리기
        player1_pos, player2_pos = board.my_pos, board.enemy_pos
        self.draw_piece(player1_pos, "red")
        self.draw_piece(player2_pos, "blue")

        # 호버된 위치 하이라이트 (사람 차례일 때만)
        if self.is_human_turn and self.hovered_position:
            self.highlight_hover(self.hovered_position)

        self.root.update()

    def update_board(self, board: Board, action_probabilities=None, player=None):
        """
        Updates the game board on the canvas.
        """
        self.board = board
        self.draw_board()

        if player is not None:
            self.cur_player = player

        if action_probabilities is not None:
            action_probabilities = self.game.get_canonical_pi(
                action_probabilities, self.cur_player
            )
            self.highlight_action_probabilities(action_probabilities)

    def decode_action(self, idx: int) -> tuple:
        """
        Decodes an action index into a specific action type and its coordinates.
        """
        if idx < self.board_size * self.board_size:  # 칸 이동
            x = idx // self.board_size
            y = idx % self.board_size
            return "move", x, y
        elif idx < self.board_size * self.board_size + (self.board_size - 1) * (
            self.board_size - 1
        ):  # 수평 벽
            idx -= self.board_size * self.board_size
            x = idx // (self.board_size - 1)
            y = idx % (self.board_size - 1)
            return "h_wall", x, y
        else:
            idx -= self.board_size * self.board_size + (self.board_size - 1) * (
                self.board_size - 1
            )
            x = idx // (self.board_size - 1)
            y = idx % (self.board_size - 1)
            return "v_wall", x, y

    def highlight_action_probabilities(self, action_probabilities):
        """
        Highlights action probabilities on the canvas.
        """

        for idx, prob in enumerate(action_probabilities):
            if prob > 0:
                action_type, x, y = self.decode_action(idx)
                intensity = int(prob * 200 + 55)

                color = (
                    f"#FF{255-intensity:02x}{255-intensity:02x}"
                    if self.cur_player == 1
                    else f"#{255-intensity:02x}{255-intensity:02x}FF"
                )

                if action_type == "move":
                    self.canvas.create_rectangle(
                        self.margin + y * self.cell_size + self.wall_margin,
                        self.margin + x * self.cell_size + self.wall_margin,
                        self.margin + (y + 1) * self.cell_size - self.wall_margin,
                        self.margin + (x + 1) * self.cell_size - self.wall_margin,
                        fill=color,
                        outline="",
                    )
                elif action_type == "h_wall":
                    self.canvas.create_rectangle(
                        self.margin + y * self.cell_size + self.wall_margin,
                        self.margin
                        + (x + 1) * self.cell_size
                        - self.wall_thickness // 2,
                        self.margin + (y + 2) * self.cell_size - self.wall_margin,
                        self.margin
                        + (x + 1) * self.cell_size
                        + self.wall_thickness // 2,
                        fill=color,
                        outline="",
                    )
                elif action_type == "v_wall":
                    self.canvas.create_rectangle(
                        self.margin
                        + (y + 1) * self.cell_size
                        - self.wall_thickness // 2,
                        self.margin + x * self.cell_size + self.wall_margin,
                        self.margin
                        + (y + 1) * self.cell_size
                        + self.wall_thickness // 2,
                        self.margin + (x + 2) * self.cell_size - self.wall_margin,
                        fill=color,
                        outline="",
                    )
        self.root.update()

    def draw_piece(self, position, color):
        """
        Draws a player piece on the canvas.
        """
        x, y = position
        x_center = self.margin + y * self.cell_size + self.cell_size // 2
        y_center = self.margin + x * self.cell_size + self.cell_size // 2
        radius = self.cell_size // 3
        self.canvas.create_oval(
            x_center - radius,
            y_center - radius,
            x_center + radius,
            y_center + radius,
            fill=color,
            outline=color,
        )

    def highlight_hover(self, hovered_position):
        """
        Highlights the hovered position
        """
        action_type, x, y = hovered_position

        if action_type == "move":
            x_start = self.margin + y * self.cell_size + self.wall_margin
            y_start = self.margin + x * self.cell_size + self.wall_margin
            x_end = x_start + self.cell_size - self.wall_margin * 2
            y_end = y_start + self.cell_size - self.wall_margin * 2
            self.canvas.create_rectangle(
                x_start,
                y_start,
                x_end,
                y_end,
                fill="",
                outline="blue",
                width=2,
                dash=(4, 2),
            )
        elif action_type == "h_wall":
            x_start = self.margin + y * self.cell_size + self.wall_margin
            y_start = self.margin + (x + 1) * self.cell_size - self.wall_thickness // 2
            x_end = x_start + self.cell_size * 2 - self.wall_margin * 2
            y_end = y_start + self.wall_thickness
            self.canvas.create_rectangle(
                x_start,
                y_start,
                x_end,
                y_end,
                fill="",
                outline="blue",
                width=2,
                dash=(4, 2),
            )
        elif action_type == "v_wall":
            x_start = self.margin + (y + 1) * self.cell_size - self.wall_thickness // 2
            y_start = self.margin + x * self.cell_size + self.wall_margin
            x_end = x_start + self.wall_thickness
            y_end = y_start + self.cell_size * 2 - self.wall_margin * 2
            self.canvas.create_rectangle(
                x_start,
                y_start,
                x_end,
                y_end,
                fill="",
                outline="blue",
                width=2,
                dash=(4, 2),
            )

    def on_click(self, event):
        """
        Handles the click event on the canvas.
        """
        if not self.is_human_turn:
            return
        self.hovered_position = self.calculate_position(event)
        self.selected_position = self.hovered_position

    def on_hover(self, event):
        """
        Handles the hover event on the canvas.
        """
        if not self.is_human_turn:
            return
        self.hovered_position = self.calculate_position(event)
        self.draw_board()

    def calculate_position(self, event):
        """
        Calculates the position of the click event.
        """
        x = (event.y - self.margin) // self.cell_size
        y = (event.x - self.margin) // self.cell_size

        click_x = event.x - self.margin
        click_y = event.y - self.margin

        x_remainder = click_y % self.cell_size
        y_remainder = click_x % self.cell_size

        # 임계값 (벽 클릭으로 간주할 거리)
        threshold = self.wall_thickness
        threshold_upper = threshold
        threshold_lower = self.cell_size - threshold

        # 수평 벽 검사
        if x_remainder < threshold_upper:
            wall_x = x - 1
            wall_y = y
            if 0 <= wall_x < self.board_size - 1 and 0 <= wall_y < self.board_size:
                return ("h_wall", wall_x, wall_y)
        elif x_remainder > threshold_lower:
            wall_x = x
            wall_y = y
            if 0 <= wall_x < self.board_size - 1 and 0 <= wall_y < self.board_size:
                return ("h_wall", wall_x, wall_y)

        # 수직 벽 검사
        if y_remainder < threshold_upper:
            wall_x = x
            wall_y = y - 1
            if 0 <= wall_x < self.board_size and 0 <= wall_y < self.board_size - 1:
                return ("v_wall", wall_x, wall_y)
        elif y_remainder > threshold_lower:
            wall_x = x
            wall_y = y
            if 0 <= wall_x < self.board_size and 0 <= wall_y < self.board_size - 1:
                return ("v_wall", wall_x, wall_y)

        # 칸 클릭
        if 0 <= x < self.board_size and 0 <= y < self.board_size:
            return ("move", x, y)

        return None
