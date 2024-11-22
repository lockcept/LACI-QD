import tkinter as tk
from Board import Board


class GUIQuoridor:
    def __init__(self, game):
        self.game = game
        self.board_size = game.n
        self.cell_size = 60
        self.wall_thickness = 10
        self.margin = 10
        self.selected_position = None  # 클릭된 위치 저장
        self.root = tk.Tk()
        self.root.title("Quoridor Game")
        self.canvas = tk.Canvas(
            self.root,
            width=self.margin * 2 + self.cell_size * self.board_size,
            height=self.margin * 2 + self.cell_size * self.board_size,
            bg="white",
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)  # 좌클릭 이벤트 바인딩

    def draw_board(self, board: Board):
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
            x_start = self.margin + y * self.cell_size
            y_start = self.margin + (x + 1) * self.cell_size - self.wall_thickness // 2
            x_end = x_start + self.cell_size * 2
            y_end = y_start + self.wall_thickness
            self.canvas.create_rectangle(
                x_start, y_start, x_end, y_end, fill="brown", outline="brown"
            )

        for wall in board.v_walls:  # 수직 벽
            x, y = wall
            x_start = self.margin + (y + 1) * self.cell_size - self.wall_thickness // 2
            y_start = self.margin + x * self.cell_size
            x_end = x_start + self.wall_thickness
            y_end = y_start + self.cell_size * 2  # 벽 길이 2
            self.canvas.create_rectangle(
                x_start, y_start, x_end, y_end, fill="brown", outline="brown"
            )

        # 플레이어 말 그리기
        player1_pos, player2_pos = board.p1_pos, board.p2_pos
        self.draw_piece(player1_pos, "red")
        self.draw_piece(player2_pos, "blue")

        self.root.update()

    def draw_piece(self, position, color):
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

    def on_click(self, event):
        x = (event.y - self.margin) // self.cell_size
        y = (event.x - self.margin) // self.cell_size

        click_x = event.x - self.margin
        click_y = event.y - self.margin

        x_remainder = click_y % self.cell_size
        y_remainder = click_x % self.cell_size

        # 임계값 (벽 클릭으로 간주할 거리)
        threshold_upper = self.wall_thickness * 2
        threshold_lower = self.cell_size - self.wall_thickness * 2

        # 수평 벽 검사
        if x_remainder < threshold_upper:
            wall_x = x - 1
            wall_y = y
            if 0 <= wall_x < self.board_size - 1 and 0 <= wall_y < self.board_size:
                self.selected_position = ("h_wall", wall_x, wall_y)
                return
        elif x_remainder > threshold_lower:
            wall_x = x
            wall_y = y
            if 0 <= wall_x < self.board_size - 1 and 0 <= wall_y < self.board_size:
                self.selected_position = ("h_wall", wall_x, wall_y)
                return

        # 수직 벽 검사
        if y_remainder < threshold_upper:
            wall_x = x
            wall_y = y - 1
            if 0 <= wall_x < self.board_size and 0 <= wall_y < self.board_size - 1:
                self.selected_position = ("v_wall", wall_x, wall_y)
                return
        elif y_remainder > threshold_lower:
            wall_x = x
            wall_y = y
            if 0 <= wall_x < self.board_size and 0 <= wall_y < self.board_size - 1:
                self.selected_position = ("v_wall", wall_x, wall_y)
                return

        if 0 <= x < self.board_size and 0 <= y < self.board_size:
            self.selected_position = ("move", x, y)
