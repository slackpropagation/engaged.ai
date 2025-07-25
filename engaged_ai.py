from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QMessageBox
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import sys

class EngagedAI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("engaged.ai - Comeback Challenge")
        self.setGeometry(100, 100, 600, 400)
        self.init_ui()
        self.current_try = 1
        self.total_points = 0

    def init_ui(self):
        self.layout = QVBoxLayout()

        self.label = QLabel("What does the len() function do in Python?")
        self.label.setFont(QFont("Arial", 16))
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)

        self.buttons = []
        answers = ["Counts elements", "Adds numbers", "Creates a loop", "Prints text"]
        correct_answer = "Counts elements"

        for answer in answers:
            btn = QPushButton(answer)
            btn.setFont(QFont("Arial", 14))
            btn.clicked.connect(lambda _, a=answer: self.handle_answer(a, correct_answer))
            self.layout.addWidget(btn)
            self.buttons.append(btn)

        self.setLayout(self.layout)

    def handle_answer(self, selected, correct):
        if selected == correct:
            score = self.calculate_score()
            self.total_points += score
            self.show_reward(score)
        else:
            QMessageBox.information(self, "Oops!", f"The correct answer was: {correct}\n\n'len()' returns the number of items in a list.")
            self.current_try += 1
            self.label.setText("Let's try another one!")
            # In the future: load the next question here

    def calculate_score(self):
        if self.current_try == 1:
            return 100
        elif self.current_try == 2:
            return 85
        elif self.current_try == 3:
            return 70
        else:
            return 50

    def show_reward(self, score):
        reward_msg = QMessageBox(self)
        reward_msg.setWindowTitle("🎉 You're Back!")
        if score == 100:
            msg = "🔥 Perfect Comeback! +100 points"
        elif score == 85:
            msg = "⭐ Nice Recovery! +85 points"
        elif score == 70:
            msg = "👍 You got it! +70 points"
        else:
            msg = "🌱 Welcome back. +50 points"

        reward_msg.setText(msg)
        reward_msg.exec_()
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EngagedAI()
    window.show()
    sys.exit(app.exec_())