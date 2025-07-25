from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton
import sys
import random

class ChallengePopup(QWidget):
    def __init__(self, question, correct_answer, explanation):
        super().__init__()
        self.question = question
        self.correct_answer = correct_answer
        self.explanation = explanation
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("⚡ Re-Focus Challenge")
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()
        self.label = QLabel(self.question)
        self.feedback = QLabel("")
        self.try_button = QPushButton("I know the answer!")
        self.try_button.clicked.connect(self.check_answer)

        layout.addWidget(self.label)
        layout.addWidget(self.try_button)
        layout.addWidget(self.feedback)
        self.setLayout(layout)

    def check_answer(self):
        # Simulate success score for now
        attempt_score = random.choice([100, 85, 70, 50])
        self.feedback.setText(f"✅ Correct! +{attempt_score} pts\n{self.explanation}")
        self.try_button.setDisabled(True)

def show_challenge():
    def run():
        app = QApplication(sys.argv)
        popup = ChallengePopup(
            question="What is 5 + 3?",
            correct_answer="8",
            explanation="Because 5 and 3 together make 8."
        )
        popup.show()
        app.exec_()

    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    p = multiprocessing.Process(target=run)
    p.start()