import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import QTimer, Qt
from keras.models import load_model
import numpy as np
import pyqtgraph as pg

class PoseClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.load_model()
        self.cap = cv2.VideoCapture(0)  # 기본 카메라(인덱스 0)를 엽니다.

    def initUI(self):
        self.setWindowTitle('Pose Classifier')
        self.setGeometry(400, 400, 800, 800)
        self.setWindowIcon(QIcon('./model/image1.png'))

        # 이미지 디스플레이 레이블
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 480)
        self.image_label.setStyleSheet("border: 2px solid #A9E2F3;")

        # 그래프 생성
        self.graph = pg.PlotWidget()
        self.graph.setFixedSize(640, 200)
        self.graph.setBackground('w')

        # 분류 결과 디스플레이 레이블
        self.result_label = QLabel(self)
        self.result_label.setFont(QFont("Arial", 16))
        self.result_label.setAlignment(Qt.AlignCenter)

        # 종료 버튼
        self.close_button = QPushButton("닫기", self)
        self.close_button.setStyleSheet("background-color: #FFFFFF; color: blue; border: 2px solid #8181F7;")
        self.close_button.setFont(QFont("Arial", 12))
        self.close_button.clicked.connect(self.close)

        # 레이아웃 설정
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        main_layout.addWidget(self.graph, alignment=Qt.AlignCenter)  # 그래프 추가
        main_layout.addWidget(self.result_label, alignment=Qt.AlignCenter)
        
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        button_layout.addWidget(self.close_button)
        button_layout.addStretch(1)

        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

        # 카메라 프레임 업데이트 타이머
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)

    def load_model(self):
        # 분류 모델을 로드합니다.
        self.model = load_model("./model/keras_Model.h5", compile=False)
        self.class_names = [line.strip() for line in open("./model/labels.txt", "r", encoding="utf-8").readlines()]

    def update_frame(self):
        ret, frame = self.cap.read()  # 카메라로부터 프레임을 읽어옵니다.

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환합니다.
            frame = cv2.flip(frame, 1)  # 올바른 방향을 위해 좌우 반전합니다.

            # 분류 전에 프레임을 처리합니다. (크기 조정, 정규화 등)
            processed_frame = self.process_frame(frame)

            # 처리된 프레임을 분류합니다.
            prediction = self.model.predict(processed_frame)
            index = np.argmax(prediction)
            class_name = self.class_names[index]
            confidence_score = prediction[0][index]

            # 분류 결과를 포함하여 프레임을 표시합니다.
            self.display_frame(frame)
            self.display_result(f'자세 : {class_name} 신뢰도 : {confidence_score:.2f}')

            # 예측 값 및 그래프 정보 콘솔 출력 (디버깅)
            print(f'예측 값: {prediction}')
            print(f'클래스 이름: {class_name}, 신뢰도: {confidence_score:.2f}')

            # 그래프 업데이트
            self.update_graph(prediction)

    def update_graph(self, prediction):
        class_names = self.class_names
        confidence_scores = prediction[0]

        # 그래프 데이터 설정
        self.graph.clear()  # 그래프 초기화
        x = np.arange(len(class_names))
        bar = pg.BarGraphItem(x=x, height=confidence_scores, width=0.5, brush=(70, 130, 180, 255))
        self.graph.addItem(bar)
        ticks = [(i, class_names[i]) for i in range(len(class_names))]
        ax = self.graph.getAxis('bottom')
        ax.setTicks([ticks])

    def process_frame(self, frame):
        # 분류 전에 프레임을 처리합니다. (크기 조정, 정규화 등)
        # 처리된 프레임을 numpy 배열로 반환합니다.
        target_size = (224, 224)
        processed_frame = cv2.resize(frame, target_size)
        processed_frame = (processed_frame.astype(np.float32) / 127.5) - 1
        processed_frame = np.expand_dims(processed_frame, axis=0)  # 배치 차원 추가
        return processed_frame

    def display_frame(self, frame):
        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        qimage = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        pixmap = pixmap.scaledToWidth(640)  # 너비 조정
        self.image_label.setPixmap(pixmap)

    def display_result(self, result):
        self.result_label.setText(result)

    def closeEvent(self, event):
        self.cap.release()  # 애플리케이션을 닫을 때 카메라를 해제합니다.

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PoseClassifierApp()
    window.show()
    sys.exit(app.exec_())
