import sys
import os
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QFileDialog,
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt


class ImageNavigator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Navigator")

        # Variables for image navigation
        self.image_folder = ""
        self.image_list = []
        self.current_index = 0
        self.image_status = {}  # Dictionary to store the image status (OK/Not OK)

        # UI setup
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.label)

        # Open folder dialog
        self.open_folder()

        # Show the first image
        if self.image_list:
            self.load_image()

    def open_folder(self):
        # Open folder dialog to select the folder with images
        self.image_folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if self.image_folder:
            self.image_list = [
                f
                for f in os.listdir(self.image_folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
            ]
            self.image_list.sort()  # Sort files alphabetically

    def load_image(self):
        # Load and display the current image
        if 0 <= self.current_index < len(self.image_list):
            image_path = os.path.join(self.image_folder, self.image_list[self.current_index])
            pixmap = QPixmap(image_path)
            self.label.setPixmap(
                pixmap.scaled(
                    self.label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
            self.setWindowTitle(
                f"Viewing: {self.image_list[self.current_index]} ({self.current_index + 1}/{len(self.image_list)})"
            )

    def next_image(self, event):
        if self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.load_image()

    def keyPressEvent(self, event):
        # Handle key events for navigation and marking
        if event.key() == Qt.Key_Right:  # Next image
            self.next_image(event)

        elif event.key() == Qt.Key_Left:  # Previous image
            if self.current_index > 0:
                self.current_index -= 1
                self.load_image()

        elif event.key() == Qt.Key_Y:  # Mark as OK
            current_image = self.image_list[self.current_index]
            self.image_status[current_image] = "OK"
            print(f"Marked {current_image} as OK")
            self.next_image(event)

        elif event.key() == Qt.Key_X:  # Mark as Not OK
            current_image = self.image_list[self.current_index]
            self.image_status[current_image] = "Not OK"
            print(f"Marked {current_image} as Not OK")
            self.next_image(event)

        elif event.key() == Qt.Key_Escape:  # Exit the application
            print("Image statuses:", self.image_status)
            self.close()

        elif event.key() == Qt.Key_Q:
            # dump the image status when the application is closed
            with open("image_status.txt", "w") as f:
                for image, status in self.image_status.items():
                    f.write(f"{image}: {status}\n")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageNavigator()
    window.show()

    sys.exit(app.exec_())
