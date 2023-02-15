import cv2
from utils.mediapipe_utils import draw_landmarks


WHITE_COLOR = (245, 242, 226)
RED_COLOR = (25, 35, 240)

HEIGHT = 600


class WebcamManager(object):
    """Object that displays image output, draws landmarks and
    predicted signs
    """

    def __init__(self):
        self.sign_detected = ""

    def update(self, frame, results, sign_detected, is_recording):
        self.sign_detected = sign_detected

        # Draw landmarks
        draw_landmarks(frame, results)

        WIDTH = int(HEIGHT * len(frame[0]) / len(frame))
        # Resize frame
        frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

        # Flip the image vertically for mirror effect
        frame = cv2.flip(frame, 1)

        # Write result if there is
        frame = self.draw_text(frame)

        # Chose circle color
        color = WHITE_COLOR
        if is_recording:
            color = RED_COLOR

        # Update the frame
        cv2.circle(frame, (30, 30), 20, color, -1)
        cv2.imshow("OpenCV Feed", frame)

    def draw_text(
        self,
        frame,
        font=cv2.FONT_HERSHEY_COMPLEX,
        font_size=1,
        font_thickness=2,
        offset=int(HEIGHT * 0.02),
        bg_color=(245, 242, 176, 0.85),
    ):
        window_w = int(HEIGHT * len(frame[0]) / len(frame))

        (text_w, text_h), _ = cv2.getTextSize(
            self.sign_detected, font, font_size, font_thickness
        )

        text_x, text_y = int((window_w - text_w) / 2), HEIGHT - text_h - offset

        cv2.rectangle(frame, (0, text_y - offset), (window_w, HEIGHT), bg_color, -1)
        cv2.putText(
            frame,
            self.sign_detected,
            (text_x, text_y + text_h + font_size - 1),
            font,
            font_size,
            (118, 62, 37),
            font_thickness,
        )
        return frame
