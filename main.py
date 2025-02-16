import cv2
import dlib
import numpy as np
import random
import pygame
import time
import sys
import threading
from scipy.spatial import distance as dist

# ------------- CONFIGURATION -------------
DEBUG = 0
MAX_NUM = 9
CAP_WIDTH = 160
CAP_HEIGHT = 120
DEBOUNCE_TIME = 0.5

# ------------- GLOBAL STATE -------------
blink_count = 0
last_result = "Waiting for first blink..."
clicked_feedback = None
streak = 0
score = 0
awaiting_guess = False
heatmap = []
changed_x, changed_y = 0, 0
blink_detected = False
last_blink_time = 0
frame_count = 0
last_face = None
needs_redraw = True
running = True

# Lock for thread safety
state_lock = threading.Lock()

# ------------- SETUP BLINK DETECTION (dlib) -------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
LEFT_EYE_IDX = list(range(42, 48))
RIGHT_EYE_IDX = list(range(36, 42))
blink_threshold = 0.18

# ------------- VIDEO CAPTURE -------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

# ------------- HELPER FUNCTIONS -------------
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def get_streak_color(streak_val):
    if streak_val >= 10:
        return (255, 0, 0)
    else:
        return (int(255 * (streak_val / 10)), 0, 0)

def get_score_color(score_val):
    if score_val >= 20:
        return (0, 255, 0)
    else:
        return (0, int(255 * (score_val / 20)), 0)

# ------------- PYGAME SETUP -------------
pygame.init()
GRID_SIZE = 8
CELL_SIZE = 80
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE
INFO_HEIGHT = 50
screen = pygame.display.set_mode((WIDTH, HEIGHT + INFO_HEIGHT))
pygame.display.set_caption("Blink Fast")
font = pygame.font.Font(None, 50)
WHITE, BLACK = (255, 255, 255), (0, 0, 0)

grid = [[random.randint(0, MAX_NUM) for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
with state_lock:
    changed_x, changed_y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)

def draw_grid():
    """Redraws the grid and info panel, ensuring the previous selection is cleared."""
    global needs_redraw
    screen.fill(WHITE)

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            num = grid[row][col]
            color = BLACK

            if DEBUG and (row, col) == (changed_x, changed_y):
                color = (DEBUG, 0, 0)

            if clicked_feedback is not None and (row, col) == (clicked_feedback[0], clicked_feedback[1]):
                color = clicked_feedback[2]
            else:
                color = BLACK  # Ensure default color is used if clicked_feedback is cleared

            for (hx, hy, age) in heatmap:
                if (row, col) == (hx, hy):
                    fade_intensity = 255 - (age * 25)
                    color = (fade_intensity, fade_intensity, 0)

            text = font.render(str(num), True, color)
            x = col * CELL_SIZE + 30
            y = row * CELL_SIZE + 20
            screen.blit(text, (x, y))

    info_panel_rect = pygame.Rect(0, HEIGHT, WIDTH, INFO_HEIGHT)
    pygame.draw.rect(screen, WHITE, info_panel_rect)

    score_text = font.render(f"Score: {score}", True, get_score_color(score))
    streak_text = font.render(f"Streak: {streak}", True, get_streak_color(streak))
    blink_text = font.render(f"Blinks: {blink_count}", True, BLACK)

    x_offset = 10
    y_offset = HEIGHT + (INFO_HEIGHT - score_text.get_height()) // 2
    screen.blit(score_text, (x_offset, y_offset))
    x_offset += score_text.get_width() + 20
    screen.blit(streak_text, (x_offset, y_offset))
    x_offset += streak_text.get_width() + 20
    screen.blit(blink_text, (x_offset, y_offset))

    pygame.display.flip()
    needs_redraw = False

def update_status():
    sys.stdout.write(f"\rBlinks: {blink_count} | Score: {score} | Streak: {streak} | Last: {last_result}  ")
    sys.stdout.flush()

# ------------- BLINK DETECTION THREAD -------------
def blink_detection_thread():
    global frame_count, last_face, blink_detected, last_blink_time
    global blink_count, changed_x, changed_y, grid, awaiting_guess, streak, last_result, needs_redraw, heatmap, running, clicked_feedback

    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        with state_lock:
            if frame_count % 10 == 0 or last_face is None:
                faces = detector(gray)
                if faces:
                    last_face = faces[0]

            if last_face is not None:
                landmarks = predictor(gray, last_face)
                left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE_IDX])
                right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE_IDX])
                avg_EAR = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                current_time = time.time()

                if avg_EAR < blink_threshold and not blink_detected and (current_time - last_blink_time >= DEBOUNCE_TIME):
                    blink_detected = True
                    last_blink_time = current_time
                    blink_count += 1

                    clicked_feedback = None  # Ensure no lingering highlight

                    if awaiting_guess:
                        streak = 0
                        last_result = "Missed! Streak reset."

                    awaiting_guess = True
                    heatmap.clear()

                    changed_x = random.randint(0, GRID_SIZE - 1)
                    changed_y = random.randint(0, GRID_SIZE - 1)
                    grid[changed_x][changed_y] = random.randint(0, MAX_NUM)

                    needs_redraw = True

                elif avg_EAR >= blink_threshold:
                    blink_detected = False

        time.sleep(0.01)

blink_thread = threading.Thread(target=blink_detection_thread)
blink_thread.daemon = True
blink_thread.start()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            clicked_x = my // CELL_SIZE
            clicked_y = mx // CELL_SIZE

            with state_lock:
                if (clicked_x, clicked_y) == (changed_x, changed_y):
                    last_result = "Correct!"
                    streak += 1
                    score += 2 ** (streak - 1)
                    awaiting_guess = False
                    clicked_feedback = (clicked_x, clicked_y, (0, 200, 0))
                else:
                    last_result = "Wrong!"
                    streak = 0
                    score = 0
                    awaiting_guess = False
                    clicked_feedback = None

                needs_redraw = True

    if needs_redraw:
        with state_lock:
            draw_grid()
            update_status()

running = False
blink_thread.join()
cap.release()
pygame.quit()
sys.exit()
