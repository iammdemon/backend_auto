from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.colors import red, blue, green, purple, orange, brown, pink, navy, maroon, teal
from math import floor
import os
from PIL import Image, UnidentifiedImageError
import cv2
import numpy as np
import pillow_heif
import uuid

# ✅ Use set to avoid duplicate failures
FAILED_IMAGES = set()

def convert_to_jpeg(original_path, root_folder):
    ext = os.path.splitext(original_path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        return original_path

    temp_dir = os.path.join(root_folder, "tmp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}.jpg")

    try:
        if ext == '.heic':
            heif_file = pillow_heif.read_heif(original_path)
            image = Image.frombytes(
                heif_file.mode, heif_file.size, heif_file.data, "raw"
            ).convert("RGB")
        else:
            image = Image.open(original_path).convert("RGB")

        image.save(temp_path, "JPEG")
        return temp_path
    except UnidentifiedImageError as e:
        print(f"⚠️ Cannot identify image: {original_path} - Error: {e}")
        FAILED_IMAGES.add(original_path)
        return None
    except Exception as e:
        import traceback
        print(f"⚠️ Error converting {original_path}: {e}")
        traceback.print_exc()
        FAILED_IMAGES.add(original_path)
        return None


def center_crop_face(image_path, root_folder):
    original_image_path = image_path
    image_path = convert_to_jpeg(original_image_path, root_folder)
    if image_path is None or not os.path.isfile(image_path):
        print(f"⚠️ Invalid path after conversion: {original_image_path}")
        FAILED_IMAGES.add(original_image_path)
        return None

    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"⚠️ OpenCV failed to read: {image_path}")
        FAILED_IMAGES.add(original_image_path)
        return None

    h, w = img.shape[:2]
    aspect_ratio = w / h

    rotated = False
    if aspect_ratio > 1.4:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        rotated = True
        h, w = img.shape[:2]

    aspect_ratio = w / h

    try:
        if aspect_ratio < 0.5:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            center_x = w // 2
            if len(faces) > 0:
                x, y, fw, fh = faces[0]
                center_x = x + fw // 2

            target_ratio = 9 / 16
            new_width = int(h * target_ratio)
            left = max(0, center_x - new_width // 2)
            right = min(w, center_x + new_width // 2)
            cropped = img[0:h, left:right]

            temp_dir = os.path.join(root_folder, "tmp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}_cropped.jpg")
            cv2.imwrite(temp_path, cropped)
            return temp_path

        elif rotated:
            temp_dir = os.path.join(root_folder, "tmp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}_rotated.jpg")
            cv2.imwrite(temp_path, img)
            return temp_path

        return image_path
    except Exception as e:
        import traceback
        print(f"⚠️ Failed during cropping: {original_image_path} — {e}")
        traceback.print_exc()
        FAILED_IMAGES.add(original_image_path)
        return None


def create_image_pdf(root_folder="temp", output_file="output/output.pdf"):
    PAGE_WIDTH, PAGE_HEIGHT = A4
    MARGIN_LEFT = 18
    MARGIN_TOP = 36
    MARGIN_BOTTOM = 43.2
    BORDER_GAP = 0
    IMAGE_WIDTH = 95
    IMAGE_HEIGHT = 143
    GAP = 5.67

    usable_width = PAGE_WIDTH - 2 * MARGIN_LEFT + GAP
    usable_height = PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM + GAP
    x_count = floor((usable_width + GAP) / (IMAGE_WIDTH + GAP))
    y_count = floor((usable_height + GAP) / (IMAGE_HEIGHT + GAP))
    images_per_page = x_count * y_count

    c = canvas.Canvas(output_file, pagesize=A4)
    current_page_image_count = 0 # Tracks images on the current page

    colors = [red, blue, green, purple, orange, brown, pink, navy, maroon, teal]
    person_color_index = 0

    print(f"Starting PDF generation for root folder: {root_folder}")
    try:
        # Assuming there's one date folder directly inside the root_folder
        date_folders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]
        if not date_folders:
            print(f"No date folders found in {root_folder}. PDF will be blank.")
            return
        
        # Take the first date folder found
        date_folder_path = os.path.join(root_folder, date_folders[0])
        print(f"Processing date folder: {date_folder_path}")

        person_folders = sorted([
            f for f in os.listdir(date_folder_path)
            if os.path.isdir(os.path.join(date_folder_path, f))
        ])
        print(f"Found person folders: {person_folders}")
    except Exception as e:
        print(f"⚠️ Error reading folders in {root_folder}: {e}")
        return

    if not person_folders:
        print(f"No person folders found in {date_folder_path}. PDF will be blank.")

    current_person_color = None
    current_person_block_tracker = []

    for person_folder in person_folders:
        person_path = os.path.join(date_folder_path, person_folder)
        print(f"Processing person folder: {person_path}")
        try:
            image_paths = []
            for dirpath, dirnames, filenames in os.walk(person_path):
                for f in filenames:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.heic')):
                        image_paths.append(os.path.join(dirpath, f))
            print(f"Found {len(image_paths)} images in {person_folder}")
            if not image_paths:
                print(f"No images found in {person_folder}. Skipping.")
                continue
        except Exception as e:
            print(f"⚠️ Skipping folder {person_folder} due to error reading images: {e}")
            continue

        # Assign color for the current person
        new_person_color = colors[person_color_index % len(colors)]
        person_color_index += 1

        # If a new person is starting and there are blocks from the previous person on the current page,
        # draw the border for the previous person's images and start a new page.
        if current_person_color is not None and new_person_color != current_person_color and current_person_block_tracker:
            draw_person_border(c, current_person_block_tracker, x_count, y_count, IMAGE_WIDTH, IMAGE_HEIGHT, GAP,
                               MARGIN_LEFT, MARGIN_TOP, PAGE_WIDTH, PAGE_HEIGHT, current_person_color, BORDER_GAP)
            current_person_block_tracker = [] # Reset for the new person

        current_person_color = new_person_color

        for img_path in image_paths:
            print(f"Processing image: {img_path}")

            # Check if a new page is needed
            if current_page_image_count > 0 and current_page_image_count % images_per_page == 0:
                draw_person_border(c, current_person_block_tracker, x_count, y_count, IMAGE_WIDTH, IMAGE_HEIGHT, GAP,
                                   MARGIN_LEFT, MARGIN_TOP, PAGE_WIDTH, PAGE_HEIGHT, current_person_color, BORDER_GAP)
                c.showPage()
                current_page_image_count = 0 # Reset for the new page
                current_person_block_tracker = [] # Reset for the new page

            processed_path = center_crop_face(img_path, root_folder)
            if not processed_path or not os.path.isfile(processed_path):
                print(f"⚠️ Failed to process or find processed image: {img_path}")
                FAILED_IMAGES.add(img_path)
                continue
            print(f"Image processed to: {processed_path}")

            col = current_page_image_count % x_count
            row = current_page_image_count // x_count
            x = MARGIN_LEFT + col * (IMAGE_WIDTH + GAP)
            y = PAGE_HEIGHT - MARGIN_TOP - (row + 1) * (IMAGE_HEIGHT + GAP) + GAP

            try:
                print(f"Drawing image {processed_path} at ({x}, {y}) with size ({IMAGE_WIDTH}, {IMAGE_HEIGHT})")
                c.drawImage(processed_path, x, y, width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
                current_person_block_tracker.append((col, row))
                current_page_image_count += 1
            except Exception as e:
                print(f"⚠️ Failed to draw image: {img_path} — {e}")
                FAILED_IMAGES.add(img_path)

        # Draw border for any remaining images of the current person on the last page
        if current_person_block_tracker:
            draw_person_border(c, current_person_block_tracker, x_count, y_count, IMAGE_WIDTH, IMAGE_HEIGHT, GAP,
                               MARGIN_LEFT, MARGIN_TOP, PAGE_WIDTH, PAGE_HEIGHT, current_person_color, BORDER_GAP)
            current_person_block_tracker = [] # Reset for the next person

    c.save()
    print(f'\n✅ PDF saved as {output_file}')

    if FAILED_IMAGES:
        print('\n⚠️ The following images failed to process or render:')
        for path in sorted(FAILED_IMAGES):
            print(f' - {path}')
    else:
        print(' All images processed and rendered successfully!')


def draw_person_border(c, blocks, x_count, y_count, img_w, img_h, gap, margin_left, margin_top, page_w, page_h, color, offset):
    if not blocks:
        return

    x0 = min(b[0] for b in blocks)
    y0 = min(b[1] for b in blocks)
    x1 = max(b[0] for b in blocks)
    y1 = max(b[1] for b in blocks)

    left = margin_left + x0 * (img_w + gap) - gap / 2
    top = page_h - margin_top - y0 * (img_h + gap) + gap / 2
    right = margin_left + (x1 + 1) * (img_w + gap) - gap / 2
    bottom = page_h - margin_top - (y1 + 1) * (img_h + gap) + gap / 2

    c.setStrokeColor(color)
    c.setLineWidth(2)
    c.rect(left, bottom, right - left, top - bottom)


if __name__ == "__main__":
    create_image_pdf()