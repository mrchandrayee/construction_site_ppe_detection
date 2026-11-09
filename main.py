import os
import argparse
import cv2
from ultralytics import YOLO

# Modern tech-HUD color palette for PPE classes (BGR format)
CLASS_COLORS = {
    "helmet": (255, 255, 0),   # Cyan
    "vest": (0, 255, 127),     # Bright lime
    "gloves": (0, 165, 255),   # Neon orange
    "boots": (255, 0, 255),    # Magenta
    "goggles": (255, 192, 0),  # Sky blue
}
DEFAULT_COLOR = (0, 255, 255)


def draw_tech_corner_bbox(img, x1, y1, x2, y2, label_name,
                          color=None, thickness=3, corner_ratio=0.18):
    """Draw a thin outline, four thick L-shaped corners and a class badge."""
    if color is None:
        color = CLASS_COLORS.get(label_name.lower(), DEFAULT_COLOR)

    w, h = x2 - x1, y2 - y1
    img_h, img_w = img.shape[:2]
    res_scale = max(1.0, min(img_w, img_h) / 720.0)

    thick_line = max(3, int(thickness * res_scale))
    thin_line = max(1, thick_line // 3)
    corner_len = max(12, min(int(min(w, h) * corner_ratio), int(45 * res_scale)))

    # Thin connecting bounding-box outline
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thin_line, cv2.LINE_AA)

    # Four thick L-shaped corner brackets
    # Top-left
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thick_line, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thick_line, cv2.LINE_AA)
    # Top-right
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thick_line, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thick_line, cv2.LINE_AA)
    # Bottom-left
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thick_line, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thick_line, cv2.LINE_AA)
    # Bottom-right
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thick_line, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thick_line, cv2.LINE_AA)

    # Label badge with class-colored background and high-contrast text
    if label_name:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65 * res_scale
        font_thick = max(2, int(2 * res_scale))
        text = str(label_name).upper()
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thick)
        pad_x = int(8 * res_scale)
        pad_y = int(6 * res_scale)

        badge_y1 = max(0, y1 - text_h - pad_y * 2)
        badge_y2 = badge_y1 + text_h + pad_y * 2
        badge_x2 = min(img_w - 1, x1 + text_w + pad_x * 2)

        cv2.rectangle(img, (x1, badge_y1), (badge_x2, badge_y2), color, cv2.FILLED)

        # Determine high-contrast text color from background luminance
        b, g, r = color
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)
        cv2.putText(img, text, (x1 + pad_x, badge_y2 - pad_y - 2),
                    font, font_scale, text_color, font_thick, cv2.LINE_AA)
    return img


def main():
    parser = argparse.ArgumentParser(
        description="Process construction-site video with a PPE detection model."
    )
    parser.add_argument(
        "--weights", type=str,
        default="runs/detect/yolo26s_ppe-2/weights/best.pt",
        help="Path to trained model weights (.pt file)"
    )
    parser.add_argument(
        "--video", type=str, required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output", type=str, default="ppe_video1.mp4",
        help="Path to output video file"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="",
                        help="Inference device: cpu, 0, 0,1; empty=auto")
    args = parser.parse_args()

    # Check weights file, with fallback paths visible in the recording.
    weights_path = args.weights
    if not os.path.exists(weights_path):
        candidates = [
            "runs/detect/yolo26s_ppe/weights/best.pt",
            "yolo26s.pt",
        ]
        weights_path = next((p for p in candidates if os.path.exists(p)), None)
        if weights_path is None:
            raise FileNotFoundError(
                "No valid model weights found. Supply --weights path/to/best.pt"
            )

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Input video '{args.video}' not found.")

    print(f"Loading model weights: {weights_path}")
    model = YOLO(weights_path)

    # Filter target PPE equipment classes by model label names.
    target_names = ["helmet", "gloves", "vest", "boots", "goggles"]
    target_cls_ids = [
        cls_id for cls_id, name in model.names.items()
        if name.lower() in target_names
    ]
    print(f"Target classes to detect & draw: {target_names} (IDs: {target_cls_ids})")
    if not target_cls_ids:
        raise RuntimeError(
            f"None of {target_names} found in model.names={model.names}. "
            "Use weights trained with these class labels."
        )

    # Open video capture
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {args.video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:  # handle invalid FPS
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input Video: {args.video} | Size: {width}x{height} | "
          f"FPS: {fps:.2f} | Total Frames: {total_frames}")

    # Setup VideoWriter
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {args.output}")

    frame_count = 0
    print("Processing video frames with tech-corner bounding boxes & high-contrast labels...")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Run inference, filtering only target classes
            predict_kwargs = {
                "source": frame,
                "conf": args.conf,
                "classes": target_cls_ids,
                "verbose": False,
                "imgsz": args.imgsz,
            }
            if args.device:
                predict_kwargs["device"] = args.device
            results = model.predict(**predict_kwargs)

            # Custom tech drawing
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cls_id = int(box.cls[0].item())
                    cls_name = model.names[cls_id]
                    draw_tech_corner_bbox(
                        frame, x1, y1, x2, y2, cls_name,
                        color=CLASS_COLORS.get(cls_name.lower(), DEFAULT_COLOR)
                    )

            # Write frame to output video
            out.write(frame)

            if frame_count % 30 == 0 or frame_count == total_frames:
                progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")

    finally:
        cap.release()
        out.release()

    print("\nProcessing complete.")
    print(f"Processed frames: {frame_count}")
    print(f"Output video: {args.output}")


if __name__ == "__main__":
    main()
