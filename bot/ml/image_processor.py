import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

if __package__:
    from .control_text import extract_fields, fill_missing_values, read_text
else:
    from control_text import extract_fields, fill_missing_values, read_text

BASE_DIR = Path(__file__).resolve().parent
TARGET_CLASSES = ("Text Input", "Radiobox", "Select")


def detect_screen_fields(image, confidence, image_size, iou, model_path):
    from ultralytics import YOLO

    # Full-model checkpoints must come from a trusted source.
    if not model_path.is_file():
        raise FileNotFoundError(f"Cannot find ScreenParser checkpoint: {model_path}")
    model = YOLO(str(model_path))
    class_ids = {label: class_id for class_id, label in model.names.items()}
    missing = set(TARGET_CLASSES) - class_ids.keys()
    if missing:
        raise ValueError(f"Checkpoint is missing ScreenParser classes: {', '.join(sorted(missing))}")
    result = model.predict(
        image,
        conf=confidence,
        imgsz=image_size,
        iou=iou,
        classes=[class_ids[label] for label in TARGET_CLASSES],
        verbose=False,
    )[0]
    return [
        (
            tuple(round(value) for value in box.xyxy[0].tolist()),
            result.names[int(box.cls.item())],
            float(box.conf.item()),
        )
        for box in result.boxes
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Detect Text Input, Radiobox, and Select with ScreenParser and save a preview."
    )
    parser.add_argument("--source", type=Path, default=BASE_DIR / "screenshot" / "radio_boxes.png")
    parser.add_argument(
        "--backend", choices=("screenparser", "yolo"), default="screenparser",
        help="yolo is an alias for screenparser",
    )
    parser.add_argument("--model", type=Path, default=BASE_DIR / "best.pt", help="Local ScreenParser checkpoint")
    parser.add_argument("--conf", type=float, default=0.10, help="ScreenParser confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="ScreenParser inference size")
    parser.add_argument("--iou", type=float, default=0.10, help="ScreenParser NMS overlap threshold")
    parser.add_argument("--output", type=Path, help="Preview path; defaults to SOURCE_detections.png")
    parser.add_argument("--json-output", type=Path, help="Structured results; defaults to SOURCE_fields.json")
    parser.add_argument("--ocr-lang", default="eng", help="Installed Tesseract language, e.g. eng or eng+fra")
    parser.add_argument("--title-gap", type=int, default=300, help="Maximum distance to a title in screenshot pixels")
    args = parser.parse_args()
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        parser.error("--conf and --iou must be between 0 and 1")
    if args.imgsz <= 0:
        parser.error("--imgsz must be positive")
    if args.title_gap <= 0:
        parser.error("--title-gap must be positive")

    try:
        with Image.open(args.source) as source:
            image = source.convert("RGB")
    except OSError as exc:
        parser.error(f"Cannot read image: {args.source}: {exc}")
    output = args.output or args.source.with_name(f"{args.source.stem}_detections.png")
    json_output = args.json_output or args.source.with_name(f"{args.source.stem}_fields.json")
    protected = {args.source.resolve(), args.model.resolve()}
    if output.resolve() in protected or json_output.resolve() in protected or output.resolve() == json_output.resolve():
        parser.error("Preview and JSON paths must differ from each other, the source image, and model")
    detections = detect_screen_fields(image, args.conf, args.imgsz, args.iou, args.model)
    words, lines = read_text(image, args.ocr_lang)
    fields = extract_fields(detections, words, lines, args.title_gap)
    fill_missing_values(image, fields, args.ocr_lang)
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default(size=15)
    for field, (box, label, confidence) in zip(fields, detections):
        index = field["id"]
        x1, y1, x2, y2 = box
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        score = f", confidence={confidence:.3f}" if confidence is not None else ""
        print(f"{index}: {label}, box={box}, center={center}{score}, title={field['title']!r}, value={field['value']!r}")
        draw.rectangle(box, outline=(0, 160, 0), width=2)
        cx, cy = center
        draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(255, 0, 0))
        caption = f"{index}: {label}"
        if confidence is not None:
            caption += f" {confidence:.2f}"
        draw.text((max(0, x1), max(0, y1 - 20)), caption, fill=(0, 120, 0), font=font)

    output.parent.mkdir(parents=True, exist_ok=True)
    preview.save(output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps({
        "source": str(args.source.resolve()), "backend": args.backend,
        "image_size": {"width": image.width, "height": image.height},
        "text_association": "OCR and geometric heuristics; verify inferred titles against the image",
        "value_note": "Values are visible text or radio option labels, not DOM values. Placeholder text may be included. Radio selection state is not inferred.",
        "fields": fields,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Found {len(detections)} candidates/detections. Preview: {output}")
    print(f"Titles and visible values: {json_output}")


if __name__ == "__main__":
    main()
