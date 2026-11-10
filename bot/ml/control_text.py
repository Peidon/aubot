"""OCR and geometric label association for screenshot controls.

These are visual estimates, not DOM attributes. No text is sent to a service.
"""

import re
from dataclasses import dataclass
from statistics import mean

from PIL import Image


@dataclass(frozen=True)
class TextLine:
    text: str
    box: tuple
    confidence: float


def read_text(image, language="eng", page_segmentation=11):
    import pytesseract

    try:
        data = pytesseract.image_to_data(
            image.convert("RGB"), lang=language,
            config=f"--psm {page_segmentation}", output_type=pytesseract.Output.DICT, timeout=60,
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError("Install Tesseract OCR and put tesseract on PATH (macOS: brew install tesseract).") from exc
    words = []
    groups = {}
    for index, text in enumerate(data["text"]):
        text = text.strip()
        confidence = float(data["conf"][index]) / 100
        if confidence < 0.4 or not (any(char.isalnum() for char in text) or text in ("&", "/", "*", "+", "%")):
            continue
        x, y, width, height = (data[key][index] for key in ("left", "top", "width", "height"))
        word = TextLine(text, (x, y, x + width, y + height), confidence)
        words.append(word)
        key = tuple(data[key][index] for key in ("block_num", "par_num", "line_num"))
        groups.setdefault(key, []).append(word)

    lines = []
    for group in groups.values():
        segment = []
        for word in sorted(group, key=lambda item: item.box[0]):
            # Avoid joining labels across separate form columns.
            if segment and word.box[0] - segment[-1].box[2] > max(30, 2.5 * (word.box[3] - word.box[1])):
                lines.append(combine_text(segment))
                segment = []
            segment.append(word)
        if segment:
            lines.append(combine_text(segment))
    return words, sorted(lines, key=lambda line: (line.box[1], line.box[0]))


def combine_text(items):
    return TextLine(
        " ".join(item.text for item in items),
        (min(item.box[0] for item in items), min(item.box[1] for item in items),
         max(item.box[2] for item in items), max(item.box[3] for item in items)),
        mean(item.confidence for item in items),
    )


def inside(word, box):
    x1, y1, x2, y2 = box
    wx1, wy1, wx2, wy2 = word.box
    return x1 <= (wx1 + wx2) / 2 <= x2 and y1 <= (wy1 + wy2) / 2 <= y2


def looks_like_title(line):
    text = line.text.strip()
    return (
        1 < len(text) <= 120 and len(text.split()) <= 14
        and not text.endswith((".", "!", ","))
        and not re.match(r"^(please\b|we\b|this is\b)", text, re.IGNORECASE)
    )


def find_title(box, lines, controls, max_gap):
    x1, y1, x2, y2 = box
    candidates = []
    for line in lines:
        if not looks_like_title(line):
            continue
        lx1, ly1, lx2, ly2 = line.box
        height = max(1, ly2 - ly1)
        # Labels to the left are common in compact forms.
        if lx2 <= x1 and abs((ly1 + ly2 - y1 - y2) / 2) < max(height, (y2 - y1) / 2):
            if x1 - lx2 <= max_gap:
                candidates.append((x1 - lx2 + height, line))
            continue
        gap = y1 - ly2
        if not 0 <= gap <= max_gap:
            continue
        if lx2 < x1 - max_gap or lx1 > x2:
            continue
        # Do not reach past a preceding control in this column.
        if any(
            cy2 <= y1 and cy1 >= ly2 and min(cx2, x2) > max(cx1, x1)
            for (cx1, cy1, cx2, cy2), kind, _ in controls if kind != "Radiobox"
        ):
            continue
        # Prefer nearby labels with a similar left edge. Long descriptions
        # and sentence fragments have already been excluded above.
        candidates.append((gap + 0.2 * abs(lx1 - x1), line))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def extract_fields(detections, words, lines, max_title_gap=300):
    """Associate visible text with detections; leave missing text as None."""
    external_words = [word for word in words if not any(inside(word, box) for box, _, _ in detections)]
    # Rebuild each external line from its remaining words, so a line spanning
    # a control and its label does not leak the control value into its title.
    external_lines = []
    for line in lines:
        remaining = [word for word in external_words if inside(word, line.box)]
        if remaining:
            external_lines.append(combine_text(remaining))

    radios = [(index, box) for index, (box, kind, _) in enumerate(detections) if kind == "Radiobox"]
    option_text = {}
    for index, box in radios:
        x1, y1, x2, y2 = box
        height = max(1, y2 - y1)
        right_limit = min(
            [other[0] for other_index, other in radios
             if other_index != index and other[0] > x2
             and abs((other[1] + other[3] - y1 - y2) / 2) < height]
            or [x2 + max(300, height * 14)]
        )
        adjacent = [
            word for word in external_words
            if x2 <= word.box[0] < right_limit
            and abs((word.box[1] + word.box[3] - y1 - y2) / 2) <= height * 0.75
        ]
        if not adjacent:
            adjacent = [
                word for word in external_words if x1 - 200 <= word.box[0] and word.box[2] <= x1
                and abs((word.box[1] + word.box[3] - y1 - y2) / 2) <= height * 0.75
            ]
        option_text[index] = combine_text(sorted(adjacent, key=lambda item: item.box[0])) if adjacent else None

    title_lines = [line for line in external_lines if not any(
        option and inside(line, option.box) for option in option_text.values()
    )]
    # Connect nearby radio options, but stop at another question heading or
    # another kind of control. Options in the same group share one title.
    groups = []
    for index, box in sorted(radios, key=lambda item: (item[1][1], item[1][0])):
        matched = []
        for group in groups:
            for _, other in group:
                height = max(box[3] - box[1], other[3] - other[1])
                same_row = abs(box[1] - other[1]) < height
                same_column = abs(box[0] - other[0]) < height * 2
                gap = box[1] - other[3]
                has_heading = any(
                    other[3] <= line.box[1] < box[1] and looks_like_title(line)
                    and abs(line.box[0] - box[0]) < max_title_gap
                    for line in title_lines
                )
                blocked = any(
                    other[3] <= cb[1] < box[1] and cb[0] <= box[2] and cb[2] >= box[0]
                    for cb, kind, _ in detections if kind != "Radiobox"
                )
                own_title = find_title(box, title_lines, detections, max_title_gap)
                other_title = find_title(other, title_lines, detections, max_title_gap)
                separate_questions = own_title and other_title and own_title.box != other_title.box
                if (same_row and not separate_questions and abs(box[0] - other[0]) <= max_title_gap * 2) or (
                    same_column and 0 <= gap <= height * 6 and not has_heading and not blocked
                ):
                    matched.append(group)
                    break
        if matched:
            group = matched[0]
            group.append((index, box))
            for additional in matched[1:]:
                group.extend(additional)
                groups.remove(additional)
        else:
            groups.append([(index, box)])
    radio_titles = {}
    for group in groups:
        group_box = (
            min(box[0] for _, box in group), min(box[1] for _, box in group),
            max(box[2] for _, box in group), max(box[3] for _, box in group),
        )
        title = find_title(group_box, title_lines, detections, max_title_gap)
        for index, _ in group:
            radio_titles[index] = title

    fields = []
    for index, (box, kind, confidence) in enumerate(detections):
        title = radio_titles.get(index) if kind == "Radiobox" else find_title(
            box, title_lines, detections, max_title_gap
        )
        # Keep complete OCR lines, allowing modest detector edge errors.
        # Cropping strictly at an inaccurate model box can lose the first word.
        x1, y1, x2, y2 = box
        contained = [
            line for line in lines
            if y1 <= (line.box[1] + line.box[3]) / 2 <= y2
            and min(x2, line.box[2]) - max(x1, line.box[0]) >= min(20, (line.box[2] - line.box[0]) * 0.2)
            and line.box[0] >= x1 - (x2 - x1) * 0.25
            and line.box[2] <= x2 + (x2 - x1) * 0.25
        ]
        value = option_text.get(index) if kind == "Radiobox" else (combine_text(contained) if contained else None)
        fields.append({
            "id": index + 1, "type": kind, "box": list(box),
            "center": [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2],
            "confidence": confidence, "title": title.text if title else None,
            "value": value.text if value else None,
            "value_kind": "option_label" if kind == "Radiobox" else "visible_text",
            "title_box": list(title.box) if title else None,
            "value_box": list(value.box) if value else None,
            "title_ocr_confidence": title.confidence if title else None,
            "value_ocr_confidence": value.confidence if value else None,
        })
    return fields


def fill_missing_values(image, fields, language="eng"):
    """Retry inside blank OCR regions; borders often hide text in page OCR."""
    width, height = image.size
    for field in fields:
        if field["type"] == "Radiobox" or field["value"] is not None:
            continue
        x1, y1, x2, y2 = field["box"]
        x1, y1 = max(0, x1 + 4), max(0, y1 + 4)
        x2, y2 = min(width, x2 - 4), min(height, y2 - 4)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image.crop((x1, y1, x2, y2)).resize(
            ((x2 - x1) * 2, (y2 - y1) * 2), resample=Image.Resampling.BICUBIC,
        )
        _, lines = read_text(crop, language, page_segmentation=7)
        if not lines:
            continue
        text = combine_text(lines)
        field["value"] = text.text
        field["value_ocr_confidence"] = text.confidence
        field["value_box"] = [
            x1 + round(text.box[0] / 2), y1 + round(text.box[1] / 2),
            x1 + round(text.box[2] / 2), y1 + round(text.box[3] / 2),
        ]
