import unittest

from control_text import TextLine, extract_fields


def line(text, box):
    return TextLine(text, box, 0.95)


class ControlTextTests(unittest.TestCase):
    def fields(self, detections, text):
        return extract_fields(detections, text, text)

    def test_title_and_visible_value_are_separate(self):
        fields = self.fields(
            [((20, 60, 250, 100), "Text Input", 0.9)],
            [line("Email", (20, 25, 75, 45)), line("test@example.com", (30, 70, 190, 90))],
        )
        self.assertEqual(fields[0]["title"], "Email")
        self.assertEqual(fields[0]["value"], "test@example.com")

    def test_description_does_not_replace_select_title(self):
        fields = self.fields(
            [((20, 150, 300, 190), "Select", 0.9)],
            [line("Pronouns*", (20, 20, 120, 40)),
             line("Please choose the pronouns you use", (20, 65, 290, 85)),
             line("you.", (20, 110, 55, 130)), line("Start typing...", (30, 160, 150, 180))],
        )
        self.assertEqual(fields[0]["title"], "Pronouns*")
        self.assertEqual(fields[0]["value"], "Start typing...")
        self.assertEqual(fields[0]["value_kind"], "visible_text")

    def test_horizontal_radio_options_share_question(self):
        fields = self.fields(
            [((20, 60, 40, 80), "Radiobox", 0.9), ((160, 60, 180, 80), "Radiobox", 0.9)],
            [line("Do you agree?", (20, 20, 160, 40)),
             line("Yes", (50, 60, 85, 80)), line("No", (190, 60, 220, 80))],
        )
        self.assertEqual([field["title"] for field in fields], ["Do you agree?"] * 2)
        self.assertEqual([field["value"] for field in fields], ["Yes", "No"])
        self.assertEqual(fields[0]["value_kind"], "option_label")

    def test_vertical_radio_options_share_question(self):
        fields = self.fields(
            [((20, 60, 40, 80), "Radiobox", 0.9), ((20, 110, 40, 130), "Radiobox", 0.9)],
            [line("Employment type", (20, 20, 190, 40)),
             line("Full time", (50, 60, 130, 80)), line("Part time", (50, 110, 135, 130))],
        )
        self.assertEqual([field["title"] for field in fields], ["Employment type"] * 2)
        self.assertEqual([field["value"] for field in fields], ["Full time", "Part time"])

    def test_next_question_breaks_radio_group(self):
        fields = self.fields(
            [((20, 60, 40, 80), "Radiobox", 0.9), ((20, 150, 40, 170), "Radiobox", 0.9)],
            [line("Work rights?", (20, 20, 140, 40)), line("Yes", (50, 60, 85, 80)),
             line("Relocate?", (20, 110, 120, 130)), line("No", (50, 150, 80, 170))],
        )
        self.assertEqual([field["title"] for field in fields], ["Work rights?", "Relocate?"])

    def test_previous_control_stops_title_search(self):
        fields = self.fields(
            [((20, 60, 250, 100), "Text Input", 0.9), ((20, 150, 250, 190), "Text Input", 0.9)],
            [line("First name", (20, 20, 130, 40))],
        )
        self.assertIsNone(fields[1]["title"])
        self.assertIsNone(fields[1]["value"])

    def test_same_row_radio_questions_keep_separate_titles(self):
        fields = self.fields(
            [((20, 60, 40, 80), "Radiobox", 0.9), ((320, 60, 340, 80), "Radiobox", 0.9)],
            [line("Work rights?", (20, 20, 140, 40)), line("Relocate?", (320, 20, 430, 40)),
             line("Yes", (50, 60, 85, 80)), line("No", (350, 60, 380, 80))],
        )
        self.assertEqual([field["title"] for field in fields], ["Work rights?", "Relocate?"])

    def test_left_label(self):
        fields = self.fields(
            [((150, 60, 350, 100), "Text Input", 0.9)],
            [line("Email", (20, 70, 75, 90)), line("hello", (160, 70, 210, 90))],
        )
        self.assertEqual(fields[0]["title"], "Email")
        self.assertEqual(fields[0]["value"], "hello")

    def test_text_near_detector_edge_keeps_reading_order(self):
        fields = self.fields(
            [((100, 60, 400, 100), "Text Input", 0.9)],
            [line("Type here...", (70, 70, 170, 90))],
        )
        self.assertEqual(fields[0]["value"], "Type here...")


if __name__ == "__main__":
    unittest.main()
