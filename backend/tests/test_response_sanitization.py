import unittest

from backend.services import vector_service


class TestResponseSanitization(unittest.TestCase):
    def test_removes_emojis_and_condenses_spaces(self):
        dirty = "Here is an answer \U0001F60A with stray \U0001F680 emojis."  # noqa: E501 - fixture data
        cleaned = vector_service._sanitize_output(dirty)
        self.assertEqual(cleaned, "Here is an answer with stray emojis.")

    def test_collapses_extra_blank_lines(self):
        dirty = "Line one.\n\n\nLine two.\n\n\n\nLine three \U0001F642"
        cleaned = vector_service._sanitize_output(dirty)
        self.assertEqual(cleaned, "Line one.\n\nLine two.\n\nLine three")

    def test_sanitize_record_removes_emoji_in_fields(self):
        record = {
            "content": "Answer with sparkles \U00002728",
            "metadata": {"source": "file \U0001F4C4", "page": 3},
            "score": 0.9,
        }
        sanitized = vector_service._sanitize_record(record)
        self.assertEqual(sanitized["content"], "Answer with sparkles")
        self.assertEqual(sanitized["metadata"].get("source"), "file")
        self.assertEqual(sanitized["metadata"].get("page"), 3)


if __name__ == "__main__":
    unittest.main()
