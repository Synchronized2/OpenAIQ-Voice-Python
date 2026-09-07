from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

from main import NextQuestionListener


class InputInterruptTests(unittest.TestCase):
    def test_typed_question_interrupts_and_is_preserved(self) -> None:
        keys = iter("next question\r")
        stopped = threading.Event()
        listener = NextQuestionListener(stopped.set)

        with (
            patch("main.msvcrt.kbhit", return_value=True),
            patch("main.msvcrt.getwch", side_effect=lambda: next(keys)),
            patch("builtins.print"),
        ):
            listener.start()
            self.assertTrue(listener.interrupted.wait(timeout=1))
            listener.close()

        self.assertTrue(stopped.is_set())
        self.assertEqual(listener.question, "next question")


if __name__ == "__main__":
    unittest.main()
