from __future__ import annotations

import unittest

from runtime_state import RuntimeMachine, RuntimeState


class RuntimeMachineTests(unittest.TestCase):
    def test_only_one_worker_can_run(self) -> None:
        runtime = RuntimeMachine()
        self.assertTrue(runtime.start("asr", RuntimeState.LISTENING))
        self.assertFalse(runtime.start("chat", RuntimeState.PROCESSING))
        self.assertTrue(runtime.is_running("asr"))
        self.assertEqual(runtime.state, RuntimeState.LISTENING)

    def test_only_owner_can_finish_worker(self) -> None:
        runtime = RuntimeMachine()
        runtime.start("agent", RuntimeState.EXECUTING)
        self.assertFalse(runtime.finish("chat"))
        self.assertTrue(runtime.busy)
        self.assertTrue(runtime.finish("agent"))
        self.assertFalse(runtime.busy)

    def test_worker_can_transition_to_speaking(self) -> None:
        runtime = RuntimeMachine()
        runtime.start("chat", RuntimeState.PROCESSING)
        runtime.transition(RuntimeState.SPEAKING)
        self.assertEqual(runtime.state, RuntimeState.SPEAKING)
        self.assertTrue(runtime.is_running("chat"))


if __name__ == "__main__":
    unittest.main()

