import unittest

from tennis_rl.actions import ACTIONS, ACTION_NAMES


class ActionTests(unittest.TestCase):
    def test_action_contract(self):
        self.assertEqual(len(ACTIONS), 19)
        self.assertEqual(len(ACTION_NAMES), 19)
        self.assertEqual(ACTIONS[0], ())
        self.assertIn("a", ACTIONS[5])
        self.assertIn("b", ACTIONS[6])


if __name__ == "__main__":
    unittest.main()
