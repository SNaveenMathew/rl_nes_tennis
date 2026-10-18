import unittest
from tennis_rl.actions import ACTIONS, ACTION_NAMES, decode_action, encode_action

class ActionTests(unittest.TestCase):
    def test_action_contract(self):
        self.assertEqual(len(ACTIONS), len(ACTION_NAMES), 27)
        self.assertEqual(ACTIONS[0], ())
        self.assertEqual(ACTIONS[1], ("a",))
        self.assertEqual(ACTIONS[2], ("b",))
        self.assertEqual(encode_action(6, 2), 20)
        self.assertEqual(decode_action(20), (6, 2))
        self.assertIn("RIGHT+UP+A", ACTION_NAMES)

if __name__ == "__main__":
    unittest.main()
