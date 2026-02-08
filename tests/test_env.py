import os
import unittest

from core.env import require_env_var


class EnvTests(unittest.TestCase):
    def test_require_env_var_missing_raises(self) -> None:
        key = "TEST_REQUIRED_ENV"
        if key in os.environ:
            del os.environ[key]
        with self.assertRaises(RuntimeError) as ctx:
            require_env_var(key)
        self.assertIn(key, str(ctx.exception))

    def test_require_env_var_present_returns(self) -> None:
        key = "TEST_REQUIRED_ENV_PRESENT"
        os.environ[key] = "value"
        self.assertEqual(require_env_var(key), "value")


if __name__ == "__main__":
    unittest.main()
