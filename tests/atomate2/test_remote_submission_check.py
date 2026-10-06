import os
import sys
import unittest
from unittest.mock import patch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.dft.atomate2_utils import Atomate2Handler


class TestRemoteSubmissionCheck(unittest.TestCase):
    def setUp(self):
        self.output_dir = ".agents/test/remote_check_test"
        os.makedirs(self.output_dir, exist_ok=True)
        self.handler = Atomate2Handler(self.output_dir)

    def tearDown(self):
        import shutil

        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_check_sshproxy_standard_worker(self):
        """Test that check passes if worker does not require special sshproxy."""
        is_ok, msg = self.handler._check_sshproxy("standard_worker")
        self.assertTrue(is_ok)
        self.assertEqual(msg, "")

    @patch("src.utils.dft.atomate2_utils.Path.exists")
    def test_check_sshproxy_missing_key(self, mock_exists):
        """Test failure when key is missing for configured remote worker."""
        worker = os.environ.get("ATOMATE2_WORKER")
        if not worker:
            self.skipTest("ATOMATE2_WORKER not set; skipping remote worker key check")

        mock_exists.return_value = False

        is_ok, msg = self.handler._check_sshproxy(worker)
        self.assertFalse(is_ok)
        self.assertIn("SSH key not found", msg)

    @patch("src.utils.dft.atomate2_utils.Path.exists")
    def test_check_sshproxy_success(self, mock_exists):
        """Test success when key exists for configured remote worker."""
        worker = os.environ.get("ATOMATE2_WORKER")
        if not worker:
            self.skipTest("ATOMATE2_WORKER not set; skipping remote worker key check")

        mock_exists.return_value = True

        is_ok, msg = self.handler._check_sshproxy(worker)
        self.assertTrue(is_ok)
        self.assertEqual(msg, "SSHProxy appears configured.")


if __name__ == "__main__":
    unittest.main()
