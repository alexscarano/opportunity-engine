import unittest
import os
import tempfile
import sys

# Ensure scripts directory is in path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
)

from db import (
    init_db,
    create_user,
    verify_user,
    add_user_project,
    get_user_projects,
    verify_project_ownership,
    delete_user_project,
    get_user_api_key,
    update_user_api_key,
)


class TestDBHelper(unittest.TestCase):
    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp()
        init_db(self.db_path)

    def tearDown(self):
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_user_creation_and_verification(self):
        user_id = create_user("testuser", "securepass123", self.db_path)
        self.assertIsNotNone(user_id)

        # Test verification success
        verified_id = verify_user("testuser", "securepass123", self.db_path)
        self.assertEqual(user_id, verified_id)

        # Test verification failure
        self.assertIsNone(verify_user("testuser", "wrongpass", self.db_path))
        self.assertIsNone(verify_user("nonexistent", "pass", self.db_path))

    def test_project_ownership_and_idor_prevention(self):
        user1 = create_user("user1", "pass", self.db_path)
        user2 = create_user("user2", "pass", self.db_path)

        add_user_project(
            user1, "Proj A", "inputs/user_1/Proj_A/config.json", self.db_path
        )

        # Verify ownership
        self.assertTrue(
            verify_project_ownership(
                user1, "inputs/user_1/Proj_A/config.json", self.db_path
            )
        )
        self.assertFalse(
            verify_project_ownership(
                user2, "inputs/user_1/Proj_A/config.json", self.db_path
            )
        )

        # Get user projects
        projects = get_user_projects(user1, self.db_path)
        self.assertEqual(len(projects), 1)
        self.assertEqual(projects[0][0], "Proj A")

    def test_delete_project(self):
        user1 = create_user("user1", "pass", self.db_path)
        user2 = create_user("user2", "pass", self.db_path)
        add_user_project(
            user1, "Proj A", "inputs/user_1/Proj_A/config.json", self.db_path
        )

        # Verify deletion success for owner
        self.assertTrue(delete_user_project(user1, "Proj A", self.db_path))
        self.assertFalse(
            verify_project_ownership(
                user1, "inputs/user_1/Proj_A/config.json", self.db_path
            )
        )

        # Verify deletion of non-owned project does not succeed/impact others
        add_user_project(
            user2, "Proj B", "inputs/user_2/Proj_B/config.json", self.db_path
        )
        self.assertFalse(
            delete_user_project(user1, "Proj B", self.db_path)
        )  # user1 tries to delete user2's project
        self.assertTrue(
            verify_project_ownership(
                user2, "inputs/user_2/Proj_B/config.json", self.db_path
            )
        )

    def test_api_key_persistence(self):
        user = create_user("user1", "pass", self.db_path)

        # Initially None
        self.assertIsNone(get_user_api_key(user, self.db_path))

        # Update and verify
        update_user_api_key(user, "AIzaSyDummyKey", self.db_path)
        self.assertEqual(get_user_api_key(user, self.db_path), "AIzaSyDummyKey")

        # Update to empty/None and verify
        update_user_api_key(user, "", self.db_path)
        self.assertIsNone(get_user_api_key(user, self.db_path))


if __name__ == "__main__":
    unittest.main()
