# Design Document: SQLite Authentication & Project Isolation

This document outlines the architecture, database schema, security guardrails, and implementation plan for adding user login, project isolation, and deletion.

## Objectives
- Implement user authentication (signup/login) using SQLite and native Python library `hashlib`.
- Establish a 1-to-N relationship between users and projects.
- Prevent IDOR (Insecure Direct Object Reference) vulnerabilities, ensuring users can only see, access, or delete their own projects.
- Allow users to delete their projects (both from the database and the filesystem).

---

## 1. Database Schema
A SQLite database file will be stored at `data/database.db`.

### Users Table
```sql
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### User Projects Table
```sql
CREATE TABLE IF NOT EXISTS user_projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    project_name TEXT NOT NULL,
    config_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE(user_id, project_name)
);
```

---

## 2. Authentication & Session Flow
- **Password Hashing**: Uses `hashlib.pbkdf2_hmac` with a unique salt per user and SHA-256 (e.g., 100,000 iterations) to hash passwords securely.
- **Session State**: Streamlit `st.session_state` stores:
  - `st.session_state['user_id']` (Integer)
  - `st.session_state['username']` (String)
  - `st.session_state['active_config_path']` (String)

---

## 3. IDOR Prevention Strategy
To completely prevent IDOR:
- **Filtering**: All lists of projects retrieved in the UI are filtered strictly by `st.session_state['user_id']`.
- **Validation Guard**: Whenever a config path is loaded, parsed, or deleted, a check is run:
  ```python
  def verify_project_ownership(user_id, config_path):
      # Checks SQLite to ensure the user owns the config_path
  ```
  If this returns false, access/deletion is blocked.
- **Project Deletion**: When a project is deleted, we:
  1. Verify ownership.
  2. Delete files in `inputs/user_{user_id}/{project_name}/` and `outputs/user_{user_id}/{project_name}/` from the filesystem.
  3. Delete the record from `user_projects` SQLite table.
- **File System Isolation**: New projects will be created in isolated subdirectories:
  `inputs/user_{user_id}/{project_name}/config.json`
  `outputs/user_{user_id}/{project_name}/`

---

## 4. UI Flow (Streamlit)
- **Unauthenticated State**: Show a clean Login / Sign Up form (card style).
- **Authenticated State**: Show the main page, dashboard tabs, and sidebar with logout button and list of the user's specific projects.
- **Project Deletion UI**: Add a "Deletar Projeto" button in the sidebar (under the selected project dropdown) with a confirmation dialog to prevent accidental clicks.

