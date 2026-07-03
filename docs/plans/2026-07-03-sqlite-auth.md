# SQLite Authentication & Project Isolation Implementation Plan

> **For Antigravity:** REQUIRED WORKFLOW: Use `.agent/workflows/execute-plan.md` to execute this plan in single-flow mode.

**Goal:** Implement user authentication (signup/login), project isolation, and project deletion using Python's native `sqlite3` and `hashlib` modules to prevent IDOR vulnerabilities.

**Architecture:** Create a `scripts/db.py` helper to manage user registration, verification, project ownership, and project deletion. Integrate this helper into `scripts/streamlit_app.py` to restrict project selection, configuration loading, and deletion to the authenticated user.

**Tech Stack:** Python 3.13, Streamlit, SQLite (`sqlite3` built-in), Hashlib (`hashlib` built-in).

---

### Task 1: Database Helper Implementation (including Deletion)

**Files:**
- Create: `scripts/db.py`
- Create: `tests/test_db.py`

**Step 1: Write the failing test**

```python
# tests/test_db.py
import unittest
import os
import tempfile
from scripts.db import init_db, create_user, verify_user, add_user_project, get_user_projects, verify_project_ownership, delete_user_project

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
        
        add_user_project(user1, "Proj A", "inputs/user_1/Proj_A/config.json", self.db_path)
        
        # Verify ownership
        self.assertTrue(verify_project_ownership(user1, "inputs/user_1/Proj_A/config.json", self.db_path))
        self.assertFalse(verify_project_ownership(user2, "inputs/user_1/Proj_A/config.json", self.db_path))
        
        # Get user projects
        projects = get_user_projects(user1, self.db_path)
        self.assertEqual(len(projects), 1)
        self.assertEqual(projects[0][0], "Proj A")

    def test_delete_project(self):
        user1 = create_user("user1", "pass", self.db_path)
        user2 = create_user("user2", "pass", self.db_path)
        add_user_project(user1, "Proj A", "inputs/user_1/Proj_A/config.json", self.db_path)
        
        # Verify deletion success for owner
        self.assertTrue(delete_user_project(user1, "Proj A", self.db_path))
        self.assertFalse(verify_project_ownership(user1, "inputs/user_1/Proj_A/config.json", self.db_path))
        
        # Verify deletion of non-owned project does not succeed/impact others
        add_user_project(user2, "Proj B", "inputs/user_2/Proj_B/config.json", self.db_path)
        self.assertFalse(delete_user_project(user1, "Proj B", self.db_path)) # user1 tries to delete user2's project
        self.assertTrue(verify_project_ownership(user2, "inputs/user_2/Proj_B/config.json", self.db_path))

if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_db.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.db'`

**Step 3: Write minimal implementation**

```python
# scripts/db.py
import sqlite3
import hashlib
import os
import secrets

def get_connection(db_path="data/database.db"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return sqlite3.connect(db_path)

def init_db(db_path="data/database.db"):
    with get_connection(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                project_name TEXT NOT NULL,
                config_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                UNIQUE(user_id, project_name)
            );
        """)
        conn.commit()

def hash_password(password: str, salt: str = None) -> tuple[str, str]:
    if salt is None:
        salt = secrets.token_hex(16)
    # Using 100,000 iterations PBKDF2-HMAC-SHA256
    pwd_hash = hashlib.pbkdf2_hmac(
        'sha256', 
        password.encode('utf-8'), 
        salt.encode('utf-8'), 
        100000
    ).hex()
    return pwd_hash, salt

def create_user(username: str, password: str, db_path="data/database.db") -> int:
    pwd_hash, salt = hash_password(password)
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
                (username.strip(), pwd_hash, salt)
            )
            conn.commit()
            return cursor.lastrowid
    except sqlite3.IntegrityError:
        raise ValueError(f"Username '{username}' already exists.")

def verify_user(username: str, password: str, db_path="data/database.db") -> int | None:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, password_hash, salt FROM users WHERE username = ?", (username.strip(),))
        row = cursor.fetchone()
        if not row:
            return None
        user_id, stored_hash, salt = row
        candidate_hash, _ = hash_password(password, salt)
        if secrets.compare_digest(stored_hash, candidate_hash):
            return user_id
    return None

def add_user_project(user_id: int, project_name: str, config_path: str, db_path="data/database.db"):
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO user_projects (user_id, project_name, config_path) VALUES (?, ?, ?)",
            (user_id, project_name.strip(), config_path)
        )
        conn.commit()

def get_user_projects(user_id: int, db_path="data/database.db") -> list[tuple[str, str]]:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT project_name, config_path FROM user_projects WHERE user_id = ?",
            (user_id,)
        )
        return cursor.fetchall()

def verify_project_ownership(user_id: int, config_path: str, db_path="data/database.db") -> bool:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM user_projects WHERE user_id = ? AND config_path = ?",
            (user_id, config_path)
        )
        return cursor.fetchone() is not None

def delete_user_project(user_id: int, project_name: str, db_path="data/database.db") -> bool:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM user_projects WHERE user_id = ? AND project_name = ?",
            (user_id, project_name.strip())
        )
        conn.commit()
        return cursor.rowcount > 0
```

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_db.py`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/db.py tests/test_db.py
git commit -m "feat: implement database helper with hashing, project ownership, and deletion"
```

---

### Task 2: Streamlit App Integration & Login Interface

**Files:**
- Modify: `scripts/streamlit_app.py`

**Step 1: Import DB helper and initialize DB**

In `scripts/streamlit_app.py` at the imports section:
```python
from db import init_db, create_user, verify_user, add_user_project, get_user_projects, verify_project_ownership, delete_user_project
init_db()  # Initializes the default SQLite DB at data/database.db
```

**Step 2: Add Login and Registration UI**

Wrap the main UI rendering code so that if a user is not authenticated, a Login / Registration form is displayed.

At the beginning of page rendering:
```python
# Authentication check
if 'user_id' not in st.session_state:
    st.subheader("Login / Cadastro")
    auth_mode = st.radio("Selecione a ação", ["Login", "Cadastrar Novo Usuário"])
    username = st.text_input("Usuário")
    password = st.text_input("Senha", type="password")
    
    if auth_mode == "Login":
        if st.button("Entrar"):
            user_id = verify_user(username, password)
            if user_id:
                st.session_state['user_id'] = user_id
                st.session_state['username'] = username
                st.success(f"Bem-vindo, {username}!")
                st.rerun()
            else:
                st.error("Usuário ou senha incorretos.")
    else:
        if st.button("Cadastrar"):
            if not username or not password:
                st.error("Preencha usuário e senha.")
            else:
                try:
                    create_user(username, password)
                    st.success("Usuário cadastrado com sucesso! Faça o login.")
                except ValueError as e:
                    st.error(str(e))
    st.stop()  # Prevents showing the dashboard if not authenticated
```

**Step 3: Add Logout Button and User Header**

Show logout option in the sidebar:
```python
st.sidebar.markdown(f"**Conectado como:** {st.session_state['username']}")
if st.sidebar.button("Sair/Logout"):
    del st.session_state['user_id']
    del st.session_state['username']
    if 'active_config_path' in st.session_state:
        del st.session_state['active_config_path']
    st.rerun()
```

**Step 4: Commit**

```bash
git add scripts/streamlit_app.py
git commit -m "feat: integrate user authentication and logout UI in streamlit"
```

---

### Task 3: Project Filtering, Isolated Storage & IDOR Verification (including Deletion UI)

**Files:**
- Modify: `scripts/streamlit_app.py`

**Step 1: Filter project selection by user**

Modify the sidebar project selection code so it reads from SQLite instead of scanning the whole `inputs/` directory:
```python
# Get projects associated with the logged-in user
db_projects = get_user_projects(st.session_state['user_id'])
project_options = {name: path for name, path in db_projects}
```

**Step 2: Enforce IDOR protection on config path load**

Validate that the selected config path actually belongs to the user:
```python
if 'active_config_path' in st.session_state and st.session_state['active_config_path']:
    path = st.session_state['active_config_path']
    # Enforce IDOR check
    if not verify_project_ownership(st.session_state['user_id'], path):
        st.error("Acesso Negado: Você não tem permissão para acessar este projeto.")
        st.session_state['active_config_path'] = ""
        st.rerun()
```

**Step 3: Isolate new project directories and record relationship in DB**

When a new project is created in the setup form, save files in a user-specific folder and register the project in SQLite:
```python
# New path structure
user_dir = f"inputs/user_{st.session_state['user_id']}/{advertiser_name}"
os.makedirs(user_dir, exist_ok=True)
config_file_path = os.path.join(user_dir, "config.json")

# Write config JSON, save uploaded CSVs to user_dir...
# After successful execution/saving:
add_user_project(st.session_state['user_id'], advertiser_name, config_file_path)
st.session_state['active_config_path'] = config_file_path
```

**Step 4: Implement project deletion UI with safety controls**

Add deletion button in the sidebar under selected project, showing a confirmation step and executing safe cleanup:
```python
if project_options and st.session_state.get('active_config_path'):
    if st.sidebar.button("Excluir Projeto Atual", key="delete_proj_btn"):
        st.session_state['show_delete_confirm'] = True

if st.session_state.get('show_delete_confirm'):
    st.sidebar.warning("Deseja mesmo excluir o projeto e todos os seus arquivos?")
    col_del1, col_del2 = st.sidebar.columns(2)
    with col_del1:
        if st.button("Sim, Excluir", key="confirm_delete_btn"):
            path_to_delete = st.session_state['active_config_path']
            # Verify ownership to prevent IDOR on deletion
            if verify_project_ownership(st.session_state['user_id'], path_to_delete):
                import shutil
                # Delete files on disk
                project_dir = os.path.dirname(path_to_delete)
                if os.path.exists(project_dir):
                    shutil.rmtree(project_dir)
                
                # Delete from outputs too
                output_dir = project_dir.replace("inputs", "outputs")
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
                
                # Delete from SQLite
                delete_user_project(st.session_state['user_id'], selected_project)
                
                st.success(f"Projeto '{selected_project}' excluído.")
                st.session_state['active_config_path'] = ""
                st.session_state['show_delete_confirm'] = False
                st.rerun()
            else:
                st.error("Erro: Acesso não autorizado.")
    with col_del2:
        if st.button("Cancelar", key="cancel_delete_btn"):
            st.session_state['show_delete_confirm'] = False
            st.rerun()
```

**Step 5: Commit**

```bash
git add scripts/streamlit_app.py
git commit -m "feat: enforce project isolation, IDOR guards, deletion UI, and file cleanup"
```
