import sqlite3
import hashlib
import os
import secrets
from contextlib import contextmanager

@contextmanager
def get_db(db_path="data/database.db"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()

def init_db(db_path="data/database.db"):
    with get_db(db_path) as conn:
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
        with get_db(db_path) as conn:
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
    with get_db(db_path) as conn:
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
    with get_db(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO user_projects (user_id, project_name, config_path) VALUES (?, ?, ?)",
            (user_id, project_name.strip(), config_path)
        )
        conn.commit()

def get_user_projects(user_id: int, db_path="data/database.db") -> list[tuple[str, str]]:
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT project_name, config_path FROM user_projects WHERE user_id = ?",
            (user_id,)
        )
        return cursor.fetchall()

def verify_project_ownership(user_id: int, config_path: str, db_path="data/database.db") -> bool:
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM user_projects WHERE user_id = ? AND config_path = ?",
            (user_id, config_path)
        )
        return cursor.fetchone() is not None

def delete_user_project(user_id: int, project_name: str, db_path="data/database.db") -> bool:
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM user_projects WHERE user_id = ? AND project_name = ?",
            (user_id, project_name.strip())
        )
        conn.commit()
        return cursor.rowcount > 0
