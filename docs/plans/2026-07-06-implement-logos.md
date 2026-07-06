# Implementar Logos com Suporte a Tema Claro e Escuro

> **For Antigravity:** REQUIRED WORKFLOW: Use `.agent/workflows/execute-plan.md` to execute this plan in single-flow mode.

**Goal:** Implement the Dash and AlmapBBDO logos in the Streamlit application with automatic theme-based (light vs dark) switching.

**Architecture:** 
- Convert the logo PNG files (`DASH_POSITIVO (1).png` / `DASH_NEGATIVO.png` and `AF_ALMAPBBDO_LOGO_FINAL FILIPE-01.png` / `AF_ALMAPBBDO_LOGO_FINAL FILIPE-04.png`) to base64 strings dynamically on app startup.
- Inject CSS in the dashboard's `PREMIUM_CSS` block that controls the visibility of `.logo-light` and `.logo-dark` elements based on both browser preference (`@media (prefers-color-scheme: dark)`) and Streamlit's internal HTML theme attributes (`[data-theme="dark"]`).
- Render the Dash logo at the top of the login screen and at the top of the sidebar.
- Render the AlmapBBDO logo as a "powered by" footer at the bottom of the login screen and at the bottom of the sidebar.

**Tech Stack:** Python 3.13, Streamlit, CSS, Base64 encoding.

---

### Task 1: Adicionar CSS de Comutação de Tema das Logos

**Files:**
- Modify: `scripts/streamlit_app.py:76-243`

**Step 1: Write the changes to PREMIUM_CSS**
Add responsive rules for `.logo-light` and `.logo-dark` so that when the theme is dark, light logos are hidden and dark logos are displayed.

**Step 2: Commit**
```bash
git add scripts/streamlit_app.py
git commit -m "style: add CSS class selectors for theme-aware logos"
```

---

### Task 2: Carregamento dos Logos em Base64

**Files:**
- Modify: `scripts/streamlit_app.py:245-281`

**Step 1: Write base64 loading utility and load logos**
Implement a cached base64 loader helper and initialize variables `dash_light`, `dash_dark`, `almap_light`, `almap_dark`.

**Step 2: Commit**
```bash
git add scripts/streamlit_app.py
git commit -m "feat: implement base64 loader and cache logo images"
```

---

### Task 3: Renderizar Logos na Tela de Login

**Files:**
- Modify: `scripts/streamlit_app.py:326-376`

**Step 1: Update login header & footer**
Replace the simple text title "Opportunity Engine" with the theme-aware Dash logo, and add the AlmapBBDO logo in a footer below the login card.

**Step 2: Commit**
```bash
git add scripts/streamlit_app.py
git commit -m "feat: add logo branding to the login page"
```

---

### Task 4: Renderizar Logos no Sidebar Principal

**Files:**
- Modify: `scripts/streamlit_app.py:413-440` and end of file

**Step 1: Add Dash logo at sidebar top and AlmapBBDO at sidebar bottom**
Place the Dash logo at the top of the sidebar. Append the AlmapBBDO logo as a footer at the bottom of the sidebar.

**Step 2: Commit**
```bash
git add scripts/streamlit_app.py
git commit -m "feat: add logo branding to sidebar header and footer"
```
