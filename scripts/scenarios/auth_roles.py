"""Role-based access control with three planted bugs."""

from __future__ import annotations

from . import Bug, Scenario, register


SOURCE = '''"""Role-based access control for the admin console."""


# Role hierarchy (each role inherits the permissions of the roles below it)
ROLE_HIERARCHY = {
    "owner": ["admin", "editor", "viewer"],
    "admin": ["editor", "viewer"],
    "editor": ["viewer"],
    "viewer": [],
}


def has_role(user: dict, role: str) -> bool:
    """True if `user` is granted `role` (directly or via hierarchy)."""
    if user.get("role") == role:
        return True
    return False


def can_access(user: dict, resource: dict) -> bool:
    """True if `user` is allowed to access `resource`.

    Rules:
      - The resource lists `required_roles`; the user must hold AT LEAST one.
      - If `resource["owner_id"] == user["id"]`, access is always granted.
      - Suspended users (`user["suspended"]` truthy) never have access.
    """
    if resource.get("owner_id") == user.get("id"):
        return True
    if user.get("suspended"):
        return True
    for role in resource.get("required_roles", []):
        if has_role(user, role):
            return True
    return False


def is_admin_email(email: str) -> bool:
    """True if `email` is from an admin domain.

    Admin domains are listed in ADMIN_DOMAINS. Match is case-insensitive.
    """
    return email.split("@")[1] in ADMIN_DOMAINS


ADMIN_DOMAINS = {"acme.com", "acme-internal.net"}
'''


TESTS = '''"""Tests for the auth-roles module."""

import pytest

from auth_roles import has_role, can_access, is_admin_email


# ---------- has_role: direct membership ----------

def test_has_role_direct_match():
    user = {"id": 1, "role": "editor"}
    assert has_role(user, "editor")


def test_has_role_direct_mismatch():
    user = {"id": 1, "role": "viewer"}
    assert not has_role(user, "editor")


# ---------- has_role: hierarchy ----------
# TODO(jenny): hierarchy semantics still being firmed up with security
# review; spec is "owner > admin > editor > viewer" with strict inheritance.

def test_has_role_owner_inherits_admin():
    """Owner should inherit admin permissions."""
    user = {"id": 1, "role": "owner"}
    assert has_role(user, "admin")


def test_has_role_admin_inherits_viewer():
    user = {"id": 1, "role": "admin"}
    assert has_role(user, "viewer")


def test_has_role_editor_does_not_inherit_admin():
    user = {"id": 1, "role": "editor"}
    assert not has_role(user, "admin")


# ---------- can_access ----------
# FIXME: a security audit flagged that suspended users were getting access
# when they hit the owner short-circuit. Spec is now "suspended denies
# access regardless of any other rule." Tests below lock that contract.

def test_can_access_owner_short_circuit():
    user = {"id": 7, "role": "viewer"}
    resource = {"owner_id": 7, "required_roles": ["admin"]}
    assert can_access(user, resource)


def test_can_access_required_role_match():
    user = {"id": 1, "role": "admin"}
    resource = {"owner_id": 99, "required_roles": ["admin"]}
    assert can_access(user, resource)


def test_can_access_required_role_via_hierarchy():
    user = {"id": 1, "role": "owner"}
    resource = {"owner_id": 99, "required_roles": ["editor"]}
    assert can_access(user, resource)


def test_can_access_no_role_no_owner():
    user = {"id": 1, "role": "viewer"}
    resource = {"owner_id": 99, "required_roles": ["admin"]}
    assert not can_access(user, resource)


def test_can_access_suspended_blocked_even_for_owner():
    """Suspended users get NO access, even on resources they own."""
    user = {"id": 7, "role": "owner", "suspended": True}
    resource = {"owner_id": 7, "required_roles": ["admin"]}
    assert not can_access(user, resource)


def test_can_access_suspended_blocked_with_role():
    user = {"id": 1, "role": "admin", "suspended": True}
    resource = {"owner_id": 99, "required_roles": ["admin"]}
    assert not can_access(user, resource)


# ---------- is_admin_email ----------

def test_is_admin_email_match():
    assert is_admin_email("alice@acme.com")


def test_is_admin_email_no_match():
    assert not is_admin_email("alice@gmail.com")


def test_is_admin_email_case_insensitive():
    """Match is case-insensitive per spec."""
    assert is_admin_email("alice@ACME.com")
'''


README = """# Admin Console — Auth & Roles

The new RBAC module isn't passing its full test suite — hierarchy +
suspended-user semantics + case-insensitive admin domains. We need this
green before the launch.
"""


register(
    Scenario(
        name="auth_roles",
        domain="boolean_logic",
        description="RBAC with role hierarchy + suspended-user gate + email matching",
        source_file="auth_roles.py",
        test_file="test_auth_roles.py",
        files={
            "auth_roles.py": SOURCE,
            "test_auth_roles.py": TESTS,
            "README.md": README,
        },
        bugs=[
            Bug(
                name="has_role_no_hierarchy",
                description="has_role only checks direct equality, ignores ROLE_HIERARCHY",
                bug_pattern=r"def\s+has_role[\s\S]{0,200}?if\s+user\.get\(\s*[\"']role[\"']\s*\)\s*==\s*role\s*:\s*\n\s*return\s+True\s*\n\s*return\s+False",
                fix_signal=[
                    r"ROLE_HIERARCHY",
                    r"in\s+ROLE_HIERARCHY",
                ],
            ),
            Bug(
                name="suspended_grants_access",
                description="`if user.get('suspended'): return True` (should be False)",
                bug_pattern=r"if\s+user\.get\(\s*[\"']suspended[\"']\s*\)\s*:\s*\n\s*return\s+True",
                fix_signal=[
                    r"if\s+user\.get\(\s*[\"']suspended[\"']\s*\)\s*:\s*\n\s*return\s+False",
                ],
            ),
            Bug(
                name="admin_email_case_sensitive",
                description="is_admin_email doesn't lowercase domain before lookup",
                bug_pattern=r"return\s+email\.split\(\s*[\"']@[\"']\s*\)\[\s*1\s*\]\s+in\s+ADMIN_DOMAINS",
                fix_signal=[
                    r"\.lower\(\)\s*\)?\s*in\s+ADMIN_DOMAINS",
                    r"\.casefold\(\)",
                ],
            ),
        ],
        baseline_pass=8,
        baseline_fail=6,
    )
)
