"""URL normaliser with three planted regex/parsing bugs."""

from __future__ import annotations

from . import Bug, Scenario, register


SOURCE = '''"""URL normalisation helpers used by the link de-duplication pipeline."""

import re
from urllib.parse import urlsplit, urlunsplit


def lowercase_host(url: str) -> str:
    """Lowercase the host portion of the URL.

    Path, query and fragment are case-sensitive and must be left alone.
    """
    return url.lower()


def strip_default_port(url: str) -> str:
    """Remove :80 from http URLs and :443 from https URLs (keep all others)."""
    return re.sub(r":\\d+", "", url)


def normalize_path(url: str) -> str:
    """Collapse ``/./`` and ``/foo/../bar`` segments in the path."""
    parts = urlsplit(url)
    return urlunsplit(parts._replace(path=parts.path.replace("/./", "/")))
'''


TESTS = '''"""Tests for URL normalisation."""

from url_normalize import lowercase_host, strip_default_port, normalize_path


# ---------- lowercase_host ----------

def test_lowercase_host_already_lower():
    assert lowercase_host("http://example.com/foo") == "http://example.com/foo"


def test_lowercase_host_mixed():
    assert lowercase_host("http://Example.COM/Foo") == "http://example.com/Foo"


# TODO(li): the link-dedupe team had a question about whether the
# fragment is also case-folded. Spec is "host-only"; preserve the rest.
def test_lowercase_host_preserves_path_case():
    """Path /API/v1 must not be lowercased."""
    assert lowercase_host("http://EXAMPLE.com/API/v1") == "http://example.com/API/v1"


def test_lowercase_host_preserves_query():
    """Query string ?Q=Foo must not be lowercased."""
    assert lowercase_host("http://Example.com/p?Q=Foo") == "http://example.com/p?Q=Foo"


# ---------- strip_default_port ----------
# FIXME: previously this was over-zealous and stripped *all* ports. The
# only ports we strip are the protocol defaults: 80 for http, 443 for https.

def test_strip_http_80():
    assert strip_default_port("http://example.com:80/x") == "http://example.com/x"


def test_strip_https_443():
    assert strip_default_port("https://example.com:443/x") == "https://example.com/x"


def test_keep_non_default_port():
    """A non-default port like 8080 must be preserved."""
    assert strip_default_port("http://example.com:8080/x") == "http://example.com:8080/x"


def test_keep_https_8443():
    """https://host:8443 must keep the port."""
    assert strip_default_port("https://example.com:8443/x") == "https://example.com:8443/x"


def test_keep_443_on_http():
    """443 is NOT a default port for http, so it must be kept."""
    assert strip_default_port("http://example.com:443/x") == "http://example.com:443/x"


# ---------- normalize_path ----------

def test_normalize_path_dot():
    assert normalize_path("http://x/a/./b") == "http://x/a/b"


def test_normalize_path_dot_dot():
    """/foo/../bar collapses to /bar."""
    assert normalize_path("http://x/foo/../bar") == "http://x/bar"


def test_normalize_path_chained():
    """/a/b/../c/./d collapses to /a/c/d."""
    assert normalize_path("http://x/a/b/../c/./d") == "http://x/a/c/d"


def test_normalize_path_no_change():
    assert normalize_path("http://x/a/b/c") == "http://x/a/b/c"
'''


README = """# Link Service — URL Normaliser

The de-dup pipeline is producing duplicates because the URL normaliser
isn't matching expected outputs on a few common patterns. Need green
tests so the next batch can ship.
"""


register(
    Scenario(
        name="url_normalize",
        domain="regex_parsing",
        description="URL host-lowercase / default-port-stripping / path-collapse",
        source_file="url_normalize.py",
        test_file="test_url_normalize.py",
        files={
            "url_normalize.py": SOURCE,
            "test_url_normalize.py": TESTS,
            "README.md": README,
        },
        bugs=[
            Bug(
                name="lowercase_whole_url",
                description="lowercase_host lowercases the entire URL, including path/query",
                bug_pattern=r"def\s+lowercase_host[\s\S]{0,200}?return\s+url\.lower\(\)",
                fix_signal=[
                    r"urlsplit\(url\)",
                    r"\.netloc\.lower\(\)",
                    r"urlunsplit",
                ],
            ),
            Bug(
                name="strip_all_ports",
                description="strip_default_port regex matches any :digits, not just defaults",
                bug_pattern=r'r":\\d\+"',
                fix_signal=[
                    r":80\b",
                    r":443\b",
                    r"urlsplit",
                    r"netloc",
                ],
            ),
            Bug(
                name="path_no_dotdot",
                description="normalize_path only collapses /./ , doesn't collapse /foo/../bar",
                bug_pattern=r"\.path\.replace\(\s*[\"']/\./[\"']",
                fix_signal=[
                    r"posixpath\.normpath",
                    r"os\.path\.normpath",
                    r"while\s+[\"']/\.\./[\"']\s+in",
                    r"\.\./",
                ],
            ),
        ],
        baseline_pass=5,
        baseline_fail=8,
    )
)
