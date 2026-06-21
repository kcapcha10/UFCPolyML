"""Solver for the ufcstats.com proof-of-work browser challenge.

ufcstats gates pages behind a lightweight PoW: the challenge page embeds a `nonce`
and a difficulty (a number of leading hex zeros), asks the browser to find an `n`
such that `sha256(f"{nonce}:{n}")` starts with that many zeros, POSTs the solution
to `/__c`, and the server then issues a clearance cookie. This module reproduces
exactly that handshake in Python — the same computation the browser performs — so
the spider can fetch publicly available fight data.
"""

from __future__ import annotations

import hashlib
import re

import httpx

CHALLENGE_MARKER = "Checking your browser"
CLEARANCE_PATH = "/__c"
SOLVE_PATH = "/statistics/events/completed?page=all"
HTTP_TIMEOUT_SECONDS = 30.0
MAX_SOLVE_ITERATIONS = 5_000_000

_NONCE_RE = re.compile(r'nonce="([a-f0-9]+)"')
_DIFFICULTY_RE = re.compile(r"target=new Array\((\d+)\+1\)")


def is_challenge(html: str) -> bool:
    """True if the HTML is the PoW interstitial rather than real content."""
    return CHALLENGE_MARKER in html


def solve_challenge(html: str) -> tuple[str, int]:
    """Extract the nonce + difficulty and return (nonce, solution_n).

    Raises ValueError if the challenge parameters cannot be located.
    """
    nonce_match = _NONCE_RE.search(html)
    difficulty_match = _DIFFICULTY_RE.search(html)
    if not nonce_match or not difficulty_match:
        raise ValueError("challenge page did not contain expected nonce/difficulty")
    nonce = nonce_match.group(1)
    leading_zeros = int(difficulty_match.group(1))
    return nonce, _find_solution(nonce, leading_zeros)


def _find_solution(nonce: str, leading_zeros: int) -> int:
    """Brute-force the smallest n whose sha256 digest has the required prefix."""
    target = "0" * leading_zeros
    for candidate in range(MAX_SOLVE_ITERATIONS):
        digest = hashlib.sha256(f"{nonce}:{candidate}".encode()).hexdigest()
        if digest.startswith(target):
            return candidate
    raise RuntimeError(f"no PoW solution found within {MAX_SOLVE_ITERATIONS} iterations")


def obtain_clearance_cookies(base_url: str, user_agent: str) -> dict[str, str]:
    """Run the full GET → solve → POST handshake and return the clearance cookies.

    Returns an empty dict if the site served real content with no challenge.
    """
    headers = {"User-Agent": user_agent}
    with httpx.Client(
        headers=headers, follow_redirects=True, timeout=HTTP_TIMEOUT_SECONDS
    ) as client:
        response = client.get(f"{base_url}{SOLVE_PATH}")
        if not is_challenge(response.text):
            return dict(client.cookies)
        nonce, solution = solve_challenge(response.text)
        client.post(
            f"{base_url}{CLEARANCE_PATH}",
            data={"nonce": nonce, "n": solution},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        return dict(client.cookies)
