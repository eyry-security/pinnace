"""Host-binding guard for agent tool calls.

A host review is bound to exactly one host. Every tool call that can name a
target is checked here before it executes: the host must equal the assigned
host (or be a subdomain of a wildcard assignment). A small, configurable set
of third-party service endpoints is treated as *reference lookups* (certificate
transparency, vulnerability databases, identity endpoints) and may be used by
read-only lookup tools without being the target of an offensive action.

The guard is deliberately conservative: if a tool argument or shell command
looks like it names a hostname, that hostname must be in scope. Rejected calls
return a clear error and never reach the tool implementation.
"""

from __future__ import annotations

import ipaddress
import os
import re
import urllib.parse
from dataclasses import dataclass, field

REFERENCE_HOSTS_ENV = "PINNACE_TARGET_REFERENCE_HOSTS"

# Reference lookups are ABOUT the target, never attacks ON a target. These are
# the built-in service endpoints; operators can extend the list with
# PINNACE_TARGET_REFERENCE_HOSTS (comma-separated hostnames).
DEFAULT_REFERENCE_HOSTS = frozenset({
    "crt.sh",
    "www.crt.sh",
    "certspotter.com",
    "api.certspotter.com",
    "certstream.calidog.io",
    "osv.dev",
    "api.osv.dev",
    "api.shodan.io",
    "www.virustotal.com",
    "safebrowsing.googleapis.com",
    "accounts.google.com",
    "oauth2.googleapis.com",
    "identitytoolkit.googleapis.com",
    "login.microsoftonline.com",
    "graph.microsoft.com",
})

# Tools whose arguments are data, not target actions.
_SKIP_TOOLS = frozenset({
    "finish",
    "read_file",
    "buffer_read",
    "todos_write",
    "todos_update",
    "set_auth_session",
    "show_auth_status",
    "elide",
})

# Tool names that always act on a target. Kept for readability/documentation;
# the default path for any non-skipped tool is to validate host-like strings.
_OFFENSIVE_TOOLS = frozenset({
    "shell",
    "curl",
    "fetch_url",
    "nuclei",
    "ffuf",
    "dalfox",
    "sqli",
    "katana",
    "hakrawler",
    "discover_params",
    "js_recon",
    "screenshot",
    "browser_auth",
    "python_scratchpad",
})

_REFERENCE_TOOLS = frozenset({"fetch_url"})

# Shell commands that actively test a target. If any appear in a command
# segment, allowlisted reference hosts are NOT exempt in that segment.
_OFFENSIVE_SHELL_COMMANDS = frozenset({
    "nuclei",
    "ffuf",
    "feroxbuster",
    "gobuster",
    "dirb",
    "dirsearch",
    "nikto",
    "sqlmap",
    "dalfox",
    "katana",
    "hakrawler",
    "nmap",
    "masscan",
    "ncat",
    "nc",
    "netcat",
})

# Read-only lookup clients. Allowlisted service endpoints may be contacted by
# these without being the assigned target.
_REFERENCE_SHELL_COMMANDS = frozenset({
    "curl",
    "wget",
    "dig",
    "host",
    "nslookup",
    "whois",
    "openssl",
})

# Common file extensions. Prevents code/paths (result.get, cookies.txt) from
# being mistaken for hostnames. This intentionally also blocks a few real
# gTLDs that collide with file suffixes (.zip, .sh); that failure mode is
# strict (a rejected call), never permissive.
_FILE_SUFFIXES = frozenset({
    "txt", "json", "jsonl", "yaml", "yml", "html", "htm", "xhtml", "php",
    "asp", "aspx", "jsp", "js", "mjs", "cjs", "ts", "tsx", "jsx", "css",
    "scss", "less", "md", "markdown", "xml", "toml", "ini", "cfg", "conf",
    "log", "csv", "sql", "db", "sqlite", "sqlite3", "py", "pyc", "sh",
    "bash", "zsh", "fish", "ps1", "bat", "cmd", "exe", "dll", "so", "dylib",
    "bin", "img", "iso", "tar", "gz", "tgz", "bz2", "xz", "zip", "rar",
    "7z", "png", "jpg", "jpeg", "gif", "webp", "bmp", "ico", "svg", "pdf",
    "doc", "docx", "xls", "xlsx", "ppt", "pptx", "woff", "woff2", "ttf",
    "otf", "eot", "mp3", "mp4", "wav", "avi", "mov", "mkv", "webm", "lock",
    "map", "wasm", "out", "err", "dat", "tmp", "bak", "old", "orig", "swp",
    "swo",
})

# Common TLDs plus the reserved test suffixes. For tokens with three or more
# labels we accept any alphabetic TLD, which covers subdomains of less common
# names without treating Python attribute chains (result.get) as hosts.
_KNOWN_TLDS = frozenset("""
com net org edu gov mil int info biz pro mobi asia travel jobs cat coop
aero museum tel xxx app dev io ai cloud site online tech store shop blog wiki
page link live news media agency digital network systems services solutions
software tools email space website fun life world today center company group
team works run build plus one top vip club art design studio photography
gallery video audio radio games game sports health care clinic dental doctor
hospital pharmacy finance bank insurance legal law attorney tax accounting
realestate properties rentals homes house land farm garden flowers gifts
jewelry shoes clothing fashion beauty hair skin fitness yoga dance music band
show theatre tickets events party wedding date love friends community church
faith bible family kids baby school university college academy education
courses degree training careers work consulting management marketing
advertising press reporter review reviews guide tips help support repair
cleaning moving security protection loans credit mortgage capital invest
money cash crypto bitcoin wallet exchange trading markets stocks business
enterprise industries industrial manufacturing engineering construction
contractors plumbing electrician roofing painting hvac transport logistics
shipping delivery taxi cars auto motorcycle bike boat yacht flights airline
airport hotel motel resort tours cruises vacation camping fishing hunting
outdoor adventure soccer football basketball baseball hockey tennis golf
cricket rugby racing motorsport gym trainer coach nutrition diet recipes food
restaurant cafe bar pub brewery winery vodka beer wine pizza sushi coffee tea
bakery butcher grocery market boutique mall outlet sale deals discount coupon
free cheap best expert master guru ninja rockstar wizard hero champion winner
leader chief boss exec premium luxury elite gold silver diamond platinum
prime first original authentic official certified licensed insured trusted
secure safe fast quick easy simple smart clever bright clear clean fresh new
modern classic vintage retro urban city town village country national
international global universal total complete full all every any some many
much more most less least few little big small large huge mega ultra super
hyper micro mini nano
ac ad ae af ag ai al am ao aq ar as at au aw ax az ba bb bd be bf bg bh bi
bj bm bn bo br bs bt bw by bz ca cc cd cf cg ch ci ck cl cm cn cr cu cv
cw cx cy cz de dj dk dm do dz ec ee eg er es et eu fi fj fk fm fo fr ga gb
gd ge gf gg gh gi gl gm gn gp gq gr gs gt gu gw gy hk hm hn hr ht hu id ie
il im in io iq ir is it je jm jo jp ke kg kh ki km kn kp kr kw ky kz la lb
lc li lk lr ls lt lu lv ly ma mc md me mg mh mk ml mm mn mo mp mq mr ms mt
mu mv mw mx my mz na nc ne nf ng ni nl no np nr nu nz om pa pe pf pg ph pk
pl pm pn pr ps pt pw py qa re ro rs ru rw sa sb sc sd se sg sh si sk sl sm
sn so sr ss st su sv sx sy sz tc td tf tg th tj tk tl tm tn to tr tt tv tw
tz ua ug uk us uy uz va vc ve vg vi vn vu wf ws ye yt za zm zw
example test invalid localhost
""".split())

_URL_RE = re.compile(
    r"(?i)\b[a-z][a-z0-9+.-]*://([^\s/?#\"'<>]+)")
_HOST_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,24}"
    r"(?![A-Za-z0-9_.-])")
_SHELL_SPLIT_RE = re.compile(r"(?:&&|\|\||;|\n|\|)")


def _normalise(host: object) -> str:
    value = str(host or "").strip().lower()
    if value.startswith("*."):
        value = value[2:]
    if value.endswith("."):
        value = value[:-1]
    # A bare authority may carry a port or userinfo; keep only the hostname.
    if "://" in value:
        try:
            value = urllib.parse.urlsplit(value).hostname or ""
        except ValueError:
            value = ""
    elif "@" in value:
        value = value.rsplit("@", 1)[-1]
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    if ":" in value and value.count(":") == 1:
        value = value.split(":", 1)[0]
    return value.strip(".").lower()


def _valid_hostname(host: str) -> bool:
    if not host or len(host) > 253 or ".." in host:
        return False
    try:
        ipaddress.ip_address(host)
        return False
    except ValueError:
        pass
    labels = host.split(".")
    if len(labels) < 2:
        return False
    for label in labels:
        if not label or len(label) > 63:
            return False
        if label.startswith("-") or label.endswith("-"):
            return False
        if not re.fullmatch(r"[a-z0-9-]+", label):
            return False
    return True


def _accept_host_token(host: str) -> bool:
    if not _valid_hostname(host):
        return False
    labels = host.split(".")
    tld = labels[-1]
    if tld in _FILE_SUFFIXES:
        return False
    if len(labels) >= 3:
        return True
    return tld in _KNOWN_TLDS


def extract_hosts(text: object) -> list[str]:
    """Return hostname-like tokens from *text* in first-seen order."""
    value = str(text or "")
    found: list[str] = []

    def _add(candidate: object, *, from_url: bool = False) -> None:
        host = _normalise(candidate)
        accepted = (_valid_hostname(host) if from_url
                    else _accept_host_token(host))
        if host and accepted and host not in found:
            found.append(host)

    for match in _URL_RE.finditer(value):
        authority = match.group(1)
        if "@" in authority:
            authority = authority.rsplit("@", 1)[-1]
        if authority.startswith("["):
            authority = authority.split("]", 1)[0] + "]"
        else:
            authority = authority.split(":", 1)[0]
        _add(authority, from_url=True)
    for match in _HOST_TOKEN_RE.finditer(value):
        _add(match.group(0))
    return found


def default_reference_hosts() -> frozenset[str]:
    """Built-in reference endpoints plus the env-configured additions."""
    hosts = set(DEFAULT_REFERENCE_HOSTS)
    raw = (os.environ.get(REFERENCE_HOSTS_ENV) or "").strip()
    for item in re.split(r"[\s,]+", raw):
        host = _normalise(item)
        if host:
            hosts.add(host)
    return frozenset(hosts)


@dataclass(frozen=True)
class TargetBinding:
    """The only host a host-review segment is allowed to act against."""

    assigned_host: str
    allow_subdomains: bool = False
    reference_hosts: frozenset[str] = field(
        default_factory=default_reference_hosts)

    def __post_init__(self) -> None:
        raw = str(self.assigned_host or "").strip()
        wildcard = raw.startswith("*.")
        host = _normalise(raw)
        if not host:
            raise ValueError("assigned_host must be a non-empty hostname")
        object.__setattr__(self, "assigned_host", host)
        object.__setattr__(
            self, "allow_subdomains",
            bool(self.allow_subdomains or wildcard))
        refs = frozenset(
            _normalise(item) for item in self.reference_hosts if _normalise(item))
        object.__setattr__(self, "reference_hosts", refs)

    @classmethod
    def from_host(cls, host: object, *,
                  allow_subdomains: bool | None = None,
                  reference_hosts=None) -> "TargetBinding":
        raw = str(host or "").strip()
        wildcard = raw.startswith("*.")
        if "://" in raw:
            try:
                raw = urllib.parse.urlsplit(raw).hostname or raw
            except ValueError:
                pass
        refs = (default_reference_hosts() if reference_hosts is None
                else frozenset(reference_hosts))
        return cls(
            assigned_host=raw,
            allow_subdomains=(wildcard if allow_subdomains is None
                              else bool(allow_subdomains)),
            reference_hosts=refs,
        )

    def allows(self, host: object) -> bool:
        candidate = _normalise(host)
        if not candidate:
            return False
        if candidate == self.assigned_host:
            return True
        return (self.allow_subdomains
                and candidate.endswith("." + self.assigned_host))

    def is_reference(self, host: object) -> bool:
        return _normalise(host) in self.reference_hosts

    def describe(self) -> str:
        return self.assigned_host

    def rejection(self, offender: object = "") -> str:
        bad = _normalise(offender)
        if bad:
            return (f"error: out of scope: this review is bound to "
                    f"{self.assigned_host}; {bad} is not in scope")
        return (f"error: out of scope: this review is bound to "
                f"{self.assigned_host}")

    def check_text(self, text: object, *,
                   allow_reference: bool = True) -> str | None:
        """Return a rejection message when *text* names an out-of-scope host."""
        for host in extract_hosts(text):
            if self.allows(host):
                continue
            if allow_reference and self.is_reference(host):
                continue
            return self.rejection(host)
        return None

    def _check_shell(self, command: str) -> str | None:
        for segment in _SHELL_SPLIT_RE.split(command or ""):
            hosts = extract_hosts(segment)
            if not hosts:
                continue
            words = set()
            for token in re.split(r"[\s'\"`]+", segment.strip()):
                if not token:
                    continue
                words.add(token.rsplit("/", 1)[-1].lower())
                words.add(token.lower())
            offensive = bool(words & _OFFENSIVE_SHELL_COMMANDS)
            reference = bool(words & _REFERENCE_SHELL_COMMANDS)
            for host in hosts:
                if self.allows(host):
                    continue
                if reference and not offensive and self.is_reference(host):
                    continue
                return self.rejection(host)
        return None

    def check_call(self, tool_name: str, args: dict | None) -> str | None:
        """Return a rejection message, or None when the call is in scope."""
        name = str(tool_name or "")
        if name in _SKIP_TOOLS:
            return None
        values = args if isinstance(args, dict) else {}
        if name == "shell":
            return self._check_shell(str(values.get("command") or ""))
        if name == "delegate_task":
            # A brief is prose: a mention of an allowlisted reference service
            # is fine, but naming any other host is a cross-host target.
            for key in ("task", "context"):
                reason = self.check_text(values.get(key), allow_reference=True)
                if reason:
                    return reason
            return None
        allow_reference = name in _REFERENCE_TOOLS
        for value in values.values():
            if isinstance(value, str):
                reason = self.check_text(value, allow_reference=allow_reference)
                if reason:
                    return reason
        return None

    def filter_output(self, text: object) -> str:
        """Drop output lines that name hosts outside the binding.

        Used before tool output enters the review context, so a legitimate
        multi-host response cannot leak another host's data into this
        segment. Reference-service lines are kept.
        """
        value = str(text or "")
        if not value:
            return value
        kept: list[str] = []
        dropped = 0
        for line in value.splitlines(keepends=True):
            bad = [host for host in extract_hosts(line)
                   if not self.allows(host) and not self.is_reference(host)]
            if bad:
                dropped += 1
                continue
            kept.append(line)
        if not dropped:
            return value
        kept.append(
            f"\n[filtered {dropped} out-of-scope output line(s); "
            f"this review is bound to {self.assigned_host}]\n")
        return "".join(kept)

    def filter_result(self, result):
        """Filter a string-like tool result, preserving FullOutput wrappers."""
        if not isinstance(result, str):
            return result
        visible = self.filter_output(result)
        full = getattr(result, "full_output", None)
        if isinstance(full, str):
            filtered_full = self.filter_output(full)
            try:
                return type(result)(visible, filtered_full)
            except Exception:  # noqa: BLE001 - fall back to the visible text
                return visible
        return visible


def guard_tool_call(binding: TargetBinding | None, tool_name: str,
                    args: dict | None) -> str | None:
    """Convenience wrapper used by the agent loops."""
    if binding is None:
        return None
    return binding.check_call(tool_name, args)
