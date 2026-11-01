"""Signed AI-output marking for exported audio (C2PA Content Credentials).

The tags in provenance.py can be edited without a trace. This adds a signed
record to the WAV, in its own RIFF chunk after the audio, so the samples aren't
touched. Changing the audio or the tags afterwards makes verification fail.

Each install signs with its own key, generated on first use and kept in the
data folder. A key shipped in an open-source app could be extracted, and anyone
could then sign as Fragmenta. So a signature shows the file hasn't changed since
that install signed it, not who made it, and verifiers report the signer as
untrusted. Files from one install share a certificate, so they can be linked.

Time-stamping sends a hash, never audio, to a public server. Offline, files are
signed without a time-stamp. Marking never blocks an export: on any failure the
tagged file stays as it is and the reason goes to the log.
"""
import datetime as dt
import os
import socket
import threading
import time
from pathlib import Path

from app.core.config import get_config
from utils.logger import get_logger

logger = get_logger("ContentCredentials")

TIMESTAMP_HOST = "timestamp.digicert.com"
TIMESTAMP_URL = f"http://{TIMESTAMP_HOST}"
_CONNECT_TIMEOUT_S = 1.5
# After a failure, don't let every export wait on the same dead network. A
# server that accepts and then stalls costs 30 s (the library's limit) once.
_TIMESTAMP_RETRY_S = 300

_CERT_YEARS = 10
_RENEW_BEFORE = dt.timedelta(days=30)
_CERT_FILE = "signing_cert.pem"
_KEY_FILE = "signing_key.pem"
# Says so in every verifier: nobody has checked this signer's identity.
_SIGNER_ORG = "Fragmenta install (self-issued, unverified)"

_key_lock = threading.Lock()
_skip_timestamp_until = 0.0


def _new_credentials() -> tuple:
    """A leaf certificate + key, plus the throwaway CA that issued it: C2PA
    rejects a self-signed leaf. The CA's key is discarded. No user or machine
    name goes in, since the certificate travels with every file."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    def name(common_name):
        return x509.Name([x509.NameAttribute(NameOID.ORGANIZATION_NAME, _SIGNER_ORG),
                          x509.NameAttribute(NameOID.COMMON_NAME, common_name)])

    def usage(**enabled):
        flags = dict.fromkeys(
            ("digital_signature", "content_commitment", "key_encipherment",
             "data_encipherment", "key_agreement", "key_cert_sign", "crl_sign",
             "encipher_only", "decipher_only"), False)
        return x509.KeyUsage(**{**flags, **enabled})

    now = dt.datetime.now(dt.timezone.utc)
    start, end = now - dt.timedelta(days=1), now + dt.timedelta(days=365 * _CERT_YEARS)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    leaf_key = ec.generate_private_key(ec.SECP256R1())
    ca = (x509.CertificateBuilder()
          .subject_name(name("Local CA")).issuer_name(name("Local CA"))
          .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
          .not_valid_before(start).not_valid_after(end)
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .add_extension(usage(key_cert_sign=True, crl_sign=True), critical=True)
          .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
          .sign(ca_key, hashes.SHA256()))
    leaf = (x509.CertificateBuilder()
            .subject_name(name("Signing key")).issuer_name(ca.subject)
            .public_key(leaf_key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(start).not_valid_after(end)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(usage(digital_signature=True), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]), critical=False)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(leaf_key.public_key()), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256()))
    chain = (leaf.public_bytes(serialization.Encoding.PEM)
             + ca.public_bytes(serialization.Encoding.PEM))
    key = leaf_key.private_bytes(serialization.Encoding.PEM,
                                 serialization.PrivateFormat.PKCS8,
                                 serialization.NoEncryption())
    return chain, key


def _usable(chain: bytes, key: bytes) -> bool:
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    leaf = x509.load_pem_x509_certificates(chain)[0]
    if leaf.not_valid_after_utc - dt.datetime.now(dt.timezone.utc) < _RENEW_BEFORE:
        return False
    private = serialization.load_pem_private_key(key, password=None)
    return private.public_key().public_numbers() == leaf.public_key().public_numbers()


def _write_private(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _load_credentials() -> tuple:
    """This install's (certificate chain, key), created on first use and
    replaced if damaged or close to expiry. Older files stay verifiable: each
    carries its own certificate."""
    folder = get_config().get_path("credentials")
    cert_path, key_path = folder / _CERT_FILE, folder / _KEY_FILE
    with _key_lock:
        try:
            chain, key = cert_path.read_bytes(), key_path.read_bytes()
            if _usable(chain, key):
                return chain, key
        except (OSError, ValueError, TypeError):
            pass
        folder.mkdir(parents=True, exist_ok=True)
        chain, key = _new_credentials()
        _write_private(key_path, key)
        _write_private(cert_path, chain)
        return chain, key


def _timestamp_url():
    """The server's URL if it's worth trying right now. Without this check a
    dead network could hold up every export while DNS or connecting times out."""
    if time.monotonic() < _skip_timestamp_until:
        return None
    try:
        socket.create_connection((TIMESTAMP_HOST, 80), timeout=_CONNECT_TIMEOUT_S).close()
        return TIMESTAMP_URL
    except OSError:
        _pause_timestamps()
        return None


def _pause_timestamps() -> None:
    global _skip_timestamp_until
    _skip_timestamp_until = time.monotonic() + _TIMESTAMP_RETRY_S


def _sign(c2pa, src: Path, dest: Path, manifest: dict, chain: bytes, key: bytes, url) -> None:
    info = c2pa.C2paSignerInfo(alg=b"es256", sign_cert=chain, private_key=key,
                               ta_url=url.encode() if url else None)
    with c2pa.Signer.from_info(info) as signer, c2pa.Builder(manifest) as builder:
        with open(src, "rb") as source, open(dest, "w+b") as output:
            builder.sign(signer, "audio/wav", source, output)


def sign_wav(path: Path, manifest: dict) -> bool:
    """Add a signed record to a WAV in place. Returns False, with the file left
    untouched, if it can't. Runs last: any later rewrite would invalidate it."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.signing")
    try:
        import c2pa

        chain, key = _load_credentials()
        url = _timestamp_url()
        try:
            _sign(c2pa, path, tmp, manifest, chain, key, url)
        except Exception as exc:
            if url is None:
                raise
            logger.info(f"Signing {path.name} without a time-stamp: {exc}")
            _pause_timestamps()
            _sign(c2pa, path, tmp, manifest, chain, key, None)
        os.replace(tmp, path)
        return True
    except Exception as exc:
        logger.warning(f"Could not sign {path.name}: {exc}")
        try:
            tmp.unlink()
        except OSError:
            pass
        return False
