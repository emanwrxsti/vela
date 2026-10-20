"""Loopback transport tests: actual released StratumClient, no CUDA execution."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import ssl
import struct
import subprocess
import threading
import time

import argparse
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("--client", required=True)
parser.add_argument("--mode", choices=["tls", "tcp"], required=True)
parser.add_argument("--output")
args = parser.parse_args()
client = str(Path(args.client).resolve())
temporary = tempfile.TemporaryDirectory(prefix="vela-stratum-")
ROOT = Path(args.output).resolve() if args.output else Path(temporary.name)
ROOT.mkdir(parents=True, exist_ok=True)
context = None
if args.mode == "tls":
    cert = Path(temporary.name)
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                    "-keyout", str(cert / "key.pem"), "-out", str(cert / "cert.pem"),
                    "-days", "1", "-subj", "/CN=localhost"], check=True, capture_output=True)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert / "cert.pem", cert / "key.pem")
results = []

def scenario(tls):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    listener.settimeout(1)
    port = listener.getsockname()[1]
    done = threading.Event()
    records = []
    errors = []
    threads = []
    def serve(conn, index):
        rec = {"connection": index, "accepted": 0, "wrong_job": 0,
               "subscribed": False, "authorized": False, "server_pongs": 0}
        records.append(rec)
        try:
            conn.settimeout(3)
            if tls:
                conn = context.wrap_socket(conn, server_side=True)
            conn.settimeout(0.1)
            buf = b""
            start = time.monotonic()
            # Reuse the same job id on every connection to exercise the session token.
            life = 0.35 if index <= 24 else 60
            def send(obj):
                conn.sendall((json.dumps(obj) + "\n").encode())
            while not done.is_set() and time.monotonic() - start < life:
                try:
                    data = conn.recv(65536)
                except socket.timeout:
                    continue
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    m = json.loads(line)
                    method = m.get("method")
                    if method == "mining.subscribe":
                        rec["subscribed"] = True
                        send({"id": m["id"], "result": {"extranonce1": f"{index:08x}"}, "error": None})
                    elif method == "mining.authorize":
                        rec["authorized"] = True
                        send({"id": m["id"], "result": True, "error": None})
                        send({"id": None, "method": "mining.set_difficulty", "params": {"difficulty": 1}})
                        send({"id": None, "method": "mining.notify", "params": {
                            "job_id": "reused-job-id", "prev_hash": "00" * 32,
                            "coinbase1": "", "coinbase2": "", "version": "00000001",
                            "bits": "1d00ffff", "ntime": "00000001", "clean_jobs": True,
                            "merkle_branch": [], "prompt_tokens": [1]}})
                        send({"id": 1000000 + index, "method": "mining.ping", "params": {}})
                    elif method == "mining.submit":
                        wrong = (m["params"]["job_id"] != "reused-job-id"
                                 or m["params"]["extranonce2"] != f"{index:08x}")
                        rec["wrong_job"] += wrong
                        rec["accepted"] += not wrong
                        send({"id": m["id"], "result": not wrong, "error": None})
                    elif method == "mining.ping":
                        send({"id": m["id"], "result": "pong", "error": None})
                    elif m.get("result") == "pong":
                        rec["server_pongs"] += 1
            if index <= 24:
                if index % 2:
                    conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("HH" if os.name == "nt" else "ii", 1, 0))
                    rec["drop"] = "RST"
                else:
                    conn.shutdown(socket.SHUT_RDWR)
                    rec["drop"] = "EOF"
        except (ConnectionResetError, BrokenPipeError, ssl.SSLEOFError):
            rec["peer_closed"] = True
        except Exception as e:
            errors.append({"connection": index, "error": repr(e)})
        finally:
            conn.close()
    def accept():
        index = 0
        while not done.is_set():
            try:
                conn, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            index += 1
            t = threading.Thread(target=serve, args=(conn, index), daemon=True)
            threads.append(t)
            t.start()
    acceptor = threading.Thread(target=accept, daemon=True)
    acceptor.start()
    mode = "tls" if tls else "tcp"
    logfile = ROOT / f"transport-{mode}.log"
    with logfile.open("w") as log:
        proc = subprocess.run([client, str(port), str(int(tls)), "16"],
                              stdout=log, stderr=subprocess.STDOUT, timeout=40)
    done.set()
    listener.close()
    acceptor.join(3)
    for t in threads:
        t.join(4)
    text = logfile.read_text()
    lines = [x[7:] for x in text.splitlines() if x.startswith("RESULT ")]
    stats = json.loads(lines[-1]) if lines else {}
    report = {"mode": mode, "returncode": proc.returncode, "stats": stats,
              "sessions": records, "errors": errors,
              "log_sha256": hashlib.sha256(logfile.read_bytes()).hexdigest()}
    report["pass"] = (
        proc.returncode == 0 and not errors and len(records) == 25
        and stats.get("connections") == 25 and stats.get("rejected") == 0
        and stats.get("accepted", 0) > 1000 and stats.get("discarded", 0) > 0
        and all(r["subscribed"] and r["authorized"] and r["accepted"] > 0
                and not r["wrong_job"] and r["server_pongs"] == 1 for r in records)
        and "AddressSanitizer" not in text and "runtime error:" not in text)
    (ROOT / f"transport-{mode}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "sessions"}), flush=True)
    return report

result = scenario(args.mode == "tls")
print("PASS" if result["pass"] else "FAIL", flush=True)
raise SystemExit(0 if result["pass"] else 1)
