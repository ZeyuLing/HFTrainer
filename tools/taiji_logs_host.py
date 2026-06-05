#!/usr/bin/env python3
"""Fetch Taiji logs while selecting a specific MPI host through a PTY."""

import os
import pty
import re
import select
import subprocess
import sys
import time


def fetch_logs(task_flag, instance_id, host_idx, tail=300, timeout=60):
    master, slave = pty.openpty()
    proc = subprocess.Popen(
        [
            'taiji_client',
            'logs',
            '--tail',
            str(tail),
            task_flag,
            instance_id,
        ],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        close_fds=True,
    )
    os.close(slave)

    output = b''
    host_selected = False
    start = time.time()
    while time.time() - start < timeout:
        r, _, _ = select.select([master], [], [], 1)
        if r:
            try:
                data = os.read(master, 8192)
                output += data
            except OSError:
                break
        if not host_selected and b'choose one host' in output:
            try:
                os.write(master, f'{host_idx}\n'.encode())
                host_selected = True
            except OSError:
                break
        if proc.poll() is not None:
            while True:
                r, _, _ = select.select([master], [], [], 0.3)
                if not r:
                    break
                try:
                    output += os.read(master, 8192)
                except OSError:
                    break
            break

    try:
        os.close(master)
    except OSError:
        pass
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

    text = output.decode('utf-8', errors='replace')
    text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
    text = re.sub(r'\x1b\][^\x07]*\x07', '', text)
    return text


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(__doc__)
        print('Usage: taiji_logs_host.py TASK_FLAG INSTANCE_ID HOST_IDX [TAIL] [TIMEOUT]', file=sys.stderr)
        sys.exit(2)
    task_flag = sys.argv[1]
    instance_id = sys.argv[2]
    host_idx = int(sys.argv[3])
    tail = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    timeout = int(sys.argv[5]) if len(sys.argv) > 5 else 60
    print(fetch_logs(task_flag, instance_id, host_idx, tail, timeout))
