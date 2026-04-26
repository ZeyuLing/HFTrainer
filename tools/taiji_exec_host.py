#!/usr/bin/env python3
"""taiji_client exec wrapper with explicit host selection.

When a Taiji task has multiple MPI hosts (launcher + workers), the interactive
`taiji_client exec` prompts for a host index. This wrapper feeds the chosen
index through the PTY before passing the user command.

Usage:
    python3 tools/taiji_exec_host.py <task_flag> <instance_id> <host_idx> <command> [timeout]
"""

import os
import pty
import re
import select
import subprocess
import sys
import time


def taiji_exec_host(task_flag, instance_id, host_idx, command, timeout=60):
    master, slave = pty.openpty()
    proc = subprocess.Popen(
        [
            'taiji_client', 'exec', task_flag, instance_id,
            'bash', '-c', command + '; echo __EXIT_CODE__$?; exit',
        ],
        stdin=slave, stdout=slave, stderr=slave,
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
                os.write(master, f"{host_idx}\n".encode())
                host_selected = True
            except OSError:
                break
        if proc.poll() is not None:
            while True:
                r, _, _ = select.select([master], [], [], 0.3)
                if r:
                    try:
                        output += os.read(master, 8192)
                    except Exception:
                        break
                else:
                    break
            break

    try:
        os.close(master)
    except Exception:
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
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

    exit_code = -1
    clean_lines = []
    for line in text.strip().split('\n'):
        if '__EXIT_CODE__' in line:
            try:
                exit_code = int(line.strip().split('__EXIT_CODE__')[-1])
            except Exception:
                exit_code = 0
        else:
            clean_lines.append(line)
    return '\n'.join(clean_lines).strip(), exit_code


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)
    task_flag = sys.argv[1]
    instance_id = sys.argv[2]
    host_idx = int(sys.argv[3])
    command = sys.argv[4]
    timeout = int(sys.argv[5]) if len(sys.argv) > 5 else 60
    out, code = taiji_exec_host(task_flag, instance_id, host_idx, command, timeout)
    print(out)
    sys.exit(code)
