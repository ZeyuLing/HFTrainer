#!/usr/bin/env python3
"""Execute a command on a Taiji container via taiji_client exec with PTY emulation.

taiji_client exec requires a TTY, which is unavailable in non-interactive
environments (e.g. Claude Code autorun). This script uses Python's pty module
to provide one, and passes the command via `bash -c` for clean output.

Usage:
    python3 tools/taiji_exec.py <task_flag> <instance_id> <command> [timeout_seconds]

Examples:
    python3 tools/taiji_exec.py lzy_debug_machine_1 39a4d1Ee... "hostname"
    python3 tools/taiji_exec.py lzy_debug_machine_1 39a4d1Ee... "ls /workspace" 30
    python3 tools/taiji_exec.py lzy_debug_machine_1 39a4d1Ee... "cd /workspace && python3 eval.py" 300
"""
import subprocess, os, pty, time, select, sys, re


def taiji_exec(task_flag, instance_id, command, timeout=60):
    """Run a command on a Taiji container and return (stdout_text, exit_code)."""
    master, slave = pty.openpty()
    # Pass command via bash -c so we don't need interactive I/O
    proc = subprocess.Popen(
        ['taiji_client', 'exec', task_flag, instance_id,
         'bash', '-c', command + '; echo __EXIT_CODE__$?'],
        stdin=slave, stdout=slave, stderr=slave,
        close_fds=True
    )
    os.close(slave)

    output = b''
    start = time.time()
    while time.time() - start < timeout:
        r, _, _ = select.select([master], [], [], 1)
        if r:
            try:
                data = os.read(master, 8192)
                output += data
            except OSError:
                break
        if proc.poll() is not None:
            # Drain remaining
            while True:
                r, _, _ = select.select([master], [], [], 0.3)
                if r:
                    try:
                        output += os.read(master, 8192)
                    except:
                        break
                else:
                    break
            break

    try:
        os.close(master)
    except:
        pass
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except:
            proc.kill()

    # Clean terminal escape sequences
    text = output.decode('utf-8', errors='replace')
    text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)    # ANSI CSI
    text = re.sub(r'\x1b\][^\x07]*\x07', '', text)         # OSC
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

    # Extract exit code from marker
    exit_code = -1
    lines = text.strip().split('\n')
    clean_lines = []
    for line in lines:
        if '__EXIT_CODE__' in line:
            try:
                exit_code = int(line.strip().split('__EXIT_CODE__')[-1])
            except:
                exit_code = 0
        else:
            clean_lines.append(line)

    result = '\n'.join(clean_lines).strip()
    return result, exit_code


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    task_flag = sys.argv[1]
    instance_id = sys.argv[2]
    command = sys.argv[3]
    timeout = int(sys.argv[4]) if len(sys.argv) > 4 else 60

    result, code = taiji_exec(task_flag, instance_id, command, timeout)
    print(result)
    sys.exit(code)
