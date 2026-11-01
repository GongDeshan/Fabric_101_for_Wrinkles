"""Windows child lifetime containment; no CUDA imports."""
import ctypes as C
from ctypes import wintypes as W
import msvcrt
import subprocess

k = C.WinDLL('kernel32', use_last_error=True)
class BASIC(C.Structure):
    _fields_ = [('process_time', C.c_longlong), ('job_time', C.c_longlong),
                ('flags', W.DWORD), ('min_ws', C.c_size_t), ('max_ws', C.c_size_t),
                ('active', W.DWORD), ('affinity', C.c_size_t), ('priority', W.DWORD), ('scheduling', W.DWORD)]
class IO(C.Structure):
    _fields_ = [(n, C.c_ulonglong) for n in ('ro','wo','oo','rb','wb','ob')]
class EXTENDED(C.Structure):
    _fields_ = [('basic', BASIC), ('io', IO), ('process_memory', C.c_size_t),
                ('job_memory', C.c_size_t), ('peak_process', C.c_size_t), ('peak_job', C.c_size_t)]
k.CreateJobObjectW.argtypes = [C.c_void_p, W.LPCWSTR]
k.CreateJobObjectW.restype = W.HANDLE
k.SetInformationJobObject.argtypes = [W.HANDLE, C.c_int, C.c_void_p, W.DWORD]
k.AssignProcessToJobObject.argtypes = [W.HANDLE, W.HANDLE]
k.TerminateJobObject.argtypes = [W.HANDLE, W.UINT]
k.CloseHandle.argtypes = [W.HANDLE]

class Lifetime:
    def __init__(self, lock_path):
        self.lock = open(lock_path, 'a+b')
        self.lock.seek(0)
        if not self.lock.read(1):
            self.lock.write(b'0'); self.lock.flush()
        self.lock.seek(0)
        try:
            msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            self.lock.close()
            raise RuntimeError('Another safe training parent is already running')
        self.handle = k.CreateJobObjectW(None, None)
        if not self.handle:
            self.lock.close(); raise C.WinError(C.get_last_error())
        info = EXTENDED()
        info.basic.flags = 0x2000  
        if not k.SetInformationJobObject(self.handle, 9, C.byref(info), C.sizeof(info)):
            self.close(); raise C.WinError(C.get_last_error())

    def assign(self, process):
        if not k.AssignProcessToJobObject(self.handle, W.HANDLE(int(process._handle))):
            error = C.get_last_error()
            process.kill(); process.wait(timeout=10)
            raise C.WinError(error)

    def close(self):
        if self.handle:
            k.TerminateJobObject(self.handle, 1)
            k.CloseHandle(self.handle)
            self.handle = None
        self.lock.close()

def close_client(client):
    p = client.process
    try:
        if p.poll() is None:
            try:
                p.stdin.write('{"stop":true}\n'); p.stdin.flush()
                p.wait(timeout=5)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                p.kill(); p.wait(timeout=10)
    finally:
        if p.stdin:
            p.stdin.close()
        client.log.close()
