using System;

namespace Microsoft.Win32 {
    public class WinUIModel : IDisposable, IChromeUIModel {
        object _WinUILock = new object();
        IntPtr[] _WinUIHandles = null;

        public WinUIModel() {
        }

        ~WinUIModel() {
            Dispose(false);
        }

        public bool IsDisposed { get; internal set; }

        public void Dispose() {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing) {
            IsDisposed = true;
            if (disposing) {
                CloseWinUIClients();
            }
        }

        protected void PostWinUIMessage() {
            lock (_WinUILock) {
                if (_WinUIHandles == null) {
                    return;
                }
                foreach (IntPtr hWnd in _WinUIHandles) {
                    if (hWnd != IntPtr.Zero) {
                        User32.PostMessage(hWnd, WM.WINMM,
                            IntPtr.Zero,
                            IntPtr.Zero);
                    }
                }
            }
        }

        public void CloseWinUIClients() {
            lock (_WinUILock) {
                if (_WinUIHandles == null) {
                    return;
                }
                foreach (IntPtr hWnd in _WinUIHandles) {
                    if (hWnd != IntPtr.Zero) {
                        User32.PostMessage(hWnd, WM.CLOSE,
                            IntPtr.Zero,
                            IntPtr.Zero);
                    }
                }
            }
        }

        public void AddWinUIClient(IntPtr hWnd) {
            if (hWnd == IntPtr.Zero) return;
            lock (_WinUILock) {
                if (_WinUIHandles == null) {
                    _WinUIHandles = new IntPtr[0];
                }
                for (int i = 0; i < _WinUIHandles.Length; i++) {
                    if (_WinUIHandles[i] == hWnd) {
                        return;
                    }
                }
                for (int i = 0; i < _WinUIHandles.Length; i++) {
                    if (_WinUIHandles[i] == IntPtr.Zero) {
                        _WinUIHandles[i] = hWnd;
                        return;
                    }
                }
                Array.Resize(ref _WinUIHandles,
                    _WinUIHandles.Length + 1);
                _WinUIHandles[_WinUIHandles.Length - 1] = hWnd;
            }
        }

        public void RemoveWinUIClient(IntPtr hWnd) {
            if (hWnd == IntPtr.Zero) return;
            lock (_WinUILock) {
                if (_WinUIHandles != null) {
                    for (int i = 0; i < _WinUIHandles.Length; i++) {
                        if (_WinUIHandles[i] == hWnd) {
                            _WinUIHandles[i] = IntPtr.Zero;
                        }
                    }
                }
            }
        }
    }
}