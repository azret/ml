using System;
using System.Drawing;

namespace Microsoft.Win32 {
    public interface IWinUI : IDisposable {
        IntPtr Handle { get; }
        bool IsHandleAllocated { get; }
        ITheme Theme { get; }
        void Show();
    }


    [Flags]
    public enum WinUIControllerOptions {
        None = 0,
        DisposeModelOnClose = 1,
        UnMuteModelOnOpen = 1 << 1,
        MuteOnSpaceBar = 1 << 2,
    }

    public static class WinUIControllerOptionsExtensions {
        public static bool IsDisposeModelOnClose(this WinUIControllerOptions options) => (options & WinUIControllerOptions.DisposeModelOnClose) == WinUIControllerOptions.DisposeModelOnClose;
        public static bool IsUnMuteModelOnOpen(this WinUIControllerOptions options) => (options & WinUIControllerOptions.UnMuteModelOnOpen) == WinUIControllerOptions.UnMuteModelOnOpen;
        public static bool IsMuteOnSpaceBar(this WinUIControllerOptions options) => (options & WinUIControllerOptions.MuteOnSpaceBar) == WinUIControllerOptions.MuteOnSpaceBar;
    }

    public interface IWinUIController {
        int DefaultWidth { get; }
        int DefaultHeight { get; }
        bool IsDisposed { get; }
        void OnShow(IWinUI winUI);
        void OnClose(IWinUI winUI);
        void OnKeyDown(IWinUI winUI, IntPtr wParam, IntPtr lParam);
        void OnPaint(IWinUI winUI, Bitmap hMemBitmap);
        void OnPaint(IWinUI winUI, Graphics g, RectangleF r);
    }

    public interface IWinUIModel {
        void AddWinUIClient(IntPtr hWnd);
        void RemoveWinUIClient(IntPtr hWnd);
        void CloseWinUIClients();
    }
}