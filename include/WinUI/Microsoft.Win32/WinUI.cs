#define _NO_DRAW_DEBUG_FRAME_

using System;
using System.ComponentModel;
using System.Configuration;
using System.Diagnostics;
using System.Drawing;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Threading;

namespace Microsoft.Win32 {
    using static Microsoft.Win32.User32;

    [SupportedOSPlatform("windows")]
    unsafe public class WinUI : IDisposable, IWinUI {
        public static ITheme GetTheme()
        {
            string themeName = ConfigurationManager.AppSettings["Theme"];
            if (string.IsNullOrWhiteSpace(themeName))
            {
                return new Dark();
            }
            switch (themeName)
            {
                case "DARK":
                case "Dark":
                case "dark":
                    return new Dark();
                case "WHITE":
                case "White":
                case "white":
                    return new White();
            }
            throw new ConfigurationErrorsException($"Invalid theme: {themeName}");
        }
        public static void StartWinUI(string text, IWinUIController controller) => StartWinUI(text, controller, () => GetTheme());
        public static void StartWinUI(string text, IWinUIController controller, Func<ITheme> theme) {
            Thread t = new Thread(() => {
                GCHandle hWinUI = GCHandle.Alloc(new WinUI(controller, text, theme()), GCHandleType.Normal);
                try {
                    ((IWinUI)hWinUI.Target).Show();
                    while (!controller.IsDisposed && GetMessage(out MSG msg, ((IWinUI)hWinUI.Target).Handle, 0, 0) != 0) {
                        TranslateMessage(ref msg);
                        DispatchMessage(ref msg);
                    }
                } catch (Exception e) {
                    Console.Error?.WriteLine(e);
                    throw;
                } finally {
                    ((IWinUI)hWinUI.Target).Dispose();
                    hWinUI.Free();
                    WinMM.PlaySound(null,
                            IntPtr.Zero,
                            WinMM.PLAYSOUNDFLAGS.SND_ASYNC |
                            WinMM.PLAYSOUNDFLAGS.SND_FILENAME |
                            WinMM.PLAYSOUNDFLAGS.SND_NODEFAULT |
                            WinMM.PLAYSOUNDFLAGS.SND_NOWAIT |
                            WinMM.PLAYSOUNDFLAGS.SND_PURGE);
                }
            });
            t.Start();
        }

        string _text;
        Icon _icon;
        IntPtr _hWnd = IntPtr.Zero;
        IntPtr _hButton = IntPtr.Zero;
        WNDCLASSEX _lpwcx;
        int _cyTopHeight;
        GCHandle _lpfnWndProcPtr;
        IWinUIController _controller;
        ITheme _theme;

        public WinUI(IWinUIController controller, string text, ITheme theme) {
            _text = text;
            _controller = controller;
            _theme = theme;
            _icon = Icon.ExtractAssociatedIcon(Assembly.GetEntryAssembly().Location);
            if (!SetProcessDpiAwarenessContext(DpiAwarenessContext.PerMonitorAwareV2)) {
                Debug.WriteLine("WARNING: SetProcessDpiAwarenessContext failed.");
            }
            _lpfnWndProcPtr = GCHandle.Alloc(new WndProc(LocalWndProc));
            IntPtr lpfnWndProcPtr = Marshal.GetFunctionPointerForDelegate(_lpfnWndProcPtr.Target);
            _lpwcx = new WNDCLASSEX();
            _lpwcx.cbSize = Marshal.SizeOf(typeof(WNDCLASSEX));
            _lpwcx.hInstance = GetModuleHandle(null);
             _lpwcx.hIcon = _icon.Handle;
            _lpwcx.style =
                ClassStyles.HorizontalRedraw |
                ClassStyles.VerticalRedraw |
                ClassStyles.OwnDC;
            _lpwcx.style &= ~ClassStyles.DropShadow;
            _lpwcx.cbClsExtra = 0;
            _lpwcx.cbWndExtra = 0;
            _lpwcx.hCursor = LoadCursorW(IntPtr.Zero, (int)Constants.IDC_ARROW);
            _lpwcx.lpfnWndProc = lpfnWndProcPtr;
            _lpwcx.lpszClassName = "W_" + Guid.NewGuid().ToString("N");
            if (RegisterClassExW(ref _lpwcx) == 0) {
                if (1410 != Marshal.GetLastWin32Error()) {
                    throw new Win32Exception(Marshal.GetLastWin32Error());
                }
            }
            int width = _controller.DefaultWidth >= 0 ? _controller.DefaultWidth : 720;
            int height = _controller.DefaultHeight >= 0 ? _controller.DefaultHeight : (int)(width * (3f / 5f));
            const int SM_CXFULLSCREEN = 16;
            const int SM_CYFULLSCREEN = 17;
            int left = GetSystemMetrics(SM_CXFULLSCREEN) / 2 - width / 2;
            int top = GetSystemMetrics(SM_CYFULLSCREEN) / 2 - height / 2;
            const int CW_USEDEFAULT = unchecked((int)0x80000000);

            _hWnd = CreateWindowExW(
                    // WindowStylesEx.WS_EX_WINDOWEDGE |
                    WindowStylesEx.WS_EX_LEFT,
                    _lpwcx.lpszClassName,
                    _text,
                    WindowStyles.WS_SYSMENU |
                    // WindowStyles.WS_BORDER |
                    WindowStyles.WS_OVERLAPPED |
                    WindowStyles.WS_CAPTION |
                    WindowStyles.WS_SIZEFRAME |
                    WindowStyles.WS_MINIMIZEBOX |
                    WindowStyles.WS_MAXIMIZEBOX,
                left,
                top - 50,
                width,
                height,
                IntPtr.Zero,
                IntPtr.Zero,
                _lpwcx.hInstance,
                IntPtr.Zero);

            if (_hWnd == IntPtr.Zero) {
                throw new Win32Exception(Marshal.GetLastWin32Error());
            }

            const int DWMWA_WINDOW_CORNER_PREFERENCE = 33;
            var attribute = DWMWA_WINDOW_CORNER_PREFERENCE;
            int preference = (int)DWM_WINDOW_CORNER_PREFERENCE.DWMWCP_ROUND;
            var res = DwmSetWindowAttribute(_hWnd, attribute, &preference, sizeof(uint));

            preference = (int)DWMNCRENDERINGPOLICY.DWMNCRP_USEWINDOWSTYLE;
            DwmSetWindowAttribute(_hWnd,
                (int)DWMWINDOWATTRIBUTE.DWMWA_NCRENDERING_POLICY, &preference, sizeof(uint));

            // int color = 0x121212;
            // byte R = theme.GetColor(ThemeColor.B).R;
            // byte G = theme.GetColor(ThemeColor.B).G;
            // byte B = theme.GetColor(ThemeColor.B).B;
            // color = (R << 16) + (G << 8) + (B);
            // DwmSetWindowAttribute(_hWnd, (int)DWMWINDOWATTRIBUTE.DWMWA_BORDER_COLOR, &color, sizeof(uint));

            // UseImmersiveDarkMode(_hWnd, false);
        }

        int _DWMWA_VISIBLE_FRAME_BORDER_THICKNESS = 0;

        void Dispose(bool disposing) {
            if (_lpfnWndProcPtr.IsAllocated) {
                _lpfnWndProcPtr.Free();
            }
            IntPtr hWnd = Interlocked.Exchange(ref _hWnd, IntPtr.Zero);
            if (hWnd != IntPtr.Zero) {
                ShowWindow(hWnd, ShowWindowCommands.Hide);
                DestroyWindow(hWnd);
            }
            Icon icon = Interlocked.Exchange(ref _icon, null);
            if (icon != null) {
                icon.Dispose();
            }
        }

        ~WinUI() {
            Dispose(false);
        }

        public void Dispose() {
            GC.SuppressFinalize(this);
            Dispose(true);
        }

        private const int DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1 = 19;
        private const int DWMWA_USE_IMMERSIVE_DARK_MODE = 20;

        private static bool UseImmersiveDarkMode(IntPtr hWnd, bool enabled) {
            if (IsWindows10OrGreater(17763)) {
                var attribute = DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1;
                if (IsWindows10OrGreater(18985)) {
                    attribute = DWMWA_USE_IMMERSIVE_DARK_MODE;
                }
                int useImmersiveDarkMode = enabled ? 1 : 0;
                return DwmSetWindowAttribute(hWnd, (int)attribute, &useImmersiveDarkMode, sizeof(int)) == 0;
            }
            return false;
        }

        private static bool IsWindows10OrGreater(int build = -1) {
            return Environment.OSVersion.Version.Major >= 10 && Environment.OSVersion.Version.Build >= build;
        }

        private static bool IsWindows11() {
            return Environment.OSVersion.Version.Major >= 10 && Environment.OSVersion.Version.Build >= 22000;
        }


        internal enum TRACKMOUSEEVENT_FLAGS : uint {
            TME_CANCEL = 0x80000000,
            TME_HOVER = 0x00000001,
            TME_LEAVE = 0x00000002,
            TME_NONCLIENT = 0x00000010,
            TME_QUERY = 0x40000000,
        }

        internal struct TRACKMOUSEEVENT {
            internal uint cbSize;
            internal TRACKMOUSEEVENT_FLAGS dwFlags;
            internal IntPtr hwndTrack;
            internal uint dwHoverTime;
        }

        [DllImport("user32.dll", ExactSpelling = true, SetLastError = true)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.System32)]
        internal static extern unsafe bool TrackMouseEvent(TRACKMOUSEEVENT* lpEventTrack);

        public enum Cursors {
            IDC_ARROW = 32512,
            IDC_IBEAM = 32513,
            IDC_WAIT = 32514,
            IDC_CROSS = 32515,
            IDC_UPARROW = 32516,
            IDC_SIZE = 32640,
            IDC_ICON = 32641,
            IDC_SIZENWSE = 32642,
            IDC_SIZENESW = 32643,
            IDC_SIZEWE = 32644,
            IDC_SIZENS = 32645,
            IDC_SIZEALL = 32646,
            IDC_NO = 32648,
            IDC_HAND = 32649,
            IDC_APPSTARTING = 32650,
            IDC_HELP = 32651
        }

        [DllImport("user32.dll")]
        public static extern IntPtr LoadCursorW(IntPtr hInstance, int lpCursorName);

        [DllImport("user32.dll")]
        public static extern IntPtr SetCursor(IntPtr hCursor);

        static void Invalidate(nint hWnd) {
            GetClientRect(hWnd, out RECT lprctw);
            InvalidateRect(hWnd, ref lprctw, false);
        }

        [Flags]
        public enum WINDOWBUTTON : int {
            None = 0,
            IsMouseDown = 2,
            Reserved = 4,
            Icon = 8,
            Close = 16,
            Zoom = 32,
            Restore = 64,
        }

        IntPtr LocalWndProc(IntPtr hWnd, WM uMsg, IntPtr wParam, IntPtr lParam) {
            Console.WriteLine(uMsg);

            switch (uMsg) {
                case WM.SHOWWINDOW:
                    _controller?.OnShow(this);
                    break;
                case WM.CLOSE:
                    SetCursor(LoadCursorW(IntPtr.Zero, (int)Cursors.IDC_ARROW));
                    _controller?.OnClose(this);
                    break;
                case WM.DESTROY:
                    Dispose();
                    PostQuitMessage(0);
                    return IntPtr.Zero;
                case WM.WINMM:
                    Invalidate(hWnd);
                    return IntPtr.Zero;
                case WM.SETTINGCHANGE:
                case WM.DPICHANGED:
                case WM.CREATE: {
                        _cyTopHeight = (int)((float)48 * GetDpiForWindow(hWnd) / 96);
                        uint value = 0;
                        int hSuccess = DwmGetWindowAttribute(hWnd,
                            (int)DWMWINDOWATTRIBUTE.DWMWA_VISIBLE_FRAME_BORDER_THICKNESS, &value, sizeof(uint));
                        if (hSuccess == 0) {
                            _DWMWA_VISIBLE_FRAME_BORDER_THICKNESS = (int)value;
                        }
                        RECT rcWnd;
                        GetWindowRect(hWnd, out rcWnd);
                        // Inform the application of the frame change to force redrawing with the new
                        // client area that is extended into the title bar
                        SetWindowPos(
                          hWnd, IntPtr.Zero,
                          rcWnd.Left, rcWnd.Top,
                          rcWnd.Right - rcWnd.Left, rcWnd.Bottom - rcWnd.Top,
                                SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE
                        );
                        RedrawWindow(hWnd, IntPtr.Zero, IntPtr.Zero, RDW_FRAME | RDW_INVALIDATE);
                        break;
                    }
                case WM.ACTIVATE:
                    break;
                case WM.ERASEBKGND:
                    return IntPtr.Zero;
                case WM.NCLBUTTONUP: {
                        int xScreen = (lParam.ToInt32()) & 0xffff;
                        int yScreen = (lParam.ToInt32() >> 16) & 0xffff;
                        int ht = TrackMouseEx(hWnd, xScreen, yScreen, false);
                        switch (ht) {
                            case HTCLOSE:
                                PostMessage(hWnd, WM.CLOSE, IntPtr.Zero, IntPtr.Zero);
                                return IntPtr.Zero;
                            case HTZOOM:
                                return IntPtr.Zero;
                        }
                        break;
                    }
                case WM.NCLBUTTONDOWN: {
                        int xScreen = (lParam.ToInt32()) & 0xffff;
                        int yScreen = (lParam.ToInt32() >> 16) & 0xffff;
                        int ht = TrackMouseEx(hWnd, xScreen, yScreen, true);
                        if (ht == HTCLOSE || ht == HTZOOM) {
                            return IntPtr.Zero;
                        }
                        break;
                    }
                case WM.NCMOUSELEAVE: {
                        TrackMouseEx(hWnd, -1, -1, false);
                        break;
                }
                case WM.NCMOUSEMOVE: {
                        var lpEventTrack = new TRACKMOUSEEVENT {
                            cbSize = (uint)Marshal.SizeOf<TRACKMOUSEEVENT>(),
                            dwFlags = TRACKMOUSEEVENT_FLAGS.TME_LEAVE | TRACKMOUSEEVENT_FLAGS.TME_NONCLIENT,
                            hwndTrack = hWnd
                        };
                        TrackMouseEvent(&lpEventTrack);
                        int xScreen = (lParam.ToInt32()) & 0xffff;
                        int yScreen = (lParam.ToInt32() >> 16) & 0xffff;
                        int ht = TrackMouseEx(hWnd, xScreen, yScreen, null);
                        if (ht == HTCLOSE || ht == HTZOOM) {
                            return IntPtr.Zero;
                        }
                        break;
                    }
                case WM.NCHITTEST: {
                    int x = (lParam.ToInt32()) & 0xffff;
                    int y = (lParam.ToInt32() >> 16) & 0xffff;
                    return CalcHitTest(hWnd, x, y);
                  }
                case WM.NCACTIVATE: {
                        return DefWindowProc(hWnd, WM.NCACTIVATE, wParam, new IntPtr(-1));
                    }
                case WM.NCCALCSIZE: {
                        var wParamIsTrue = wParam != 0;
                        RECT* rect0 = (RECT*)lParam;
                        var top = rect0->Top;
                        var bottom = rect0->Bottom;
                        DefWindowProc(hWnd, uMsg, wParam, lParam);
                        rect0->Top = top;
                        if (IsWindows11()) {
                            rect0->Top += (int)_DWMWA_VISIBLE_FRAME_BORDER_THICKNESS; // Windows 11 top border bug. When the window is inactive.
                        }
                        var dpi = GetDpiForWindow(hWnd);
                        if (IsMaximized(hWnd)) {
                            rect0->Top -=
                                + GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi)
                                + GetSystemMetricsForDpi(SM_CXSIZEFRAME, dpi)
                                + GetSystemMetricsForDpi(SM_CXBORDER, dpi);
                        }
                        const int WVR_REDRAW = 0x0300;
                        const int WVR_VALIDRECTS = 0x0400;
                        var retVal = IntPtr.Zero;
                        if (wParamIsTrue) {
                            retVal = new IntPtr((WVR_REDRAW | WVR_VALIDRECTS));
                        }
                        return retVal;
                    }
                case WM.NCPAINT: {
                        var retVal= DefWindowProc(hWnd, (WM)uMsg, wParam, lParam);
                        const IntPtr HRGN_FULL = 0x1;
                        IntPtr hRgn = wParam;
                        if (hRgn == HRGN_FULL)
                            hRgn = IntPtr.Zero;
                        DeviceContextValues flags = DeviceContextValues.DCX_WINDOW
                            | DeviceContextValues.DCX_LOCKWINDOWUPDATE | DeviceContextValues.DCX_USESTYLE
                            | DeviceContextValues.DCX_CLIPSIBLINGS
                            | DeviceContextValues.DCX_CLIPCHILDREN;
                        if (hRgn != IntPtr.Zero)
                            flags |= DeviceContextValues.DCX_INTERSECTRGN | DeviceContextValues.DCX_NODELETERGN;
                        var hdc0 = GetDCEx(hWnd, hRgn, flags);
                        if (hdc0 != IntPtr.Zero) {
                            try {
                                PaintNonClientArea(hWnd, hdc0);
                            } finally {
                                ReleaseDC(hWnd, hdc0);
                            }
                        }
                        return retVal;
                    }
                case WM.PAINT: {
                        IntPtr hdc0 = BeginPaint(hWnd, out PAINTSTRUCT ps);
                        try {
                            PaintClientArea(hWnd, hdc0);
                        } finally {
                            EndPaint(hWnd, ref ps);
                        }
                        return IntPtr.Zero;
                    }
            }
            return DefWindowProc(hWnd, (WM)uMsg, wParam, lParam);
        }

        int TrackMouseEx(IntPtr hWnd, int xMouse, int yMouse, bool? isMouseDown) {
            var ht = CalcHitTest(hWnd, xMouse, yMouse);

            WINDOWBUTTON prev = (WINDOWBUTTON)GetWindowLongPtr(hWnd, (int)WindowLongFlags.GWL_USERDATA);

            WINDOWBUTTON wp = WINDOWBUTTON.None;

            if (isMouseDown.HasValue) {
                if (isMouseDown.Value) {
                    wp |= WINDOWBUTTON.IsMouseDown;
                }
            } else {
                if ((prev & WINDOWBUTTON.IsMouseDown) == WINDOWBUTTON.IsMouseDown) {
                    wp |= WINDOWBUTTON.IsMouseDown;
                }
            }

            if (ht == HTCLOSE) {
                wp |= WINDOWBUTTON.Close;
            }

            if (wp != prev) {
                SetWindowLongPtr(
                    hWnd, (int)WindowLongFlags.GWL_USERDATA, (IntPtr)(int)wp);

                GetWindowRect(hWnd, out RECT rcWnd0);

                rcWnd0.Offset(
                    -rcWnd0.Left,
                    -rcWnd0.Top
                );

                rcWnd0.Bottom = rcWnd0.Top + _cyTopHeight;

                var hrgn = Gdi32.CreateRectRgn(
                    rcWnd0.Left,
                    rcWnd0.Top,
                    rcWnd0.Right,
                    rcWnd0.Bottom);

                RedrawWindow(hWnd,
                    IntPtr.Zero, hrgn, RDW_FRAME | RDW_INVALIDATE);

                Gdi32.DeleteObject(hrgn);
            }

            return ht;
        }

        void PaintNonClientArea(nint hWnd, nint hdc0) {
            GetWindowRect(hWnd, out var rcWnd0);

            rcWnd0.Offset(
                -rcWnd0.Left,
                -rcWnd0.Top
            );

            ExcludeClipRect(hdc0,
                rcWnd0.Left,
                rcWnd0.Top + _cyTopHeight,
                rcWnd0.Right,
                rcWnd0.Bottom);

            SetDCBrushColor(hdc0, ColorTranslator.ToWin32(_theme.GetColor(ThemeColor.TitleBar)));

            FillRect(hdc0, ref rcWnd0, GetStockObject(StockObjects.DC_BRUSH));

            var dpi = GetDpiForWindow(hWnd);

            var rcCaption = rcWnd0;

            rcCaption.Bottom = _cyTopHeight;

            rcCaption.Left = rcCaption.Left
                    + GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi)
                    + GetSystemMetricsForDpi(SM_CXSIZEFRAME, dpi)
                    + GetSystemMetricsForDpi(SM_CXBORDER, dpi);

            rcCaption.Right = rcCaption.Right
                    - GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi)
                    - GetSystemMetricsForDpi(SM_CXSIZEFRAME, dpi)
                    - GetSystemMetricsForDpi(SM_CXBORDER, dpi);

            if (IsMaximized(hWnd)) {
                rcCaption.Top -= 
                          GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi)
                        - GetSystemMetricsForDpi(SM_CXSIZEFRAME, dpi);
            }

            SetBkColor(hdc0, ColorTranslator.ToWin32(_theme.GetColor(ThemeColor.TitleBar)));
            SetTextColor(hdc0, ColorTranslator.ToWin32(_theme.GetColor(ThemeColor.TitleText)));

            Gdi32.LOGFONTW lplf = new Gdi32.LOGFONTW();

            lplf.lfCharSet = Gdi32.FontCharSet.DEFAULT_CHARSET;
            lplf.lfWeight = Gdi32.FontWeight.FW_MEDIUM;
            lplf.lfHeight = -
                Gdi32.MulDiv(8, Gdi32.GetDeviceCaps(hdc0, Gdi32.LOGPIXELSY), 72);

            lplf.lfHeight = (int)((float)lplf.lfHeight * GetDpiForWindow(hWnd) / 96);

            lplf.lfFaceName = "Segoe UI";

            var rcText = rcCaption;

            rcText.Left += Math.Abs(lplf.lfHeight);
            rcText.Right -= Math.Abs(lplf.lfHeight) + _cyTopHeight;
            rcText.Top += _DWMWA_VISIBLE_FRAME_BORDER_THICKNESS;

            if (IsMaximized(hWnd)) {
                rcText.Top +=
                        GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi)
                        - _DWMWA_VISIBLE_FRAME_BORDER_THICKNESS;
                rcText.Left -= Math.Abs(lplf.lfHeight) / 2;
                rcText.Bottom -= GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi)
                    + GetSystemMetricsForDpi(SM_CXSIZEFRAME, dpi)
                    + _DWMWA_VISIBLE_FRAME_BORDER_THICKNESS;
            }

#if _DRAW_DEBUG_FRAME_
            var rcDbg = rcText;
            SetDCBrushColor(hdc0, ColorTranslator.ToWin32(_theme.GetColor(ThemeColor.B)));
            FrameRect(hdc0, ref rcDbg, GetStockObject(StockObjects.DC_BRUSH));
#endif

            var hSegoeUI = Gdi32.CreateFontIndirectW(lplf);

            SelectObject(hdc0, hSegoeUI);

            DrawTextW(
                hdc0,
                _text,
                -1,
                ref rcText,
                DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS);

            Gdi32.DeleteObject(hSegoeUI);

            var rcChromeClose = rcWnd0;

            rcChromeClose.Bottom = _cyTopHeight;

            rcChromeClose.Left = rcChromeClose.Left
                    + GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi)
                    + GetSystemMetricsForDpi(SM_CXSIZEFRAME, dpi);

            rcChromeClose.Right = rcChromeClose.Right
                    - GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi)
                    - GetSystemMetricsForDpi(SM_CXSIZEFRAME, dpi);

            rcChromeClose.Left = rcChromeClose.Right - _cyTopHeight;

            if (IsMaximized(hWnd)) {
                rcChromeClose.Top -=
                    GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi) - GetSystemMetricsForDpi(SM_CXSIZEFRAME, dpi);
            }

            GetCursorPos(out var pt);

            var ht = CalcHitTest(hWnd, pt.x, pt.y);

            WINDOWBUTTON state = (WINDOWBUTTON)GetWindowLongPtr(hWnd, (int)WindowLongFlags.GWL_USERDATA);

            if (ht == HTCLOSE && ((state & WINDOWBUTTON.Close) == WINDOWBUTTON.Close)) {
                if (((state & WINDOWBUTTON.IsMouseDown) == WINDOWBUTTON.IsMouseDown)) {
                    SetTextColor(hdc0, 0x00B2B2B2);
                    SetDCBrushColor(hdc0, ColorTranslator.ToWin32(_theme.GetColor(ThemeColor.ChromeClosePressed)));
                    FillRect(hdc0, ref rcChromeClose, GetStockObject(StockObjects.DC_BRUSH));
                    SetBkColor(hdc0, ColorTranslator.ToWin32(_theme.GetColor(ThemeColor.ChromeClosePressed)));
                } else {
                    SetTextColor(hdc0, 0x00FFFFFF);
                    SetDCBrushColor(hdc0, ColorTranslator.ToWin32(_theme.GetColor(ThemeColor.ChromeClose)));
                    FillRect(hdc0, ref rcChromeClose, GetStockObject(StockObjects.DC_BRUSH));
                    SetBkColor(hdc0, ColorTranslator.ToWin32(_theme.GetColor(ThemeColor.ChromeClose)));
                }
            } else {
            }

            lplf = new Gdi32.LOGFONTW();

            lplf.lfCharSet = Gdi32.FontCharSet.DEFAULT_CHARSET;
            lplf.lfHeight = -Gdi32.MulDiv(
                (int)((float)7 * GetDpiForWindow(hWnd) / 96), Gdi32.GetDeviceCaps(hdc0, Gdi32.LOGPIXELSY), 72);
            lplf.lfFaceName = "Segoe Fluent Icons";

            var hSegoeIcons = Gdi32.CreateFontIndirectW(lplf);

            SelectObject(hdc0, hSegoeIcons);

            DrawTextW(
                hdc0,
                "\U0000e8bb",
                -1,
                ref rcChromeClose,
                DT_CENTER | DT_VCENTER | DT_SINGLELINE);

#if _DRAW_DEBUG_FRAME_
            rcDbg = rcChromeClose;
            rcDbg.Top += 1;
            SetDCBrushColor(hdc0, ColorTranslator.ToWin32(_theme.GetColor(ThemeColor.C)));
            FrameRect(hdc0, ref rcDbg, GetStockObject(StockObjects.DC_BRUSH));
#endif

            Gdi32.DeleteObject(hSegoeIcons);
        }

        private void PaintClientArea(nint hWnd, nint hdc0) {
            GetClientRect(hWnd, out var rcClient);
            rcClient.Bottom -= _cyTopHeight - _DWMWA_VISIBLE_FRAME_BORDER_THICKNESS;
            var hBmp = CreateCompatibleBitmap(
                hdc0,
                rcClient.Width,
                rcClient.Height);
            var hMemDC = CreateCompatibleDC(hdc0);
            SelectObject(hMemDC, hBmp);
            var hdcGraphics = Graphics.FromHdc(hMemDC);
            try {
                RectangleF rectF = new RectangleF(
                    rcClient.Left,
                    rcClient.Top,
                    rcClient.Width,
                    rcClient.Height);
                hdcGraphics.FillRectangle(_theme.GetBrush(ThemeColor.Background), rectF);
#if _DRAW_DEBUG_FRAME_
                hdcGraphics.DrawRectangle(
                    _theme.GetPen(ThemeColor.A),
                    rectF.Left,
                    rectF.Top,
                    rectF.Width - 1,
                    rectF.Height - 1);
#else
                _controller?.OnPaint(this,
                    hdcGraphics,
                    rectF);
#endif
                BitBlt(hdc0,
                     0,
                     _cyTopHeight - _DWMWA_VISIBLE_FRAME_BORDER_THICKNESS,
                     rcClient.Width,
                     rcClient.Height,
                     hMemDC,
                     0,
                     0,
                     TernaryRasterOperations.SRCCOPY);
            } finally {
                hdcGraphics.Dispose();
            }
            DeleteObject(hBmp);
            DeleteDC(hMemDC);
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct WINDOWPLACEMENT {
            public int length;
            public uint flags;
            public uint showCmd;
            public POINT ptMinPosition;
            public POINT ptMaxPosition;
            public RECT rcNormalPosition;
        }

        const int SW_HIDE = 0;
        const int SW_SHOWNORMAL = 1;
        const int SW_NORMAL = 1;
        const int SW_SHOWMINIMIZED = 2;
        const int SW_SHOWMAXIMIZED = 3;
        const int SW_MAXIMIZE = 3;
        const int SW_SHOWNOACTIVATE = 4;
        const int SW_SHOW = 5;
        const int SW_MINIMIZE = 6;
        const int SW_SHOWMINNOACTIVE = 7;
        const int SW_SHOWNA = 8;
        const int SW_RESTORE = 9;
        const int SW_SHOWDEFAULT = 10;
        const int SW_FORCEMINIMIZE = 11;
        const int SW_MAX = 11;

        [DllImport("user32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool GetWindowPlacement(
              IntPtr hWnd,
              ref WINDOWPLACEMENT lpwndpl);

        [DllImport("user32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool IsWindowArranged(
              IntPtr hWnd);

        static bool IsMaximized(IntPtr hWnd) {
            WINDOWPLACEMENT placement = new WINDOWPLACEMENT();
            placement.length = Marshal.SizeOf<WINDOWPLACEMENT>();
            if (GetWindowPlacement(hWnd, ref placement)) {
                return placement.showCmd == SW_SHOWMAXIMIZED;
            }
            return false;
        }

        int CalcHitTest(nint hWnd, int xScreen, int yScreen) {
            GetWindowRect(hWnd, out var rcWnd0);

            var dpi = GetDpiForWindow(hWnd);

            int left = 
                    GetSystemMetricsForDpi(SM_CXSIZEFRAME, dpi) 
                +   GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi)
                +   GetSystemMetricsForDpi(SM_CXBORDER, dpi);

            int right = 
                    GetSystemMetricsForDpi(SM_CXSIZEFRAME, dpi) 
                +   GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi)
                +   GetSystemMetricsForDpi(SM_CXBORDER, dpi);

            int top = GetSystemMetricsForDpi(SM_CYSIZEFRAME, dpi);

            int bottom =
                    GetSystemMetricsForDpi(SM_CYSIZEFRAME, dpi)
                +   GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi)
                +   GetSystemMetricsForDpi(SM_CYBORDER, dpi);

            if (xScreen <= rcWnd0.Left + left && yScreen <= rcWnd0.Top + top) {
                return HTTOPLEFT;
            } else if (xScreen >= rcWnd0.Right - right && yScreen <= rcWnd0.Top + top) {
                return HTTOPRIGHT;
            } else if (yScreen >= rcWnd0.Bottom - bottom && xScreen >= rcWnd0.Right - right) {
                return HTBOTTOMRIGHT;
            } else if (xScreen <= rcWnd0.Left + left && yScreen >= rcWnd0.Bottom - bottom) {
                return HTBOTTOMLEFT;
            } else if (xScreen <= rcWnd0.Left + left) {
                return HTLEFT;
            } else if (yScreen <= rcWnd0.Top + top) {
                return HTTOP;
            } else if (xScreen >= rcWnd0.Right - right) {
                return HTRIGHT;
            } else if (yScreen >= rcWnd0.Bottom - bottom) {
                return HTBOTTOM;
            }

            int cy = _cyTopHeight;
            int cx = _cyTopHeight;

            cx += GetSystemMetricsForDpi(SM_CXSIZEFRAME, dpi)
                + GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi)
                + GetSystemMetricsForDpi(SM_CXBORDER, dpi);

            if (IsMaximized(hWnd)) {
                cy -=
                    GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi) - GetSystemMetricsForDpi(SM_CXSIZEFRAME, dpi);
            }

            if (xScreen >= rcWnd0.Right - cx && yScreen <= rcWnd0.Top + cy) {
                return HTCLOSE;
            }

            return HTCAPTION;
        }

        public IntPtr Handle {
            get {
                return _hWnd;
            }
        }

        ITheme IWinUI.Theme { get => _theme; }

        public void Show() {
            if (_hWnd == IntPtr.Zero) {
                throw new ObjectDisposedException(GetType().Name);
            }
            ShowWindow(_hWnd, ShowWindowCommands.Normal);
            UpdateWindow(_hWnd);
        }
    }
}