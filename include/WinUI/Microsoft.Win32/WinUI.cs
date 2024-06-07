using System;
using System.ComponentModel;
using System.Configuration;
using System.Drawing;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;

namespace Microsoft.Win32 {
    unsafe public class WinUI : IDisposable, IWinUI {
        public static ITheme GetTheme()
        {
            string themeName = ConfigurationManager.AppSettings["Theme"];
            if (string.IsNullOrWhiteSpace(themeName))
            {
                return new Black();
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
                GCHandle hWnd = GCHandle.Alloc(new WinUI(controller, text, theme()), GCHandleType.Normal);
                try {
                    ((IWinUI)hWnd.Target).Show();
                    while (!controller.IsDisposed && User32.GetMessage(out MSG msg, ((IWinUI)hWnd.Target).Handle, 0, 0) != 0) {
                        User32.TranslateMessage(ref msg);
                        User32.DispatchMessage(ref msg);
                    }
                } catch (Exception e) {
                    Console.Error?.WriteLine(e);
                    throw;
                } finally {
                    ((IWinUI)hWnd.Target).Dispose();
                    hWnd.Free();
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

        Icon _icon;
        IntPtr _hWnd = IntPtr.Zero;
        WNDCLASSEX _lpwcx;
        WndProc _lpfnWndProcPtr;
        IWinUIController _controller;
        ITheme _theme;
        static IntPtr _defaultWindowProc;
        internal partial class Kernel32 {
            [DllImport("Kernel32", CharSet = CharSet.Unicode, SetLastError = true, ExactSpelling = true)]
            public static extern IntPtr GetModuleHandleW(string moduleName);
            [DllImport("Kernel32", CharSet = CharSet.Ansi, BestFitMapping = false, ExactSpelling = true)]
            public static extern IntPtr GetProcAddress(IntPtr hModule, string lpProcName);
        }

        [DllImport("uxtheme.dll", ExactSpelling = true, CharSet = CharSet.Unicode)]
        private static extern int SetWindowTheme(IntPtr hwnd, string pszSubAppName, string pszSubIdList);

        internal static IntPtr DefaultWindowProc {
            get {
                if (_defaultWindowProc == IntPtr.Zero) {
                    _defaultWindowProc = Kernel32.GetProcAddress(
                        Kernel32.GetModuleHandleW("User32"),
                        "DefWindowProcA");
                    if (_defaultWindowProc == IntPtr.Zero) {
                        throw new Win32Exception();
                    }
                }
                return _defaultWindowProc;
            }
        }
        public WinUI(IWinUIController controller, string text, ITheme theme) {
            _controller = controller;
            _theme = theme;
            _icon = Icon.ExtractAssociatedIcon(Assembly.GetEntryAssembly().Location);
            // var bmp = _icon.ToBitmap();
            // _icon.Dispose();
            // for (int i = 0; i < bmp.Width; i++) {
            //     for (int j = 0; j < bmp.Height; j++) {
            //         var p = bmp.GetPixel(i, j);
            //         if (p.R == 0 && p.G == 0 && p.B == 0 && p.A == 0) {
            //         } else if (p.A != 255) {
            //             Console.WriteLine(p);
            //             bmp.SetPixel(i, j, Color.White);
            //         }
            //     }
            // }
            // _icon = Icon.FromHandle(bmp.GetHicon());
            // bmp.Dispose();
            _lpwcx = new WNDCLASSEX();
            _lpwcx.cbSize = Marshal.SizeOf(typeof(WNDCLASSEX));
            _lpwcx.hInstance = User32.GetModuleHandle(null);
             _lpwcx.hIcon = _icon.Handle;
            _lpwcx.style = ClassStyles.HorizontalRedraw | ClassStyles.VerticalRedraw | ClassStyles.OwnDC ;
            _lpwcx.cbClsExtra = 0;
            _lpwcx.cbWndExtra = 0;
            _lpwcx.hCursor = User32.LoadCursor(IntPtr.Zero, (int)Constants.IDC_ARROW);
            _lpwcx.hbrBackground = User32.CreateSolidBrush(ColorTranslator.ToWin32(_theme.GetColor(ThemeColor.Background)));
            _lpwcx.lpfnWndProc = DefaultWindowProc;
            _lpwcx.lpszClassName = "W_" + Guid.NewGuid().ToString("N");
            if (User32.RegisterClassExA(ref _lpwcx) == 0) {
                if (1410 != Marshal.GetLastWin32Error()) {
                    throw new Win32Exception(Marshal.GetLastWin32Error());
                }
            }
            _hWnd = User32.CreateWindowExA(
                    WindowStylesEx.WS_EX_WINDOWEDGE |
                    WindowStylesEx.WS_EX_LEFT,
                    _lpwcx.lpszClassName,
                    text,
                    WindowStyles.WS_SYSMENU |
                    WindowStyles.WS_BORDER |
                    WindowStyles.WS_SIZEFRAME |
                    WindowStyles.WS_MINIMIZEBOX |
                    WindowStyles.WS_MAXIMIZEBOX,
                160,
                230,
                720,
                (int)(720 * (3f / 5f)),
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

            // attribute = (int)DWMWINDOWATTRIBUTE.DWMWA_NCRENDERING_POLICY;
            // preference = (int)DWMNCRENDERINGPOLICY.DWMNCRP_DISABLED;
            // res = DwmSetWindowAttribute(_hWnd, attribute, &preference, sizeof(uint));

            // attribute = (int)DWMWINDOWATTRIBUTE.DWMWA_ALLOW_NCPAINT;
            // preference = (int)0;
            // res = DwmSetWindowAttribute(_hWnd, attribute, &preference, sizeof(uint));

            attribute = (int)DWMWINDOWATTRIBUTE.DWMWA_BORDER_COLOR;
            preference = (int)0x121212;
            res = DwmSetWindowAttribute(_hWnd, attribute, &preference, sizeof(uint));

            const int DWMWA_CAPTION_COLOR = 35;
            attribute = DWMWA_CAPTION_COLOR;
            // byte R = theme.GetColor(ThemeColor.Background).R;
            // byte G = theme.GetColor(ThemeColor.Background).G;
            // byte B = theme.GetColor(ThemeColor.Background).B;
            // preference = (R << 16) + (G << 8) + (B);
            preference = 0x2E2E2E;
            res = DwmSetWindowAttribute(_hWnd, attribute, &preference, sizeof(uint));


            _lpfnWndProcPtr = LocalWndProc;
            IntPtr lpfnWndProcPtr = Marshal.GetFunctionPointerForDelegate(_lpfnWndProcPtr);
            if (User32.SetWindowLongPtr(_hWnd, (int)User32.WindowLongFlags.GWL_WNDPROC, lpfnWndProcPtr) == IntPtr.Zero) {
                var lastWin32Error = Marshal.GetLastWin32Error();
                User32.ShowWindow(_hWnd, ShowWindowCommands.Hide);
                User32.DestroyWindow(_hWnd);
                throw new Win32Exception(lastWin32Error);
            }
        }

        void Dispose(bool disposing) {
            IntPtr hWnd = Interlocked.Exchange(ref _hWnd, IntPtr.Zero);
            if (hWnd != IntPtr.Zero) {
                User32.ShowWindow(hWnd, ShowWindowCommands.Hide);
                User32.DestroyWindow(hWnd);
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

        [StructLayout(LayoutKind.Sequential)]
        struct NCCALCSIZE_PARAMS {
            public RECT rgrc0, rgrc1, rgrc2;
            public IntPtr lppos;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct WINDOWPOS {
            public IntPtr hwnd;
            public IntPtr hwndInsertAfter;
            public int x, y;
            public int cx, cy;
            public int flags;
        }

        enum DWMNCRENDERINGPOLICY : uint {
            DWMNCRP_USEWINDOWSTYLE,
            DWMNCRP_DISABLED,
            DWMNCRP_ENABLED,
            DWMNCRP_LAST
        };

        [Flags]
        public enum DWMWINDOWATTRIBUTE : uint {
            /// <summary>
            /// Use with DwmGetWindowAttribute. Discovers whether non-client rendering is enabled. The retrieved value is of type BOOL. TRUE if non-client rendering is enabled; otherwise, FALSE.
            /// </summary>
            DWMWA_NCRENDERING_ENABLED = 1,

            /// <summary>
            /// Use with DwmSetWindowAttribute. Sets the non-client rendering policy. The pvAttribute parameter points to a value from the DWMNCRENDERINGPOLICY enumeration.
            /// </summary>
            DWMWA_NCRENDERING_POLICY,

            /// <summary>
            /// Use with DwmSetWindowAttribute. Enables or forcibly disables DWM transitions. The pvAttribute parameter points to a value of type BOOL. TRUE to disable transitions, or FALSE to enable transitions.
            /// </summary>
            DWMWA_TRANSITIONS_FORCEDISABLED,

            /// <summary>
            /// Use with DwmSetWindowAttribute. Enables content rendered in the non-client area to be visible on the frame drawn by DWM. The pvAttribute parameter points to a value of type BOOL. TRUE to enable content rendered in the non-client area to be visible on the frame; otherwise, FALSE.
            /// </summary>
            DWMWA_ALLOW_NCPAINT,

            /// <summary>
            /// Use with DwmGetWindowAttribute. Retrieves the bounds of the caption button area in the window-relative space. The retrieved value is of type RECT. If the window is minimized or otherwise not visible to the user, then the value of the RECT retrieved is undefined. You should check whether the retrieved RECT contains a boundary that you can work with, and if it doesn't then you can conclude that the window is minimized or otherwise not visible.
            /// </summary>
            DWMWA_CAPTION_BUTTON_BOUNDS,

            /// <summary>
            /// Use with DwmSetWindowAttribute. Specifies whether non-client content is right-to-left (RTL) mirrored. The pvAttribute parameter points to a value of type BOOL. TRUE if the non-client content is right-to-left (RTL) mirrored; otherwise, FALSE.
            /// </summary>
            DWMWA_NONCLIENT_RTL_LAYOUT,

            /// <summary>
            /// Use with DwmSetWindowAttribute. Forces the window to display an iconic thumbnail or peek representation (a static bitmap), even if a live or snapshot representation of the window is available. This value is normally set during a window's creation, and not changed throughout the window's lifetime. Some scenarios, however, might require the value to change over time. The pvAttribute parameter points to a value of type BOOL. TRUE to require a iconic thumbnail or peek representation; otherwise, FALSE.
            /// </summary>
            DWMWA_FORCE_ICONIC_REPRESENTATION,

            /// <summary>
            /// Use with DwmSetWindowAttribute. Sets how Flip3D treats the window. The pvAttribute parameter points to a value from the DWMFLIP3DWINDOWPOLICY enumeration.
            /// </summary>
            DWMWA_FLIP3D_POLICY,

            /// <summary>
            /// Use with DwmGetWindowAttribute. Retrieves the extended frame bounds rectangle in screen space. The retrieved value is of type RECT.
            /// </summary>
            DWMWA_EXTENDED_FRAME_BOUNDS,

            /// <summary>
            /// Use with DwmSetWindowAttribute. The window will provide a bitmap for use by DWM as an iconic thumbnail or peek representation (a static bitmap) for the window. DWMWA_HAS_ICONIC_BITMAP can be specified with DWMWA_FORCE_ICONIC_REPRESENTATION. DWMWA_HAS_ICONIC_BITMAP normally is set during a window's creation and not changed throughout the window's lifetime. Some scenarios, however, might require the value to change over time. The pvAttribute parameter points to a value of type BOOL. TRUE to inform DWM that the window will provide an iconic thumbnail or peek representation; otherwise, FALSE. Windows Vista and earlier: This value is not supported.
            /// </summary>
            DWMWA_HAS_ICONIC_BITMAP,

            /// <summary>
            /// Use with DwmSetWindowAttribute. Do not show peek preview for the window. The peek view shows a full-sized preview of the window when the mouse hovers over the window's thumbnail in the taskbar. If this attribute is set, hovering the mouse pointer over the window's thumbnail dismisses peek (in case another window in the group has a peek preview showing). The pvAttribute parameter points to a value of type BOOL. TRUE to prevent peek functionality, or FALSE to allow it. Windows Vista and earlier: This value is not supported.
            /// </summary>
            DWMWA_DISALLOW_PEEK,

            /// <summary>
            /// Use with DwmSetWindowAttribute. Prevents a window from fading to a glass sheet when peek is invoked. The pvAttribute parameter points to a value of type BOOL. TRUE to prevent the window from fading during another window's peek, or FALSE for normal behavior. Windows Vista and earlier: This value is not supported.
            /// </summary>
            DWMWA_EXCLUDED_FROM_PEEK,

            /// <summary>
            /// Use with DwmSetWindowAttribute. Cloaks the window such that it is not visible to the user. The window is still composed by DWM. Using with DirectComposition: Use the DWMWA_CLOAK flag to cloak the layered child window when animating a representation of the window's content via a DirectComposition visual that has been associated with the layered child window. For more details on this usage case, see How to animate the bitmap of a layered child window. Windows 7 and earlier: This value is not supported.
            /// </summary>
            DWMWA_CLOAK,

            /// <summary>
            /// Use with DwmGetWindowAttribute. If the window is cloaked, provides one of the following values explaining why. DWM_CLOAKED_APP (value 0x0000001). The window was cloaked by its owner application. DWM_CLOAKED_SHELL(value 0x0000002). The window was cloaked by the Shell. DWM_CLOAKED_INHERITED(value 0x0000004). The cloak value was inherited from its owner window. Windows 7 and earlier: This value is not supported.
            /// </summary>
            DWMWA_CLOAKED,

            /// <summary>
            /// Use with DwmSetWindowAttribute. Freeze the window's thumbnail image with its current visuals. Do no further live updates on the thumbnail image to match the window's contents. Windows 7 and earlier: This value is not supported.
            /// </summary>
            DWMWA_FREEZE_REPRESENTATION,

            /// <summary>
            /// Use with DwmSetWindowAttribute. Enables a non-UWP window to use host backdrop brushes. If this flag is set, then a Win32 app that calls Windows::UI::Composition APIs can build transparency effects using the host backdrop brush (see Compositor.CreateHostBackdropBrush). The pvAttribute parameter points to a value of type BOOL. TRUE to enable host backdrop brushes for the window, or FALSE to disable it. This value is supported starting with Windows 11 Build 22000.
            /// </summary>
            DWMWA_USE_HOSTBACKDROPBRUSH,

            /// <summary>
            /// Use with DwmSetWindowAttribute. Allows the window frame for this window to be drawn in dark mode colors when the dark mode system setting is enabled. For compatibility reasons, all windows default to light mode regardless of the system setting. The pvAttribute parameter points to a value of type BOOL. TRUE to honor dark mode for the window, FALSE to always use light mode. This value is supported starting with Windows 10 Build 17763.
            /// </summary>
            DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1 = 19,

            /// <summary>
            /// Use with DwmSetWindowAttribute. Allows the window frame for this window to be drawn in dark mode colors when the dark mode system setting is enabled. For compatibility reasons, all windows default to light mode regardless of the system setting. The pvAttribute parameter points to a value of type BOOL. TRUE to honor dark mode for the window, FALSE to always use light mode. This value is supported starting with Windows 11 Build 22000.
            /// </summary>
            DWMWA_USE_IMMERSIVE_DARK_MODE = 20,

            /// <summary>
            /// Use with DwmSetWindowAttribute. Specifies the rounded corner preference for a window. The pvAttribute parameter points to a value of type DWM_WINDOW_CORNER_PREFERENCE. This value is supported starting with Windows 11 Build 22000.
            /// </summary>
            DWMWA_WINDOW_CORNER_PREFERENCE = 33,

            /// <summary>
            /// Use with DwmSetWindowAttribute. Specifies the color of the window border. The pvAttribute parameter points to a value of type COLORREF. The app is responsible for changing the border color according to state changes, such as a change in window activation. This value is supported starting with Windows 11 Build 22000.
            /// </summary>
            DWMWA_BORDER_COLOR,

            /// <summary>
            /// Use with DwmSetWindowAttribute. Specifies the color of the caption. The pvAttribute parameter points to a value of type COLORREF. This value is supported starting with Windows 11 Build 22000.
            /// </summary>
            DWMWA_CAPTION_COLOR,

            /// <summary>
            /// Use with DwmSetWindowAttribute. Specifies the color of the caption text. The pvAttribute parameter points to a value of type COLORREF. This value is supported starting with Windows 11 Build 22000.
            /// </summary>
            DWMWA_TEXT_COLOR,

            /// <summary>
            /// Use with DwmGetWindowAttribute. Retrieves the width of the outer border that the DWM would draw around this window. The value can vary depending on the DPI of the window. The pvAttribute parameter points to a value of type UINT. This value is supported starting with Windows 11 Build 22000.
            /// </summary>
            DWMWA_VISIBLE_FRAME_BORDER_THICKNESS,

            /// <summary>
            /// The maximum recognized DWMWINDOWATTRIBUTE value, used for validation purposes.
            /// </summary>
            DWMWA_LAST,
        }

        public enum DWM_WINDOW_CORNER_PREFERENCE {
            DWMWCP_DEFAULT = 0,
            DWMWCP_DONOTROUND = 1,
            DWMWCP_ROUND = 2,
            DWMWCP_ROUNDSMALL = 3
        }

        [DllImport("dwmapi.dll")]
        private static extern int DwmGetWindowAttribute(IntPtr hwnd, int dwAttribute, IntPtr pvAttribute, int cbAttribute);

        [DllImport("dwmapi.dll")]
        private static extern int DwmSetWindowAttribute(IntPtr hwnd, int dwAttribute, int* pvAttribute, int cbAttribute);


        IntPtr LocalWndProc(IntPtr hWnd, WM msg, IntPtr wParam, IntPtr lParam) {
            switch ((WM)msg) {
                case WM.CREATE:
                     
                    break;
                case WM.SHOWWINDOW:
                    _controller?.OnShow(this);
                    break;
                case WM.CLOSE:
                    _controller?.OnClose(this);
                    break; 
                case WM.PAINT:
                    Paint();
                    return IntPtr.Zero;
#if NCPAINT
                case WM.NCACTIVATE:
                    InvalidateAll();
                    break;
                case WM.ACTIVATE:
                    InvalidateAll();
                    break;

                case WM.NCHITTEST: {
                    RECT rect;
                    User32.GetWindowRect(hWnd, out rect);
                
                    int _SM_CYSIZEFRAME = GetFrameSize();
                    int _SM_CYCAPTION = GetCaptionSize();
                
                    int x = lParam.ToInt32() & 0xffff;
                    int y = lParam.ToInt32() >> 16;
                
                    const int HTCAPTION = 2;
                    const int HTLEFT = 10;
                    const int HTCLIENT = 1;
                    const int HTRIGHT = 11;
                    const int HTTOP = 12;
                    const int HTTOPLEFT = 13;
                    const int HTTOPRIGHT = 14;
                    const int HTBOTTOM = 15;
                    const int HTBOTTOMLEFT = 16;
                    const int HTBOTTOMRIGHT = 17;
                
                    if (x <= rect.Left + _SM_CYSIZEFRAME && y <= rect.Top + _SM_CYSIZEFRAME) {
                
                        return (IntPtr)HTTOPLEFT;
                
                    } else if (x >= rect.Right - _SM_CYSIZEFRAME && y <= rect.Top + _SM_CYSIZEFRAME) {
                
                        return (IntPtr)HTTOPRIGHT;
                
                    } else if (x <= rect.Left + _SM_CYSIZEFRAME && y >= rect.Bottom - _SM_CYSIZEFRAME) {
                
                        return (IntPtr)HTBOTTOMLEFT;
                
                    } else if (x >= rect.Right - _SM_CYSIZEFRAME && y >= rect.Bottom - _SM_CYSIZEFRAME) {
                
                        return (IntPtr)HTBOTTOMRIGHT;
                
                    } else if (x <= rect.Left + _SM_CYSIZEFRAME) {
                
                        return (IntPtr)HTLEFT;
                
                    } else if (x >= rect.Right - _SM_CYSIZEFRAME) {
                
                        return (IntPtr)HTRIGHT;
                
                    } else if (y <= rect.Top + _SM_CYSIZEFRAME) {
                
                        return (IntPtr)HTTOP;
                
                    } else if (y >= rect.Bottom - _SM_CYSIZEFRAME) {
                
                        return (IntPtr)HTBOTTOM;
                
                    }
                    return (IntPtr)HTCAPTION;
                }

                case WM.NCCALCSIZE: {
                        if (wParam == IntPtr.Zero) return IntPtr.Zero;
                
                        int _SM_CYSIZEFRAME = GetFrameSize();
                        int _SM_CYCAPTION = GetCaptionSize();
                
                        IntPtr WVR_REDRAW = (IntPtr)0x0300;
                        RECT rect;
                        User32.GetWindowRect(hWnd, out rect);
                        NCCALCSIZE_PARAMS* ncParams = (NCCALCSIZE_PARAMS*)lParam;
                        ncParams->rgrc0.Top = rect.Top + _SM_CYCAPTION;
                        ncParams->rgrc0.Left = rect.Left + _SM_CYSIZEFRAME;
                        ncParams->rgrc0.Bottom = rect.Bottom - _SM_CYSIZEFRAME;
                        ncParams->rgrc0.Right = rect.Right - _SM_CYSIZEFRAME;
                        return WVR_REDRAW;
                    }

                case WM.NCPAINT:
                    return WmNcPaint(hWnd, msg, wParam, lParam);
#endif

                case WM.WINMM:
                    User32.GetClientRect(hWnd, out RECT lprctw);
                    User32.InvalidateRect(hWnd, ref lprctw, false);
                    return IntPtr.Zero;
                case WM.KEYDOWN:
                    _controller?.OnKeyDown(this, wParam, lParam);
                    return IntPtr.Zero;
                case WM.DESTROY:
                    Dispose();
                    User32.PostQuitMessage(0);
                    return IntPtr.Zero;
            }
            return User32.DefWindowProc(hWnd, (WM)msg, wParam, lParam);
        }

        private static int GetFrameSize() {
            const int SM_CYSIZEFRAME = 33;
            int _SM_CYSIZEFRAME = User32.GetSystemMetrics(SM_CYSIZEFRAME);
            return _SM_CYSIZEFRAME;
        }

        private static int GetCaptionSize() {
            const int SM_CYCAPTION = 4;
            int _SM_CYCAPTION = User32.GetSystemMetrics(SM_CYCAPTION);
            return _SM_CYCAPTION * 2 - GetFrameSize();
        }

        public IntPtr Handle {
            get {
                if (_hWnd == IntPtr.Zero) {
                    // throw new ObjectDisposedException(GetType().Name);
                }
                return _hWnd;
            }
        }

        bool IWinUI.IsHandleAllocated { get => _hWnd != IntPtr.Zero; }
        ITheme IWinUI.Theme { get => _theme; }

        public void Show() {
            if (_hWnd == IntPtr.Zero) {
                throw new ObjectDisposedException(GetType().Name);
            }
            User32.ShowWindow(_hWnd, ShowWindowCommands.Normal);
            User32.UpdateWindow(_hWnd);
        }

        const int RDW_FRAME = 0x400;
        const int RDW_INVALIDATE = 0x1;

        [DllImport("user32.dll")]
        static extern bool RedrawWindow(IntPtr hWnd, IntPtr lprcUpdate, IntPtr hrgnUpdate, int flags);

        private void InvalidateAll() {
            RedrawWindow(this.Handle, IntPtr.Zero, IntPtr.Zero, RDW_FRAME | RDW_INVALIDATE);
        }

        [DllImport("gdi32.dll")]
        public static extern IntPtr CreateRectRgn(int nLeftRect, int nTopRect, int nReghtRect, int nBottomRect);

        [DllImport("gdi32.dll")]
        public static extern int CombineRgn(
              IntPtr hrgnDst,
              IntPtr hrgnSrc1,
              IntPtr hrgnSrc2,
              int iMode);

        const int RGN_AND = 0x01;
        const int RGN_OR = 0x02;
        const int RGN_XOR = 0x03;
        const int RGN_DIFF = 0x04;
        const int RGN_COPY = 0x05;

        [DllImport("user32.dll")]
        public static extern int MapWindowPoints(
          IntPtr hWndFrom,
          IntPtr hWndTo,
          POINT* lpPoints,
          uint cPoints
        );

        [DllImport("user32.dll")]
        public static extern bool DrawCaption(
          IntPtr hwnd,
          IntPtr hdc,
          RECT* lprect,
          uint flags
        );

        [DllImport("user32.dll")]
        public static extern bool DrawFrameControl(
              IntPtr hdc,
              RECT* lprc,
              uint uType,
              uint uState);

        IntPtr WmNcPaint(IntPtr hWnd, WM msg, IntPtr wParam, IntPtr lParam) {
            // User32.DefWindowProc(hWnd, msg, wParam, lParam);

            // User32.GetWindowRect(this.Handle, out RECT rect);
            // 
            // IntPtr region = IntPtr.Zero;
            // if (wParam == (IntPtr)1 || IntPtr.Zero == wParam) {
            //     region = CreateRectRgn(rect.Left, rect.Top, rect.Right, rect.Bottom);
            // } else {
            //     IntPtr copy = CreateRectRgn(0, 0, 0, 0);
            //     if (CombineRgn(copy, wParam, IntPtr.Zero, RGN_COPY) != 0) {
            //         region = copy;
            //     } else {
            //         User32.DeleteObject(copy);
            //     }
            // }
            // 
            // IntPtr hdc;
            // hdc = User32.GetDCEx(hWnd, IntPtr.Zero, (DeviceContextValues.DCX_WINDOW | DeviceContextValues.DCX_INTERSECTRGN));
            // 
            // using (Graphics g = Graphics.FromHdc(hdc)) {
            //     // g.SetClip(clip, CombineMode.Exclude);
            // 
            //     g.FillRectangle(Brushes.SeaGreen, 0, 0,
            //         rect.Width, rect.Height);
            // }
            // 
            // User32.ReleaseDC(hWnd, hdc);

            int w;
             int h;
             Rectangle clip;
             IntPtr hdc;

             User32.GetClientRect(this.Handle, out RECT clientRect);
             User32.GetWindowRect(this.Handle, out RECT windowRect);

             w = windowRect.Right - windowRect.Left;
             h = windowRect.Bottom - windowRect.Top;

             clip = new Rectangle(
                 clientRect.Left,
                 clientRect.Top,
                 clientRect.Right,
                 clientRect.Bottom);

             hdc = User32.GetWindowDC(this.Handle);
             // 
             using (Graphics g = Graphics.FromHdc(hdc)) {
                // g.SetClip(clip, CombineMode.Exclude);

                int _SM_CYSIZEFRAME = GetFrameSize();
                int _SM_CYCAPTION = GetCaptionSize();

                POINT pt = new POINT();
                pt.x = windowRect.Left;
                pt.Y = windowRect.Top;

                MapWindowPoints(IntPtr.Zero, hWnd, &pt, 1);

                Brush ncbrush = new SolidBrush(Color.FromArgb(12, 12, 12));

                g.FillRectangle(ncbrush,
                    windowRect.Width - _SM_CYSIZEFRAME,
                    0,
                    _SM_CYSIZEFRAME,
                    windowRect.Height);

                g.FillRectangle(ncbrush,
                    0,
                    windowRect.Height - _SM_CYSIZEFRAME,
                    windowRect.Width,
                    _SM_CYSIZEFRAME * 5);

                g.FillRectangle(ncbrush,
                    0,
                    windowRect.Height - _SM_CYSIZEFRAME,
                    windowRect.Width,
                    _SM_CYSIZEFRAME);

                g.FillRectangle(ncbrush,
                    0,
                    0,
                    _SM_CYSIZEFRAME,
                    windowRect.Height);

                ncbrush.Dispose();

                ncbrush = new SolidBrush(Color.FromArgb(0x2E, 0x2E, 0x2E));

                g.FillRectangle(Brushes.DarkRed,
                    windowRect.Width - _SM_CYCAPTION,
                    0,
                    _SM_CYCAPTION,
                    _SM_CYCAPTION);

                g.FillRectangle(ncbrush,
                    0,
                    0,
                    windowRect.Width,
                    _SM_CYCAPTION);

                ncbrush.Dispose();

            }
            // WM_NCCALCSIZE 
            User32.ReleaseDC(this.Handle, hdc);
             return IntPtr.Zero;

             return User32.DefWindowProc(hWnd, msg, wParam, lParam);
        }

        void Paint() {
            User32.GetClientRect(
                _hWnd,
                out RECT lprct);

            if (lprct.Width <= 0 || lprct.Height <= 0) {
                return;
            }

            IntPtr hdc = User32.BeginPaint(_hWnd, out PAINTSTRUCT ps);
            try {
                Graphics hdcGraphics = Graphics.FromHdc(hdc);
                try {
                    DrawClientArea(
                        lprct,
                        hdcGraphics);
                } finally {
                    hdcGraphics.Dispose();
                }
            } finally {
                User32.EndPaint(_hWnd, ref ps);
            }
        }

        void DrawClientArea(RECT lprct, Graphics hdcGraphics) {
            Bitmap hMemBitmap = new Bitmap(
                lprct.Width, lprct.Height, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            try {
                RectangleF rectF = new RectangleF(
                    0,
                    0,
                    lprct.Width,
                    lprct.Height);
                Graphics hdcBitmap = Graphics.FromImage(hMemBitmap);
                try {
                    hdcBitmap.FillRectangle(
                        _theme.GetBrush(ThemeColor.Background), rectF);
                    _controller?.OnPaint(this, hMemBitmap);
                    _controller?.OnPaint(this, hdcBitmap, rectF);
                } finally {
                    hdcBitmap.Dispose();
                }
                hdcGraphics.DrawImage(
                    hMemBitmap,
                    rectF);
            } finally {
                hMemBitmap.Dispose();
            }
        }
    }
}