using System;
using System.ComponentModel;
using System.Configuration;
using System.Diagnostics;
using System.Drawing;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Threading;
using static Microsoft.Win32.User32;

namespace Microsoft.Win32 {

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
                GCHandle hWnd = GCHandle.Alloc(new WinUI(controller, text, theme()), GCHandleType.Normal);
                try {
                    ((IWinUI)hWnd.Target).Show();
                    while (!controller.IsDisposed && GetMessage(out MSG msg, ((IWinUI)hWnd.Target).Handle, 0, 0) != 0) {
                        TranslateMessage(ref msg);
                        DispatchMessage(ref msg);
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

        bool _NCRENDERING_ENABLED = false;
        string _text;
        Icon _icon;
        IntPtr _hWnd = IntPtr.Zero;
        WNDCLASSEX _lpwcx;
        WndProc _lpfnWndProcPtr;
        IWinUIController _controller;
        ITheme _theme;
        internal partial class Kernel32 {
            [DllImport("Kernel32", CharSet = CharSet.Unicode, SetLastError = true, ExactSpelling = true)]
            public static extern IntPtr GetModuleHandleW(string moduleName);
            [DllImport("Kernel32", CharSet = CharSet.Ansi, BestFitMapping = false, ExactSpelling = true)]
            public static extern IntPtr GetProcAddress(IntPtr hModule, string lpProcName);
        }

        [DllImport("uxtheme.dll", ExactSpelling = true, CharSet = CharSet.Unicode)]
        private static extern int SetWindowTheme(IntPtr hwnd, string pszSubAppName, string pszSubIdList);

        public WinUI(IWinUIController controller, string text, ITheme theme) {
            _text = text;
            _controller = controller;
            _theme = theme;
            _icon = Icon.ExtractAssociatedIcon(Assembly.GetEntryAssembly().Location);

            if (!SetProcessDpiAwarenessContext(DpiAwarenessContext.PerMonitorAwareV2)) {
                Debug.WriteLine("WARNING: SetProcessDpiAwarenessContext failed.");
            }
            _lpfnWndProcPtr = LocalWndProc;
            IntPtr lpfnWndProcPtr = Marshal.GetFunctionPointerForDelegate(_lpfnWndProcPtr);
            _lpwcx = new WNDCLASSEX();
            _lpwcx.cbSize = Marshal.SizeOf(typeof(WNDCLASSEX));
            _lpwcx.hInstance = GetModuleHandle(null);
             _lpwcx.hIcon = _icon.Handle;
            _lpwcx.style = 
                ClassStyles.HorizontalRedraw |
                ClassStyles.VerticalRedraw |
                ClassStyles.OwnDC |
                ClassStyles.DropShadow;
            _lpwcx.cbClsExtra = 0;
            _lpwcx.cbWndExtra = 0;
            _lpwcx.hCursor = LoadCursor(IntPtr.Zero, (int)Constants.IDC_ARROW);
            _lpwcx.lpfnWndProc = lpfnWndProcPtr;
            _lpwcx.lpszClassName = "W_" + Guid.NewGuid().ToString("N");
            if (RegisterClassExA(ref _lpwcx) == 0) {
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
            _hWnd = CreateWindowExA(
                    WindowStylesEx.WS_EX_WINDOWEDGE |
                    WindowStylesEx.WS_EX_LEFT,
                    _lpwcx.lpszClassName,
                    text,
                    WindowStyles.WS_SYSMENU |
                    WindowStyles.WS_BORDER |
                    WindowStyles.WS_OVERLAPPED |
                    WindowStyles.WS_CAPTION |
                    WindowStyles.WS_SIZEFRAME |
                    WindowStyles.WS_MINIMIZEBOX |
                    WindowStyles.WS_MAXIMIZEBOX,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
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

            preference = (int)DWMNCRENDERINGPOLICY.DWMNCRP_ENABLED;
            DwmSetWindowAttribute(_hWnd,
                (int)DWMWINDOWATTRIBUTE.DWMWA_NCRENDERING_POLICY, &preference, sizeof(uint));

            preference = (int)0;
            DwmSetWindowAttribute(_hWnd,
                (int)DWMWINDOWATTRIBUTE.DWMWA_ALLOW_NCPAINT, &preference, sizeof(uint));

            UseImmersiveDarkMode(_hWnd, true);

            if (_NCRENDERING_ENABLED) {
                attribute = (int)DWMWINDOWATTRIBUTE.DWMWA_BORDER_COLOR;
                int color = 0x121212;
                byte R = theme.GetColor(ThemeColor.DarkLine).R;
                byte G = theme.GetColor(ThemeColor.DarkLine).G;
                byte B = theme.GetColor(ThemeColor.DarkLine).B;
                color = (R << 16) + (G << 8) + (B);
                DwmSetWindowAttribute(_hWnd, (int)DWMWINDOWATTRIBUTE.DWMWA_BORDER_COLOR, &color, sizeof(uint));

                const int DWMWA_CAPTION_COLOR = 35;
                color = 0x2E2E2E;
                R = theme.GetColor(ThemeColor.TitleBar).R;
                G = theme.GetColor(ThemeColor.TitleBar).G;
                B = theme.GetColor(ThemeColor.TitleBar).B;
                color = (R << 16) + (G << 8) + (B);
                DwmSetWindowAttribute(_hWnd, DWMWA_CAPTION_COLOR, &color, sizeof(uint));
            }
        }

        private const int DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1 = 19;
        private const int DWMWA_USE_IMMERSIVE_DARK_MODE = 20;
        private static bool UseImmersiveDarkMode(IntPtr handle, bool enabled) {
            if (IsWindows10OrGreater(17763)) {
                var attribute = DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1;
                if (IsWindows10OrGreater(18985)) {
                    attribute = DWMWA_USE_IMMERSIVE_DARK_MODE;
                }
                int useImmersiveDarkMode = enabled ? 1 : 0;
                return DwmSetWindowAttribute(handle, (int)attribute, &useImmersiveDarkMode, sizeof(int)) == 0;
            }
            return false;
        }
        private static bool IsWindows10OrGreater(int build = -1) {
            return Environment.OSVersion.Version.Major >= 10 && Environment.OSVersion.Version.Build >= build;
        }

        void Dispose(bool disposing) {
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

        void GetCloseButtonRect(IntPtr hWnd, out RECT lprcb) {
            var dpi = GetDpiForWindow(hWnd);
            int dx = GetSystemMetricsForDpi(SM_CXSIZEFRAME, dpi);
            int dy = GetSystemMetricsForDpi(SM_CYSIZEFRAME, dpi);
            int extra = GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi);
            int title = GetSystemMetricsForDpi(SM_CYCAPTION, dpi);
            int border = GetSystemMetricsForDpi(SM_CYBORDER, dpi);
            RECT rcclient;
            GetClientRect(hWnd, out rcclient);
            rcclient.Bottom    = rcclient.Top + (2 * title + dy - 2 * border);
            rcclient.Left   = rcclient.Right - (2 * title + dy - 2 * border);
            lprcb = rcclient;
        }

        void PaintNonClient_(IntPtr hWnd, IntPtr hdc, RECT lprct) {
            var dpi = GetDpiForWindow(hWnd);
            int extra = GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi);
            GetCloseButtonRect(hWnd, out var cbrc);
            lprct.Top = 0;
            lprct.Bottom = cbrc.Height;
            var hMemDC = CreateCompatibleDC(hdc);
            var hBmp = CreateCompatibleBitmap(hdc, lprct.Width, lprct.Height);
            SelectObject(hMemDC, hBmp);
            var hbBg = CreateSolidBrush(ColorTranslator.ToWin32(_theme.GetColor(ThemeColor.TitleBar)));
            SelectObject(hMemDC, hbBg);
            IntPtr hFont = IntPtr.Zero;
            using (var font = new Font("Segoe UI", 9.5f * (dpi / 100f), FontStyle.Regular)) {
                hFont = font.ToHfont();
            }
            SelectObject(hMemDC, hFont);
            SetBkColor(hMemDC, ColorTranslator.ToWin32(_theme.GetColor(ThemeColor.TitleBar)));
            SetTextColor(hMemDC, ColorTranslator.ToWin32(Color.White));
            try {
                FillRect(hMemDC, ref lprct, hbBg);

                var rcText = lprct;
                rcText.Left += 2 * (int)SystemFonts.DefaultFont.Size;
                rcText.Right -= 2 * cbrc.Height;

                if (IsMaximized(hWnd)) {
                    rcText.Top += extra;
                }

                DrawText(
                    hMemDC,
                    _text,
                    -1,
                    ref rcText,
                    DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS);

                using (Graphics hdcGraphics = Graphics.FromHdc(hMemDC)) {

                    WINDOWBUTTON wp = (WINDOWBUTTON)GetWindowLongPtr(hWnd, (int)WindowLongFlags.GWL_USERDATA);
                    if ((wp & WINDOWBUTTON.Close) == WINDOWBUTTON.Close) {
                        var cbb = new SolidBrush(Color.FromArgb(196, 43, 28));
                        hdcGraphics.FillRectangle(cbb,
                            cbrc.Left,
                            cbrc.Top,
                            cbrc.Width,
                            cbrc.Height);
                        cbb.Dispose();
                    }

                    // var closeBmp = Bitmap.FromFile("D:\\Dark-Close-Normal.bmp");
                    // 
                    // hdcGraphics.DrawImage(
                    //     closeBmp,
                    //     cbrc.Left + cbrc.Width / 2f - closeBmp.Width / 2f,
                    //     cbrc.Top + cbrc.Height / 2f - closeBmp.Height / 2f);
                    // 
                    // closeBmp.Dispose();

                    // 
                    // const int DFC_CAPTION             = 1;
                    // const int DFC_MENU                = 2;
                    // const int DFC_SCROLL              = 3;
                    // const int DFC_BUTTON = 4;
                    // 
                    // const int DFCS_CAPTIONCLOSE       = 0x0000;
                    // const int DFCS_CAPTIONMIN         = 0x0001;
                    // const int DFCS_CAPTIONMAX         = 0x0002;
                    // const int DFCS_CAPTIONRESTORE     = 0x0003;
                    // const int DFCS_CAPTIONHELP = 0x0004;
                    // 
                    // const int DFCS_INACTIVE = 0x0100;
                    // const int DFCS_PUSHED = 0x0200;
                    // const int DFCS_CHECKED = 0x0400;
                    // const int DFCS_TRANSPARENT = 0x0800;
                    // const int DFCS_HOT = 0x1000;
                    // const int DFCS_ADJUSTRECT = 0x2000;
                    // const int DFCS_FLAT = 0x4000;
                    // const int DFCS_MONO = 0x8000;
                    // 
                    // DrawFrameControl(hMemDC, &rcic, DFC_CAPTION, DFCS_CAPTIONCLOSE | DFCS_MONO);

                    // var rcic = lprcb;
                    // rcic.Top += 19;
                    // rcic.Bottom = rcic.Top + 13;
                    // rcic.Left += 19;
                    // rcic.Right = rcic.Left + 13;

                    // font = new Font(FontFamily.GenericMonospace, 11);
                    // sz = hdcGraphics.MeasureString("✕",
                    //     font,
                    //     lprcb.Width);
                    // hdcGraphics.DrawString("✕",
                    //     font,
                    //     Brushes.White,
                    //     lprcb.Left + lprcb.Width / 2 +  sz.Width / 2,
                    //     lprcb.Top + lprcb.Height / 2 + sz.Height / 2);
                    // font.Dispose();
    
                    var cbrc2 = cbrc; if (IsMaximized(hWnd)) {
                        cbrc2.Top += extra;
                        cbrc2.Right -= extra;
                    }
                    float pw = 1.3f * (dpi / 100f);
                    float W = 10f * (dpi / 100f);
                    float Offset = cbrc2.Height / 2f - W / 2f - 1f;
                    var pen =
                        (wp & WINDOWBUTTON.Close) == WINDOWBUTTON.Close
                                && (wp & WINDOWBUTTON.MouseDown) == WINDOWBUTTON.MouseDown
                        ? new Pen(Color.Silver, pw)
                        : new Pen(Color.White, pw);
                    hdcGraphics.DrawLine(pen,
                        new PointF(cbrc2.Left + Offset + W,
                            cbrc.Top + Offset),
                        new PointF(cbrc2.Left + Offset,
                            cbrc2.Top + Offset + W));
                    hdcGraphics.DrawLine(pen,
                        new PointF(cbrc2.Left + Offset,
                            cbrc2.Top + Offset),
                        new PointF(cbrc2.Left + Offset + W,
                            cbrc2.Top + Offset + W));
                    pen.Dispose();

                    BitBlt(hdc,
                        0,
                        0,
                        lprct.Width,
                        lprct.Height,
                        hMemDC,
                        0,
                        0,
                        TernaryRasterOperations.SRCCOPY);
                }
            } finally {
                DeleteObject(hFont);
                DeleteObject(hbBg);
                DeleteObject(hBmp);
                DeleteDC(hMemDC);
            }
        }

        void PaintClient(IntPtr hWnd, IntPtr hdc, RECT lprct) {
            int OffsetY = 0;
            if (_NCRENDERING_ENABLED) {
                GetCloseButtonRect(hWnd, out var cbrc);
                lprct.Bottom -= cbrc.Height;
                OffsetY = cbrc.Height;
            }
            var hMemDC = CreateCompatibleDC(hdc);
            var hBmp = CreateCompatibleBitmap(hdc, lprct.Width, lprct.Height);
            SelectObject(hMemDC, hBmp);
            try {
                using (Graphics hdcGraphics = Graphics.FromHdc(hMemDC)) {
                    DrawClientArea(lprct, hdcGraphics);
                    BitBlt(hdc,
                        0,
                        OffsetY,
                        lprct.Width,
                        lprct.Height,
                        hMemDC,
                        0,
                        0,
                        TernaryRasterOperations.SRCCOPY);
                }
            } finally {
                DeleteObject(hBmp);
                DeleteDC(hMemDC);
            }
        }

        [Flags]
        public enum WINDOWBUTTON : int {
            None = 0,
            Close = 2,
            MouseDown = 4,
        }

        public enum WINDOWPARTS : int {
            WP_NONE = 0,
            WP_CAPTION = 1,
            WP_SMALLCAPTION = 2,
            WP_MINCAPTION = 3,
            WP_SMALLMINCAPTION = 4,
            WP_MAXCAPTION = 5,
            WP_SMALLMAXCAPTION = 6,
            WP_FRAMELEFT = 7,
            WP_FRAMERIGHT = 8,
            WP_FRAMEBOTTOM = 9,
            WP_SMALLFRAMELEFT = 10,
            WP_SMALLFRAMERIGHT = 11,
            WP_SMALLFRAMEBOTTOM = 12,
            WP_SYSBUTTON = 13,
            WP_MDISYSBUTTON = 14,
            WP_MINBUTTON = 15,
            WP_MDIMINBUTTON = 16,
            WP_MAXBUTTON = 17,
            WP_CLOSEBUTTON = 18,
            WP_SMALLCLOSEBUTTON = 19,
            WP_MDICLOSEBUTTON = 20,
            WP_RESTOREBUTTON = 21,
            WP_MDIRESTOREBUTTON = 22,
            WP_HELPBUTTON = 23,
            WP_MDIHELPBUTTON = 24,
            WP_HORZSCROLL = 25,
            WP_HORZTHUMB = 26,
            WP_VERTSCROLL = 27,
            WP_VERTTHUMB = 28,
            WP_DIALOG = 29,
            WP_CAPTIONSIZINGTEMPLATE = 30,
            WP_SMALLCAPTIONSIZINGTEMPLATE = 31,
            WP_FRAMELEFTSIZINGTEMPLATE = 32,
            WP_SMALLFRAMELEFTSIZINGTEMPLATE = 33,
            WP_FRAMERIGHTSIZINGTEMPLATE = 34,
            WP_SMALLFRAMERIGHTSIZINGTEMPLATE = 35,
            WP_FRAMEBOTTOMSIZINGTEMPLATE = 36,
            WP_SMALLFRAMEBOTTOMSIZINGTEMPLATE = 37,
            WP_FRAME = 38,
            WP_BORDER = 39,
        };

        public enum CLOSEBUTTONSTATES : int {
            CBS_NORMAL = 1,
            CBS_HOT = 2,
            CBS_PUSHED = 3,
            CBS_DISABLED = 4,
        };
        public enum CAPTIONSTATES : int {
            CS_ACTIVE = 1,
            CS_INACTIVE = 2,
            CS_DISABLED = 3,
        };

        private IntPtr PaintNonClient(IntPtr hWnd) {
            int w;
            int h;
            IntPtr hdc;
            IntPtr hTheme;


            // int partId;
            // int stateId;

            GetWindowRect(hWnd, out RECT rcWindow);

            rcWindow.Right = rcWindow.Width;
            rcWindow.Bottom = rcWindow.Height;
            rcWindow.Left = 0;
            rcWindow.Top = 0;

            hdc = GetWindowDC(hWnd);

            GetClientRect(hWnd, out RECT lprctw2);

            // var bg_brush = User32.CreateSolidBrush(ColorTranslator.ToWin32(Color.Black));
            // User32.FillRect(hdc, ref rcWindow, bg_brush);
            // User32.DeleteObject(bg_brush);

            // PaintClient(hWnd, hdc);

            var hMemDC = CreateCompatibleDC(hdc);

            var hBmp = CreateCompatibleBitmap(hdc, rcWindow.Width, rcWindow.Height);

            SelectObject(hMemDC, hBmp);

            // ExcludeClipRect(hdc, 100, 100, 100, 100);

            var bg_brush = CreateSolidBrush(ColorTranslator.ToWin32(Color.Green));
            FillRect(hMemDC, ref rcWindow, bg_brush);
            DeleteObject(bg_brush);

            // User32.GetClientRect(hWnd, out RECT lprctw2);

            BitBlt(hdc,
                0,
                0,
                rcWindow.Width,
                rcWindow.Height,
                hMemDC,
                0,
                0,
                TernaryRasterOperations.SRCCOPY);

            DeleteObject(hBmp);

            DeleteDC(hMemDC);

            // 

            ReleaseDC(hWnd, hdc);

            return IntPtr.Zero;

            // w = windowRect.Right - windowRect.Left;
            // h = windowRect.Bottom - windowRect.Top;
            // 


            hdc = GetWindowDC(hWnd);

             hMemDC = CreateCompatibleDC(hdc);

            // var hBmp = User32.CreateCompatibleBitmap(hdc, rcWindow.Width, rcWindow.Height);

            // User32.SelectObject(hMemDC, hBmp);

            var dpi = GetDpiForWindow(hWnd);
            const int SM_CYCAPTION = 4;
            var _C_Y_CAPTION = GetSystemMetricsForDpi(SM_CYCAPTION, dpi);

             bg_brush = CreateSolidBrush(ColorTranslator.ToWin32(Color.Green));
            FillRect(hMemDC, ref rcWindow, bg_brush);
            DeleteObject(bg_brush);

            BitBlt(hdc,
                0,
                0,
                rcWindow.Width,
                _C_Y_CAPTION,
                hMemDC,
                0,
                0,
                TernaryRasterOperations.SRCCOPY);

            // User32.DeleteObject(hBmp);

            DeleteDC(hMemDC);

            // memBmp = CreateCompatibleBitmap(hdc, mRect.right, mRect.bottom);
            // //memBmp = zCreateDibSection(hdc, mRect.right, mRect.bottom, 24);
            // 
            // SelectObject(memDC, memBmp);


            // ExcludeClipRect(hdc, 0, _C_Y_CAPTION, rcWindow.Right, rcWindow.Bottom);
            // 
            // var bg_brush = User32.CreateSolidBrush(ColorTranslator.ToWin32(Color.Red));
            // User32.FillRect(hdc, ref rcWindow, bg_brush);
            // User32.DeleteObject(bg_brush);
            // 
            // rcWindow.Right = rcWindow.Width - 1;
            // rcWindow.Bottom = rcWindow.Height - 1;
            // rcWindow.Left = 1;
            // rcWindow.Top = 1;


            // partId = EP_EDITTEXT;
            // 
            // stateId = ETS_NORMAL;
            // // this.Enabled
            // //   ? ETS_NORMAL
            // //   : ETS_DISABLED;
            // 
            // 
            // // 
            // // if (IsThemeBackgroundPartiallyTransparent(hTheme, partId, stateId) != 0) {
            // //     DrawThemeParentBackground(this.Handle, hdc, ref windowRect);
            // // }
            // 
            // // DrawThemeBackground(hTheme, hdc, partId, stateId, ref windowRect, IntPtr.Zero);
            // 
            // // Paint Background
            // // COLORREF bg_color = RGB(200, 250, 230);
            // var bg_brush = User32.CreateSolidBrush(ColorTranslator.ToWin32(Color.Red));
            // User32.FillRect(hdc, ref windowRect, bg_brush);
            // User32.DeleteObject(bg_brush);
            // 
            // // IntPtr handle = OpenThemeData(IntPtr.Zero, "WINDOW");

            // var opts = new DTTOPTS();
            // opts.dwSize = Marshal.SizeOf<DTTOPTS>();
            // 
            // DrawThemeTextEx(
            //     hTheme,
            //     hdc,
            //     WP_CAPTION, CS_ACTIVE, "Hello",
            //     5,
            //     0,
            //     ref rcWindow,
            //     ref opts);



            // PaintClient(hWnd, hdc, rcWindow);

            ReleaseDC(hWnd, hdc);


            return IntPtr.Zero;

            // RedrawWindow(hWnd, IntPtr.Zero, IntPtr.Zero, RDW_FRAME | RDW_INVALIDATE);
            // return User32.DefWindowProc(hWnd, WM.NCPAINT, wParam, lParam);
        }

        const int SWP_NOSIZE           = 0x0001;
        const int SWP_NOMOVE           = 0x0002;
        const int SWP_NOZORDER         = 0x0004;
        const int SWP_NOREDRAW         = 0x0008;
        const int SWP_NOACTIVATE       = 0x0010;
        const int SWP_FRAMECHANGED     = 0x0020; /* The frame changed: send WM_NCCALCSIZE */
        const int SWP_SHOWWINDOW       = 0x0040;
        const int SWP_HIDEWINDOW       = 0x0080;
        const int SWP_NOCOPYBITS       = 0x0100;
        const int SWP_NOOWNERZORDER    = 0x0200; /* Don't do owner Z ordering */
        const int SWP_NOSENDCHANGING   = 0x0400; /* Don't send WM_WINDOWPOSCHANGING */
        const int SWP_DRAWFRAME = SWP_FRAMECHANGED;
        const int SWP_NOREPOSITION = SWP_NOOWNERZORDER;

        [Serializable]
        [StructLayout(LayoutKind.Sequential)]
        public struct WINDOWPLACEMENT {
            public int length;
            public uint flags;
            public uint showCmd;
            public POINT ptMinPosition;
            public POINT ptMaxPosition;
            public RECT rcNormalPosition;
        }

        [DllImport("user32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool GetWindowPlacement(
              IntPtr hWnd,
              ref WINDOWPLACEMENT lpwndpl);

        /*
     * ShowWindow() Commands
     */
            const int SW_HIDE = 0
             ; const int SW_SHOWNORMAL = 1
             ; const int SW_NORMAL = 1
             ; const int SW_SHOWMINIMIZED = 2
             ; const int SW_SHOWMAXIMIZED = 3
             ; const int SW_MAXIMIZE = 3
             ; const int SW_SHOWNOACTIVATE = 4
             ; const int SW_SHOW = 5
             ; const int SW_MINIMIZE = 6
             ; const int SW_SHOWMINNOACTIVE = 7
             ; const int SW_SHOWNA = 8
             ; const int SW_RESTORE = 9
             ; const int SW_SHOWDEFAULT = 10;
            const int SW_FORCEMINIMIZE = 11;
            const int SW_MAX = 11;

        static bool IsMaximized(IntPtr hWnd) {
            WINDOWPLACEMENT placement = new WINDOWPLACEMENT();
            placement.length = Marshal.SizeOf<WINDOWPLACEMENT>();
            if (GetWindowPlacement(hWnd, ref placement)) {
                return placement.showCmd == SW_SHOWMAXIMIZED;
            }
            return false;
        }

        IntPtr LocalWndProc(IntPtr hWnd, WM msg, IntPtr wParam, IntPtr lParam) {
            switch (msg) {
                case WM.SHOWWINDOW:
                    _controller?.OnShow(this);
                    break;
                case WM.CLOSE:
                    _controller?.OnClose(this);
                    break;

                case WM.PAINT:
                    IntPtr hdc = BeginPaint(hWnd, out PAINTSTRUCT ps);
                    try {
                        RECT rcclient;
                        GetClientRect(hWnd, out rcclient);
                        if (_NCRENDERING_ENABLED)
                            PaintNonClient_(hWnd, hdc, rcclient);
                        PaintClient(hWnd, hdc, rcclient);
                    } finally {
                        EndPaint(hWnd, ref ps);
                    }
                    return IntPtr.Zero;

                case WM.WINMM:
                    Invalidate(hWnd);
                    return IntPtr.Zero;

                case WM.KEYDOWN:
                    _controller?.OnKeyDown(this, wParam, lParam);
                    return IntPtr.Zero;

                case WM.CREATE:
                    RECT size_rect;
                    GetWindowRect(hWnd, out size_rect);
                    // Inform the application of the frame change to force redrawing with the new
                    // client area that is extended into the title bar
                    SetWindowPos(
                      hWnd, IntPtr.Zero,
                      size_rect.Left, size_rect.Top,
                      size_rect.Right - size_rect.Left, size_rect.Bottom - size_rect.Top,
                            SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE
                    );
                    break;

                case WM.DESTROY:
                    Dispose();
                    PostQuitMessage(0);
                    return IntPtr.Zero;

                case WM.NCPAINT:
                    break;

                // case WM.SIZE:
                //     // InvalidateRect()
                //     RedrawWindow(hWnd);
                //     // PaintNonClient(hWnd);
                //     Invalidate(hWnd);
                //     break;
                case WM.NCACTIVATE:
                case WM.ACTIVATE:
                    // var pMarInset = new _MARGINS();
                    // pMarInset.cyTopHeight = 0;
                    // DwmExtendFrameIntoClientArea(hWnd, ref pMarInset);
                    Invalidate(hWnd);
                    // RedrawNonClientArea(hWnd);
                    break;

                case WM.ERASEBKGND:
                    if (!_NCRENDERING_ENABLED) break;
                    return 1;

                // Handling this event allows us to extend client (paintable) area into the title bar region
                // The information is partially coming from:
                // https://docs.microsoft.com/en-us/windows/win32/dwm/customframe#extending-the-client-frame
                // Most important paragraph is:
                //   To remove the standard window frame, you must handle the WM_NCCALCSIZE message,
                //   specifically when its wParam value is TRUE and the return value is 0.
                //   By doing so, your application uses the entire window region as the client area,
                //   removing the standard frame.
                case WM.NCCALCSIZE: {
                        if (!_NCRENDERING_ENABLED) break;

                        if (wParam == IntPtr.Zero)
                            return IntPtr.Zero; // DefWindowProc(hWnd, msg, wParam, lParam);
                         
                        var dpi = GetDpiForWindow(hWnd);

                        int dx = GetSystemMetricsForDpi(SM_CXSIZEFRAME, dpi);
                        int dy = GetSystemMetricsForDpi(SM_CYSIZEFRAME, dpi);
                        int extra = GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi);
                        int title = GetSystemMetricsForDpi(SM_CYCAPTION, dpi);

                        NCCALCSIZE_PARAMS * params2 = (NCCALCSIZE_PARAMS*)lParam;

                        // if (IsMaximized(hWnd)) {
                        //     params2->rgrc0.Top -= title + extra;
                        //     params2->rgrc0.Bottom += extra;
                        // } else {
                        //     params2->rgrc0.Right    -= dx + extra;
                        //     params2->rgrc0.Left     += dx + extra;
                        //     params2->rgrc0.Bottom   -= dy + extra;
                        //     // params2->rgrc0.Top      += title;
                        // }
                        // RedrawNonClientArea(hWnd);

                        return IntPtr.Zero; // DefWindowProc(hWnd, msg, wParam, lParam);
                    }

                case WM.NCHITTEST: {
                        if (!_NCRENDERING_ENABLED) break;

                        RECT rect;
                        GetWindowRect(hWnd, out rect);
                        const int SM_CXSIZEFRAME = 32;
                        const int SM_CYSIZEFRAME = 33;
                        var _C_X_SIZEFRAME = 2 * GetSystemMetrics(SM_CXSIZEFRAME);
                        var _C_Y_SIZEFRAME = 2 * GetSystemMetrics(SM_CYSIZEFRAME);
                        int x = lParam.ToInt32() & 0xffff;
                        int y = lParam.ToInt32() >> 16;
                        if (x <= rect.Left + _C_X_SIZEFRAME && y <= rect.Top + _C_Y_SIZEFRAME) {
                            return HTTOPLEFT;
                        } else if (x >= rect.Right - _C_X_SIZEFRAME && y <= rect.Top + _C_Y_SIZEFRAME) {
                            return HTTOPRIGHT;
                        } else if (y >= rect.Bottom - _C_Y_SIZEFRAME && x >= rect.Right - _C_X_SIZEFRAME) {
                            return HTBOTTOMRIGHT;
                        } else if (x <= rect.Left + _C_X_SIZEFRAME && y >= rect.Bottom - _C_Y_SIZEFRAME) {
                            return HTBOTTOMLEFT;
                        }
                        else if (x <= rect.Left + _C_X_SIZEFRAME) {
                            return HTLEFT;
                        } else if (y <= rect.Top + _C_Y_SIZEFRAME) {
                            return HTTOP;
                        } else if (x >= rect.Right - _C_X_SIZEFRAME) {
                            return HTRIGHT;
                        } else if (y >= rect.Bottom - _C_Y_SIZEFRAME) {
                            return HTBOTTOM;
                        }
                        RECT rccb;
                        GetCloseButtonRect(hWnd, out rccb);
                        var pt = new POINT() { x = x, Y = y };
                        ScreenToClient(hWnd, ref pt);
                        if (PtInRect(ref rccb, pt)) {
                            return HTCLIENT;
                        }
                        return HTCAPTION;
                    }

                case WM.NCLBUTTONUP:
                case WM.LBUTTONUP:
                case WM.NCLBUTTONDOWN:
                case WM.LBUTTONDOWN:
                case WM.NCMOUSELEAVE:
                case WM.MOUSELEAVE:
                case WM.NCMOUSEHOVER:
                case WM.MOUSEHOVER:
                case WM.MOUSEMOVE:
                case WM.NCMOUSEMOVE: {
                        if (!_NCRENDERING_ENABLED) break;

                        POINT pt;
                        GetCursorPos(out pt);
                        ScreenToClient(hWnd, ref pt);
                        RECT rccb;
                        GetCloseButtonRect(hWnd, out rccb);
                        WINDOWBUTTON prev = (WINDOWBUTTON)GetWindowLongPtr(hWnd, (int)WindowLongFlags.GWL_USERDATA);
                        WINDOWBUTTON wp = prev;
                        if (PtInRect(ref rccb, pt)) {
                            wp |= WINDOWBUTTON.Close;
                            if (msg == WM.LBUTTONDOWN || msg == WM.NCLBUTTONDOWN) {
                                wp |= WINDOWBUTTON.MouseDown;
                            }
                        } else {
                            wp &= ~WINDOWBUTTON.Close;
                        }
                        if (msg == WM.LBUTTONUP || msg == WM.NCLBUTTONUP) {
                            wp &= ~WINDOWBUTTON.MouseDown;
                            if ((wp & WINDOWBUTTON.Close) == WINDOWBUTTON.Close) {
                                InvalidateRect(hWnd, ref rccb, false);
                                PostMessage(hWnd, WM.CLOSE, 0, 0);
                            }
                        }
                        SetWindowLongPtr(hWnd, (int)WindowLongFlags.GWL_USERDATA, (IntPtr)wp);
                        if (prev != wp) {
                            InvalidateRect(hWnd, ref rccb, false);
                        }
                        // Console.WriteLine(msg);
                        // Console.WriteLine(wp);
                        break;
                    }
            }

            return DefWindowProc(hWnd, (WM)msg, wParam, lParam);
        }

        static void Invalidate(nint hWnd) {
            GetClientRect(hWnd, out RECT lprctw);
            InvalidateRect(hWnd, ref lprctw, false);
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
            ShowWindow(_hWnd, ShowWindowCommands.Normal);
            UpdateWindow(_hWnd);
        }

        private void RedrawNonClientArea(IntPtr hWnd) {
            User32.RedrawWindow(hWnd, IntPtr.Zero, 
                IntPtr.Zero, RDW_FRAME | RDW_INVALIDATE | RDW_UPDATENOW);
        }

        [DllImport("gdi32.dll")]
        public static extern IntPtr CreateRectRgn(int nLeftRect, int nTopRect, int nReghtRect, int nBottomRect);

        [DllImport("gdi32.dll")]
        public static extern int CombineRgn(
              IntPtr hrgnDst,
              IntPtr hrgnSrc1,
              IntPtr hrgnSrc2,
              int iMode);

        public const int RGN_AND = 0x01;
        public const int RGN_OR = 0x02;
        public const int RGN_XOR = 0x03;
        public const int RGN_DIFF = 0x04;
        public const int RGN_COPY = 0x05;

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

        [DllImport("dwmapi.dll")]
        public static extern void DwmIsCompositionEnabled(ref bool pfEnabled);

        [DllImport("uxtheme.dll", CharSet = CharSet.Unicode)]
        public static extern IntPtr OpenThemeData(IntPtr hwnd, string pszClassList);

        [DllImport("uxtheme.dll", CharSet = CharSet.Unicode)]
        public extern static Int32 DrawThemeTextEx(IntPtr hTheme, IntPtr hdc, int iPartId, int iStateId, string pszText, int iCharCount, uint flags, ref RECT rect, ref DTTOPTS poptions);

        [StructLayout(LayoutKind.Sequential)]
        public struct DTTOPTS {
            public int dwSize;
            public int dwFlags;
            public int crText;
            public int crBorder;
            public int crShadow;
            public int iTextShadowType;
            public int ptShadowOffsetX;
            public int ptShadowOffsetY;
            public int iBorderSize;
            public int iFontPropId;
            public int iColorPropId;
            public int iStateId;
            public bool fApplyOverlay;
            public int iGlowSize;
            public IntPtr pfnDrawTextCallback;
            public IntPtr lParam;
        }

        void DrawClientArea(RECT rcClient, Graphics hdcGraphics) {

            if (rcClient.Width <= 0 || rcClient.Height <= 0) return;
            // Bitmap hMemBitmap = new Bitmap(
            //     lprct.Width, lprct.Height, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            try {
                RectangleF rectF = new RectangleF(
                    rcClient.Left,
                    rcClient.Top,
                    rcClient.Width,
                    rcClient.Height);
                // Graphics hdcBitmap = Graphics.FromHdc(hdc);
                try {
                    // hdcBitmap.FillRectangle(
                    //     _theme.GetBrush(ThemeColor.Background), rectF);
                    // _controller?.OnPaint(this, hMemBitmap);
                    _controller?.OnPaint(this, hdcGraphics, rectF);
                } finally {
                   // hdcBitmap.Dispose();
                }
                // hdcGraphics.DrawImage(
                //     hMemBitmap,
                //     rectF);
            } finally {
                // hMemBitmap.Dispose();
            }
        }
    }
}