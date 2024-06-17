namespace Microsoft.Win32 {
    using System;
    using System.Drawing;
    using Microsoft.WinMM;

    public class WinUIController<TViewModel> : IWinUIController {
        public virtual int DefaultWidth { get; set; } = -1;
        public virtual int DefaultHeight { get; set; } = -1;

        TViewModel _model;
        WinUIControllerOptions _options;
        public WinUIController(TViewModel model,
            WinUIControllerOptions options = WinUIControllerOptions.None) {
            _model = model;
            _options = options;
        }
        public TViewModel Model { get => _model; }
        public virtual void OnShow(IWinUI winUI) {
            if (_model is IChromeUIModel winUIModel && winUI.Handle != IntPtr.Zero) winUIModel.AddWinUIClient(winUI.Handle);
            if (_options.IsUnMuteModelOnOpen() && _model is IThreadProc threadProc) { threadProc.UnMute(); }
        }
        public bool IsDisposed { get => (_model is WinUIModel model) ? model.IsDisposed : false; }
        public virtual void OnClose(IWinUI winUI) {
            if (_model is IChromeUIModel winUIModel && winUI.Handle != IntPtr.Zero) winUIModel.RemoveWinUIClient(winUI.Handle);
            if (_options.IsDisposeModelOnClose() && _model is IDisposable disp) { disp.Dispose(); }
        }
        public virtual void OnKeyDown(IWinUI winUI, IntPtr wParam, IntPtr lParam) {
            if (wParam == new IntPtr(0x20)) {
                if (_options.IsMuteOnSpaceBar() && _model is IThreadProc threadProc) {
                    WinMM.PlaySound(null,
                            IntPtr.Zero,
                            WinMM.PLAYSOUNDFLAGS.SND_ASYNC |
                            WinMM.PLAYSOUNDFLAGS.SND_FILENAME |
                            WinMM.PLAYSOUNDFLAGS.SND_NODEFAULT |
                            WinMM.PLAYSOUNDFLAGS.SND_NOWAIT |
                            WinMM.PLAYSOUNDFLAGS.SND_PURGE);
                    threadProc.Toggle();
                }
            }
        }
        public virtual void OnPaint(IWinUI winUI, Graphics g, RectangleF r) {}
    }
}