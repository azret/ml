namespace Microsoft.WinMM {
    using System;
    using Microsoft.Win32;
    
    public interface IThreadProc : IDisposable, IChromeUIModel {
        long Mem { get; }
        void Mute();
        void Toggle();
        void UnMute();
        bool IsMuted { get; }
    }
}