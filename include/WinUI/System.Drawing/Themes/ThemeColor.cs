namespace System.Drawing {
    public enum ThemeColor {
        Background = 0,
        Foreground,
        A,
        B,
        C,
        D,
        E,
        DarkLine,
        LightLine,
        TitleBar,
    }

    public static class ThemeExtensions {
        public static ThemeColor Next(this ThemeColor c) {
            if ((c + 1) > ThemeColor.E || (c + 1) < ThemeColor.A) {
                return ThemeColor.A;
            }
            return c + 1;
        }
    }
}