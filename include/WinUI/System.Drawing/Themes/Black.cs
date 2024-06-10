namespace System.Drawing {
    using System;
    using System.Drawing;
    using System.Threading;

    public class Black : ITheme {
        public readonly int ThreadId = Thread.CurrentThread.ManagedThreadId;
        class _Fonts {
            public readonly Font ExtraSmall = new Font("Consolas", 5.5f);
            public readonly Font Small = new Font("Consolas", 7.5f);
            public readonly Font Normal = new Font("Consolas", 11.5f);
            public readonly Font Large = new Font("Consolas", 14.5f);
            public readonly Font ExtraLarge = new Font("Consolas", 17.5f);
        }
        _Fonts Fonts = new _Fonts();
        class _Colors {
            public readonly Color Background = Color.FromArgb(0x0C, 0x0C, 0x0C);
            public readonly Color Foreground = Color.FromArgb(178, 178, 178);
            public readonly Color A = Color.FromArgb(255, 227, 158);
            public readonly Color B = Color.FromArgb(245, 87, 98);
            public readonly Color C = Color.FromArgb(112, 218, 255);
            public readonly Color D = Color.FromArgb(141, 227, 141);
            public readonly Color E = Color.White;
            public readonly Color LightLine = Color.FromArgb(46, 46, 46);
            public readonly Color DarkLine = Color.FromArgb(0x2F, 0x2F, 0x2F);
        }
        _Colors Colors = new _Colors();
        class _Brushes {
            _Colors Colors;
            public readonly Brush Background;
            public readonly Brush Foreground;
            public readonly Brush A;
            public readonly Brush B;
            public readonly Brush C;
            public readonly Brush D;
            public readonly Brush E;
            public _Brushes(_Colors colors) {
                Colors = colors;
                Background = new SolidBrush(Colors.Background);
                Foreground = new SolidBrush(Colors.Foreground);
                A = new SolidBrush(Colors.A);
                B = new SolidBrush(Colors.B);
                C = new SolidBrush(Colors.C);
                D = new SolidBrush(Colors.D);
                E = new SolidBrush(Colors.E);
            }
        }
        _Brushes Brushes;
        class _Pens {
            _Colors Colors;
            public readonly Pen Background;
            public readonly Pen Foreground;
            public readonly Pen A;
            public readonly Pen B;
            public readonly Pen C;
            public readonly Pen D;
            public readonly Pen E;
            public readonly Pen LightLine;
            public readonly Pen DarkLine;
            public _Pens(_Colors colors) {
                Colors = colors;
                Background = new Pen(Colors.Background);
                Foreground = new Pen(Colors.Foreground);
                A = new Pen(Colors.A);
                B = new Pen(Colors.B);
                C = new Pen(Colors.C);
                D = new Pen(Colors.D);
                E = new Pen(Colors.E);
                LightLine = new Pen(Colors.LightLine);
                DarkLine = new Pen(Colors.DarkLine);
            }
        }
        _Pens Pens;
        public Black() {
            Pens = new _Pens(Colors);
            Brushes = new _Brushes(Colors);
        }
        Font ITheme.GetFont(ThemeFont font) {
            if (ThreadId != Thread.CurrentThread.ManagedThreadId) {
                throw new InvalidOperationException();
            }
            switch (font) {
                case ThemeFont.ExtraSmall:
                    return Fonts.ExtraSmall;
                case ThemeFont.Small:
                    return Fonts.Small;
                case ThemeFont.Normal:
                    return Fonts.Normal;
                case ThemeFont.Large:
                    return Fonts.Large;
                case ThemeFont.ExtraLarge:
                    return Fonts.ExtraLarge;
            }
            throw new NotImplementedException();
        }
        Color ITheme.GetColor(ThemeColor color) {
            if (ThreadId != Thread.CurrentThread.ManagedThreadId) {
                throw new InvalidOperationException();
            }
            switch (color) {
                case ThemeColor.Background:
                    return Colors.Background;
                case ThemeColor.Foreground:
                    return Colors.Foreground;
                case ThemeColor.A:
                    return Colors.A;
                case ThemeColor.B:
                    return Colors.B;
                case ThemeColor.C:
                    return Colors.C;
                case ThemeColor.D:
                    return Colors.D;
                case ThemeColor.E:
                    return Colors.E;
                case ThemeColor.DarkLine:
                    return Colors.DarkLine;
                case ThemeColor.LightLine:
                    return Colors.LightLine;
            }
            throw new NotImplementedException();
        }
        Brush ITheme.GetBrush(ThemeColor color) {
            if (ThreadId != Thread.CurrentThread.ManagedThreadId) {
                throw new InvalidOperationException();
            }
            switch (color) {
                case ThemeColor.Background:
                    return Brushes.Background;
                case ThemeColor.Foreground:
                    return Brushes.Foreground;
                case ThemeColor.A:
                    return Brushes.A;
                case ThemeColor.B:
                    return Brushes.B;
                case ThemeColor.C:
                    return Brushes.C;
                case ThemeColor.D:
                    return Brushes.D;
                case ThemeColor.E:
                    return Brushes.E;
            }
            throw new NotImplementedException();
        }
        Pen ITheme.GetPen(ThemeColor color) {
            if (ThreadId != Thread.CurrentThread.ManagedThreadId) {
                throw new InvalidOperationException();
            }
            switch (color) {
                case ThemeColor.Background:
                    return Pens.Background;
                case ThemeColor.Foreground:
                    return Pens.Foreground;
                case ThemeColor.A:
                    return Pens.A;
                case ThemeColor.B:
                    return Pens.B;
                case ThemeColor.C:
                    return Pens.C;
                case ThemeColor.D:
                    return Pens.D;
                case ThemeColor.E:
                    return Pens.E;
                case ThemeColor.LightLine:
                    return Pens.LightLine;
                case ThemeColor.DarkLine:
                    return Pens.DarkLine;
            }
            throw new NotImplementedException();
        }
    }
}