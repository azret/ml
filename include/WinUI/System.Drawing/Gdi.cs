using System.Diagnostics;

namespace System.Drawing {
    public enum SignalType {
        Line,
        Dot,
        Curve,
    }

    public static class Gdi {
        const byte Xscale = 13;
        const byte Yscale = 13;

        static float Map(float value, float fromLow, float fromHigh, float toLow, float toHigh) {
            return (value - fromLow) * (toHigh - toLow) / (fromHigh - fromLow) + toLow;
        }

        public static void DrawPaper(Graphics g, RectangleF r, ITheme theme) {
            DrawPaper(g, r, theme, Xscale, Yscale);
        }

        public static void DrawPaper(Graphics g, RectangleF r, ITheme theme, byte Xscale, byte Yscale) {
            var PixelOffsetMode = g.PixelOffsetMode;
            g.PixelOffsetMode = Drawing2D.PixelOffsetMode.None;
            var pen = theme.GetPen(ThemeColor.LightLine);
            int n = 0;
            for (float x = r.Left; x < r.Right; x += Xscale) {
                if (n > 0)
                    g.DrawLine(pen,
                        new PointF(x, r.Top),
                        new PointF(x, r.Bottom));
                n++;
            }
            n = 0;
            for (float y = r.Top; y < r.Bottom; y += Yscale) {
                if (n > 0)
                    g.DrawLine(pen,
                        new PointF(r.Left, y),
                        new PointF(r.Right, y));
                n++;
            }
            pen = theme.GetPen(ThemeColor.DarkLine);
            n = 0;
            for (float x = r.Left; x < r.Right; x += Xscale * 8) {
                if (n > 0)
                    g.DrawLine(pen,
                        new PointF(x, r.Top),
                        new PointF(x, r.Bottom));
                n++;
            }
            n = 0;
            for (float y = r.Top; y < r.Bottom; y += Yscale * 8) {
                if (n > 0)
                    g.DrawLine(pen,
                    new PointF(r.Left, y),
                    new PointF(r.Right, y));
                n++;
            }
            g.PixelOffsetMode = PixelOffsetMode;
        }

        public static Color Blend(Color srcColor, Color dstColor, double amount) {
            var r = (byte)((srcColor.R * amount) + dstColor.R * (1 - amount));
            var g = (byte)((srcColor.G * amount) + dstColor.G * (1 - amount));
            var b = (byte)((srcColor.B * amount) + dstColor.B * (1 - amount));
            var a = srcColor.A;
            return Color.FromArgb(a, r, g, b);
        }

        public static void DrawBars(Graphics g, RectangleF r, int cc, Func<int, (float ampl, string label)> X, Brush bg, Font font, Brush fg) {
            byte Xscale = Gdi.Xscale;
            if (cc == 18) {
                Xscale *= 4;
            }

            var PixelOffsetMode = g.PixelOffsetMode;
            g.PixelOffsetMode = Drawing2D.PixelOffsetMode.HighQuality;
            g.SmoothingMode = Drawing2D.SmoothingMode.HighQuality;
            int i = 0;
            for (int x = 0; x < r.Width; x += Xscale) {
                if (x % Xscale == 0) {
                    if (i >= cc) break;
                    var med = (int)((int)(r.Height / Yscale) / 2) * Yscale;
                    float w = Xscale / 2;
                    var Pt = X(i);
                    if (Pt.ampl < 0) {
                        Diagnostics.Debugger.Break();
                    }
                    var h = ((int)((int)(r.Height / Yscale) / 2) * Pt.ampl) * Yscale - (w / 2);
                    Brush bg1 = bg is SolidBrush bgSolidColorBrush
                         ? new SolidBrush(Blend(bgSolidColorBrush.Color, Color.Gray, Pt.ampl)) : (Brush)bg.Clone();

                    if (h > 0) {
                        g.FillRectangle(bg1,
                            1 + x + w / 2, med - h, w, h);
                    } else if (h < 0) {
                        g.FillRectangle(bg1,
                            1 + x + w / 2, med - h, w, h);
                    }
                    g.FillEllipse(
                        bg1,
                        1 + x + w / 2,
                        1 + med + Yscale / 2 - 3, w, w);

                    var s = Pt.label;
                    if (!string.IsNullOrWhiteSpace(s)) {
                        var sz = g.MeasureString(s, font);
                        StringFormat drawFormat = new StringFormat();
                        drawFormat.FormatFlags = StringFormatFlags.NoWrap
                            | StringFormatFlags.DirectionVertical | StringFormatFlags.DirectionRightToLeft;
                        g.DrawString(
                            "- " + s, font, bg1,
                            1 + x + w + sz.Height / 2,
                            2 * w + 1 + med + Yscale / 2 - 3,
                            drawFormat);
                    }

                    bg1.Dispose();

                    i++;
                }
            }
            g.PixelOffsetMode = PixelOffsetMode;
        }
 
        
        public static void DrawChannels2(
            Graphics g,
            RectangleF r,
            int cc,
            Func<int, Brush> bg,
            Font font,
            ITheme theme,
            Func<int, (float[] ampl, string label)> X = null,
            bool blend = true) {

            if (r.Width <= 0 || r.Height <= 0) {
                return;
            }

            float barWidth = r.Width / cc;
            if (barWidth <= 0) barWidth = 1f;
            float space = ((r.Width - barWidth) / (cc - 1)) - barWidth;

            // if (barWidth < 8) { barWidth = 8; space = 0; }
            // if (barWidth > 64) { barWidth = 64; space = ((r.Width - barWidth) / (cc - 1)) - barWidth; }

            // DrawPaper(g, r, theme, (byte)Math.Ceiling(barWidth), Yscale);

            var PixelOffsetMode = g.PixelOffsetMode;
            g.PixelOffsetMode = Drawing2D.PixelOffsetMode.HighQuality;
            g.SmoothingMode = Drawing2D.SmoothingMode.HighQuality;

            float fontSize = barWidth / 2;
            if (fontSize > 11) fontSize = 11;
            Font fgFont = new Font(font.FontFamily, fontSize);

            float x = r.Left;

            for (int nCh = 0; nCh < cc; nCh++) {

                var barArea = new RectangleF(x, r.Top, barWidth, r.Height);
                var padding = barWidth / 4;
                barArea.Height /= 2;
                barArea.Height -= barArea.Width / 2;
                barArea.Inflate(-padding, 0);

                var Pt = X(nCh);

                for (int k = 0; k < Pt.ampl.Length; k++) {

                    var ampl = Pt.ampl[k];

                    if (ampl < -17 || ampl > +17) {
                        ampl = 17f * (float)Math.Sign(ampl);
                        // throw new ArgumentOutOfRangeException();
                    }

                    var h = barArea.Height * Math.Abs(ampl);
                    var w = barArea.Width / Pt.ampl.Length;

                    Brush bgBlendCh = blend && (bg(k) is SolidBrush bgSolidColorBrushCh)
                         ? new SolidBrush(Blend(bgSolidColorBrushCh.Color, Color.Gray, (ampl))) : (Brush)bg(k).Clone();

                    if (ampl > 0) {
                        g.FillRectangle(bgBlendCh,
                            barArea.X + k * w,
                            barArea.Y + (barArea.Height - h),
                            w,
                            h);
                    } else if (ampl < 0) {
                        g.FillRectangle(bgBlendCh,
                            barArea.X + k * w,
                            barArea.Y + barArea.Height + padding + barArea.Width + padding,
                            w,
                            h);
                    }

                    bgBlendCh.Dispose();
                }

                // Brush bgBlend = blend && (bg(nCh) is SolidBrush bgSolidColorBrush)
                //      ? new SolidBrush(Blend(bgSolidColorBrush.Color, Color.Gray, MathF.Mean(Pt.ampl))) : (Brush)bg.Clone();
                // 
                // g.FillEllipse(bgBlend,
                //     barArea.X,
                //     barArea.Height + padding, barArea.Width, barArea.Width);
                // 
                // if (!string.IsNullOrWhiteSpace(Pt.label)) {
                //     var sz = g.MeasureString(Pt.label, fgFont);
                //     StringFormat drawFormat = new StringFormat();
                //     drawFormat.FormatFlags = StringFormatFlags.NoWrap
                //         | StringFormatFlags.DirectionVertical | StringFormatFlags.DirectionRightToLeft;
                //     g.DrawString(
                //         "- " + Pt.label, fgFont, bgBlend,
                //         barArea.X + sz.Height / 2 + barArea.Width / 2,
                //         barArea.Y + barArea.Height + padding + barArea.Width + padding,
                //         drawFormat);
                // }
                // 
                // bgBlend.Dispose();

                x += barWidth + space;
            }

            fgFont.Dispose();
            g.PixelOffsetMode = PixelOffsetMode;
        }

#if xxx
        public static void DrawElectrodes(Graphics g, RectangleF r, int cc, Brush bg, Font font, Brush fg,
    ITheme theme, RandomF seed, Func<int, bool> drawCarrier = null, Func<int, (float[] ampl, string label)> X = null,
    bool blend = true, Func<int, float> Z = null) {
            DrawElectrodes2(g,
                r, cc, (i) => bg, font, fg, theme, seed, drawCarrier,
                X, blend, Z);
        }
        public static void DrawElectrodes2(Graphics g, RectangleF r, int cc, Func<int, Brush> bg, Font font, Brush fg,
            ITheme theme,
            RandomF seed,
            Func<int, bool> drawCarrier = null, Func<int, (float[] ampl, string label)> X = null,
            bool blend = true, Func<int, float> Z = null) {

            float barWidth = r.Width / cc;
            float space = ((r.Width - barWidth) / (cc - 1)) - barWidth;
            if (barWidth < 9) { barWidth = 9; space = 0; }
            if (barWidth > 64) { barWidth = 64; space = ((r.Width - barWidth) / (cc - 1)) - barWidth; }

            DrawPaper(g, r, theme, (byte)barWidth, Yscale);

            var PixelOffsetMode = g.PixelOffsetMode;
            g.PixelOffsetMode = Drawing2D.PixelOffsetMode.HighQuality;
            g.SmoothingMode = Drawing2D.SmoothingMode.HighQuality;

            float fontSize = barWidth / 2;
            if (fontSize > 8) fontSize = 8;
            Font fgFont = new Font(font.FontFamily, fontSize);

            float x = 0;

            for (int nElec = 0; nElec < cc; nElec++) {
                var barArea = new RectangleF(x, 0, barWidth, r.Height);
                var padding = barWidth / 4;
                barArea.Height /= 2;
                barArea.Height -= barArea.Width / 2;
                barArea.Inflate(-padding, 0);

                var Pt = X(nElec);

                for (int k = 0; k < Pt.ampl.Length; k++) {

                    var ampl = Pt.ampl[k];

                    if (ampl < -17 || ampl > +17) {
                        ampl = 17f * (float)Math.Sign(ampl);
                        // throw new ArgumentOutOfRangeException();
                    } 
                    var h = barArea.Height * Math.Abs(ampl);
                    var w = barArea.Width / Pt.ampl.Length;

                    Brush bgBlendCh = blend && (bg(k) is SolidBrush bgSolidColorBrushCh)
                         ? new SolidBrush(Blend(bgSolidColorBrushCh.Color, Color.Gray, (ampl))) : (Brush)bg.Clone();

                    if (ampl > 0) {
                        g.FillRectangle(bgBlendCh,
                            barArea.X + k * w,
                            barArea.Y + (barArea.Height - h),
                            w,
                            h);
                    } else if (ampl < 0) {
                        g.FillRectangle(bgBlendCh,
                            barArea.X + k * w,
                            barArea.Y + barArea.Height + padding + barArea.Width + padding,
                            w,
                            h);
                    }

                    bgBlendCh.Dispose();
                }

                Brush bgBlend = blend && (bg(nElec) is SolidBrush bgSolidColorBrush)
                     ? new SolidBrush(Blend(bgSolidColorBrush.Color, Color.Gray, MathF.Mean(Pt.ampl))) : (Brush)bg.Clone();

                g.FillEllipse(bgBlend,
                    barArea.X,
                    barArea.Height + padding, barArea.Width, barArea.Width);

                if (!string.IsNullOrWhiteSpace(Pt.label)) {
                    var sz = g.MeasureString(Pt.label, fgFont);
                    StringFormat drawFormat = new StringFormat();
                    drawFormat.FormatFlags = StringFormatFlags.NoWrap
                        | StringFormatFlags.DirectionVertical | StringFormatFlags.DirectionRightToLeft;
                    g.DrawString(
                        "- " + Pt.label, fgFont, bgBlend,
                        barArea.X + sz.Height / 2 + barArea.Width / 2,
                        barArea.Y + barArea.Height + padding + barArea.Width + padding,
                        drawFormat);
                }

                bgBlend.Dispose();

                x += barWidth + space;
            }

            fgFont.Dispose();
            g.PixelOffsetMode = PixelOffsetMode;
        }

        public static float BarWidth(RectangleF r, int cc, out float space) {
            float barWidth = r.Width / cc;
            space = ((r.Width - barWidth) / (cc - 1)) - barWidth;
            if (barWidth < 8) { barWidth = 8; space = 0; }
            if (barWidth > 64) { barWidth = 64; space = ((r.Width - barWidth) / (cc - 1)) - barWidth; }
            return barWidth;
        }

        public static float DrawLayer(Graphics g, RectangleF r, int cc, Brush bg, Font font, Brush fg,
            ITheme theme, bool blend = false) {


            float barWidth = BarWidth(r, cc, out float space);

           // g.FillRectangle(bg, r);


            // DrawPaper(g, r, theme, (byte)barWidth, Yscale);

            var PixelOffsetMode = g.PixelOffsetMode;
            g.PixelOffsetMode = Drawing2D.PixelOffsetMode.HighQuality;
            g.SmoothingMode = Drawing2D.SmoothingMode.HighQuality;

            float fontSize = barWidth / 2;
            if (fontSize > 11) fontSize = 11;
            Font fgFont = new Font(font.FontFamily, fontSize);

            float x = r.X,
                y = r.Y;

            for (int nDot = 0; nDot < cc; nDot++) {

                var barArea = new RectangleF(x, y, barWidth, r.Height);

                var ampl = 1f;

                if (ampl < -17 || ampl > +17) {
                    ampl = 17f * (float)Math.Sign(ampl);
                    // throw new ArgumentOutOfRangeException();
                }

                // var h = barArea.Height * Math.Abs(ampl);
                // var w = barArea.Width;

                // Brush bgBlendCh = blend && (bg(nDot) is SolidBrush bgSolidColorBrushCh)
                //      ? new SolidBrush(Blend(bgSolidColorBrushCh.Color, Color.Gray, (ampl))) : (Brush)bg.Clone();
                // 
                // if (ampl > 0) {
                //     g.FillRectangle(bgBlendCh,
                //         barArea.X,
                //         barArea.Y + (barArea.Height - h),
                //         w,
                //         h);
                // } else if (ampl < 0) {
                //     g.FillRectangle(bgBlendCh,
                //         barArea.X,
                //         barArea.Y + barArea.Height + padding + barArea.Width + padding,
                //         w,
                //         h);
                // }
                // 
                // bgBlendCh.Dispose();

                g.FillEllipse(bg,
                    barArea.X,
                    barArea.Y + barArea.Height / 2f - barArea.Width / 2f, barArea.Width, barArea.Width);

                string label = $" - {nDot + 1}";

                if (!string.IsNullOrWhiteSpace(label)) {
                    var sz = g.MeasureString(label, fgFont);
                    StringFormat drawFormat = new StringFormat();
                    drawFormat.FormatFlags = StringFormatFlags.NoWrap
                        | StringFormatFlags.DirectionVertical | StringFormatFlags.DirectionRightToLeft;
                    g.DrawString(
                        label, fgFont, bg,
                        barArea.X + sz.Height / 2f + barArea.Width / 2f,
                        barArea.Y + barArea.Height / 2f + barArea.Width / 2f,
                        drawFormat);
                }

                x += barWidth + space;
            }

            fgFont.Dispose();
            g.PixelOffsetMode = PixelOffsetMode;

            return barWidth;
        }

#endif
        // public static void DrawPiano(Graphics g, RectangleF r, Brush bg, Font font, Brush fg,
        //     ITheme theme, Func<PianoKey, (float ampl, string label)> X) {
        //     var PixelOffsetMode = g.PixelOffsetMode;
        //     g.PixelOffsetMode = Drawing2D.PixelOffsetMode.HighQuality;
        //     g.SmoothingMode = Drawing2D.SmoothingMode.HighQuality;
        // 
        //     int cc = 7 * 4;
        //     float barWidth = r.Width / cc;
        //     float space = ((r.Width - barWidth) / (cc - 1)) - barWidth;
        //     if (barWidth < 8) { barWidth = 8; space = 0; }
        //     if (barWidth > 64) { barWidth = 64; space = ((r.Width - barWidth) / (cc - 1)) - barWidth; }
        // 
        //     float fontSize = barWidth / 2;
        //     if (fontSize > 8) fontSize = 8;
        //     Font fgFont = new Font(font.FontFamily, fontSize);
        // 
        //     float x = 0;
        // 
        //     int k = 0;
        // 
        //     // White Keys
        // 
        //     const string WhiteKeys = "CDEFGAB";
        // 
        //     for (int i = 0; i < cc; i++) {
        //         int Octave = (i / WhiteKeys.Length) + 2;
        // 
        //         string s = WhiteKeys[k % WhiteKeys.Length].ToString() + Octave.ToString();
        // 
        //         PianoKey key = (PianoKey)Enum.Parse(typeof(PianoKey), s);
        // 
        //         var Pt = X(key);
        // 
        //         var barArea = new RectangleF(x, 0, barWidth, r.Height);
        //         var padding = barWidth / 16;
        // 
        //         barArea.Inflate(-padding, -r.Height / 4);
        // 
        //         g.FillRectangle(Brushes.White,
        //             barArea.X,
        //             barArea.Y,
        //             barArea.Width,
        //             barArea.Height);
        // 
        //         Brush bgBlend = bg is SolidBrush bgSolidColorBrush
        //                  ? new SolidBrush(Blend(Color.White, bgSolidColorBrush.Color, 1 - Pt.ampl)) : (Brush)bg.Clone();
        // 
        //         // if (i == 7 || i == 13)
        //         {
        //             barArea.Inflate(-0, -0);
        // 
        //             g.FillRectangle(bgBlend,
        //                 barArea.X,
        //                 barArea.Y,
        //                 barArea.Width,
        //                 barArea.Height);
        //         }
        // 
        //         bgBlend.Dispose();
        // 
        //         s += " - " + Pt.label;
        // 
        //         if (!string.IsNullOrWhiteSpace(s)) {
        //             var sz = g.MeasureString(s, fgFont);
        //             StringFormat drawFormat = new StringFormat();
        //             drawFormat.FormatFlags = StringFormatFlags.NoWrap
        //                 | StringFormatFlags.DirectionVertical | StringFormatFlags.DirectionRightToLeft;
        //             g.DrawString(
        //                 s, fgFont, theme.GetBrush(ThemeColor.Foreground),
        //                 barArea.X + sz.Height / 2 + barArea.Width / 2,
        //                 barArea.Y + barArea.Height - barArea.Width,
        //                 drawFormat);
        //         }
        // 
        //         k++;
        // 
        //         x += barWidth + space;
        //     }
        // 
        //     x = 0;
        // 
        //     // Black Keys
        // 
        //     string BlackKeys = "CDFGA";
        //     
        //     k = 0;
        // 
        //     for (int i = 0; i < cc; i++) {
        //         int Octave = (i / WhiteKeys.Length) + 2;
        // 
        //         string s = BlackKeys[k % BlackKeys.Length].ToString() + "Sharp" + Octave.ToString();
        // 
        //         PianoKey key = (PianoKey)Enum.Parse(typeof(PianoKey), s);
        // 
        //         var Pt = X(key);
        // 
        //         var barArea = new RectangleF(x, 0, barWidth, r.Height);
        //         var padding = barWidth / 16;
        // 
        //         barArea.Inflate(-2 * padding, -r.Height / 4);
        // 
        //         if (i % 7 == 0 || i % 7 == 1 || i % 7 == 3 || i % 7 == 4 || i % 7 == 5) {
        // 
        //             barArea = new RectangleF(barArea.X + 2 * padding + barArea.Width / 2,
        //                 barArea.Y - 1,
        //                 barArea.Width,
        //                 barArea.Height - 4 - barArea.Height / 4);
        // 
        //             g.FillRectangle(Brushes.Black,
        //                 barArea.X,
        //                 barArea.Y,
        //                 barArea.Width,
        //                 barArea.Height);
        // 
        //             Brush bgBlend = bg is SolidBrush bgSolidColorBrush
        //  ? new SolidBrush(Blend(Color.Black, bgSolidColorBrush.Color, 1 - Pt.ampl)) : (Brush)bg.Clone();
        // 
        //             // if (i == 12 || i == 15)
        //             {
        //                 barArea.Inflate(-2, -2);
        // 
        //                 g.FillRectangle(bgBlend,
        //                     barArea.X,
        //                     barArea.Y - 1,
        //                     barArea.Width,
        //                     barArea.Height + 2);
        //             }
        // 
        //             bgBlend.Dispose();
        // 
        //             s = BlackKeys[k % BlackKeys.Length].ToString() + "#" + Octave.ToString() + " - " + Pt.label;
        // 
        //             if (!string.IsNullOrWhiteSpace(s)) {
        //                 var sz = g.MeasureString(s, fgFont);
        //                 StringFormat drawFormat = new StringFormat();
        //                 drawFormat.FormatFlags = StringFormatFlags.NoWrap
        //                     | StringFormatFlags.DirectionVertical | StringFormatFlags.DirectionRightToLeft;
        //                 g.DrawString(
        //                     s, fgFont, theme.GetBrush(ThemeColor.Foreground),
        //                     barArea.X + sz.Height / 2 + barArea.Width / 2,
        //                     barArea.Y + barArea.Height - barArea.Width - sz.Width,
        //                     drawFormat);
        //             }
        // 
        //             k++;
        // 
        //         }
        // 
        //         x += barWidth + space;
        //     }
        // 
        //     fgFont.Dispose();
        //     g.PixelOffsetMode = PixelOffsetMode;
        // }

        public static void DrawSignal(Graphics g,
            RectangleF r,
            Brush brush,
            Func<int, float?> X,
            SignalType type,
            int cc, float penWidth = 1.7f) {
            var PixelOffsetMode = g.PixelOffsetMode;
            if (X != null) {
                g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighQuality;
                PointF[] dots = new PointF[cc];
                for (int i = 0; i < cc; i++) {
                    var ampl = X(i);
                    if (!ampl.HasValue) {
                        continue;
                    }
                    if (ampl < -1 || ampl > +1) {
                       // throw new ArgumentOutOfRangeException();
                    }
                    float y = Map(-ampl.Value, -1f, 1f, r.Top + 1, r.Bottom - 1);
                    float x = Map(i, 0, cc - 1, r.Left + 1, r.Right - 1);
                    dots[i] = new PointF(x, y);
                    if (float.IsNaN(x) || float.IsNaN(y)) {
                     //   Debugger.Break();
                    }
                    if (type == SignalType.Dot) {
                        g.FillEllipse(brush, x - 2, y - 2, 4, 4);
                    }
                }
                if (dots.Length > 1) {
                    if (type == SignalType.Line) {
                        var pen = new Pen(brush, penWidth);
                        g.DrawLines(
                            pen,
                            dots);
                        pen.Dispose();
                    } else if (type == SignalType.Curve) {
                        var pen = new Pen(brush, penWidth);
                        g.DrawCurve(
                            pen,
                            dots);
                        pen.Dispose();
                    }
                }
            }
            g.PixelOffsetMode = PixelOffsetMode;
        }

        public static void DrawSignal(Graphics g,
            RectangleF r,
            Brush brush,
            float[] X,
            SignalType signalType = SignalType.Line) {
            if (X != null)
                DrawSignal(g, r, brush, (t) => X[t], signalType, X.Length);
        }
    }
}