namespace System.Drawing
{
    public interface ITheme {
        Font GetFont(ThemeFont font);
        Color GetColor(ThemeColor color);
        Brush GetBrush(ThemeColor color);
        Pen GetPen(ThemeColor color);
    }
}