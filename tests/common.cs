using System;

unsafe internal static class Common {
    public static string pretty_logits(float[] logits) {
        fixed (float* logits0 = logits) {
            return pretty_logits(logits0, (uint)logits.Length);
        }
    }

    public static string pretty_logits(float* logits0, uint cc, uint max_ = 0xFFFFFFFF, int decimals = 4) {
        uint n = Math.Min(cc, max_);
        string row = "[";
        for (int j = 0; j < n; j++) {
            var val = Math.Round(logits0[j], decimals);
            if (val == -0) {
                val = 0;
            }
            string s = $"{val:f4}";
            row += s;
            if (j == n - 1) {
                if (n < cc)
                    row += ", ...";
            } else {
                row += ", ";
            }
        }
        row += "]";
        return row;
    }
}