using System;

unsafe internal static class Common {
    public static string pretty_logits(float[] logits) {
        fixed (float* logits0 = logits) {
            return pretty_logits(logits0, (uint)logits.Length);
        }
    }

    public static string pretty_logits(float* logits0, uint cc, uint max_ = 7) {
        uint n = Math.Min(cc, max_);
        string row = "[";
        for (int j = 0; j < n; j++) {
            row += $"{logits0[j]:f4}";
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