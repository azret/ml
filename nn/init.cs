namespace nn {
    using System;

    public static class init {
        public static double calculate_gain(string nonlinearity, double? a = null) {
            switch (nonlinearity) {
                case "linear":
                case "conv1d":
                case "conv2d":
                case "conv3d":
                case "conv_transpose1d":
                case "conv_transpose2d":
                case "conv_transpose3d":
                case "sigmoid":
                    return 1d;
                case "tanh":
                    return 5.0 / 3;
                case "relu":
                    return Math.Sqrt(2.0);
                case "leaky_relu":
                    double negative_slope = a.HasValue
                        ? a.Value
                        : 0.01;
                    return Math.Sqrt(2.0 / (1 + Math.Pow(negative_slope, 2)));
                case "selu":
                    return 3.0 / 4;
                default:
                    throw new NotSupportedException($"Unsupported nonlinearity {nonlinearity}");
            }
        }

        public unsafe static void kaiming_uniform_(float* data, uint numel, IRNG g, uint fan_in, float a, string nonlinearity = "leaky_relu") {
            var gain = calculate_gain(nonlinearity, a);
            var std = gain / Math.Sqrt(fan_in);
            var bound = Math.Sqrt(3.0) * std;
            nn.rand.uniform_(data,
                numel,
                g,
                -(float)bound, (float)bound);
        }
    }
}