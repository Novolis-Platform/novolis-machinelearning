using System.Globalization;
using System.Text;

using Novolis.MachineLearning.Neural;
using Novolis.MachineLearning.TestInfrastructure;

using TUnit.Assertions;

namespace Novolis.MachineLearning.Neural;

/// <summary>
/// Exercises the NN stack through human-readable “visual” encodings of weights
/// (ASCII heatmap and tiny SVG). Useful for debugging and as a stable shape check.
/// </summary>
public sealed class DenseNetworkWeightVisualizationTests : BaseTest
{
    private const string AsciiPalette = " .:-=+*#%@";

    private static DenseLayer MakeLayer(int inCount, int outCount, double[,] weights, double[] biases, ActivationKind act)
        => new() { InputCount = inCount, OutputCount = outCount, Activation = act, Weights = weights, Biases = biases };

    private static DenseNetwork MakeNetwork(string name, params DenseLayer[] layers)
        => new() { Name = name, Layers = layers };

    /// <summary>
    /// Maps each weight cell to one ASCII glyph using min–max normalization within the matrix.
    /// </summary>
    private static string RenderWeightMatrixAscii(double[,] w)
    {
        int rows = w.GetLength(0);
        int cols = w.GetLength(1);
        double min = double.PositiveInfinity;
        double max = double.NegativeInfinity;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double v = w[i, j];
                if (v < min) min = v;
                if (v > max) max = v;
            }
        }

        var sb = new StringBuilder(rows * (cols + 1));
        int last = AsciiPalette.Length - 1;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double v = w[i, j];
                double t = max > min ? (v - min) / (max - min) : 0.5;
                int idx = (int)Math.Round(t * last);
                idx = Math.Clamp(idx, 0, last);
                sb.Append(AsciiPalette[idx]);
            }
            sb.Append('\n');
        }
        return sb.ToString();
    }

    /// <summary>
    /// Renders one layer’s weights as an SVG row of squares (grayscale from normalized weights).
    /// </summary>
    private static string RenderWeightMatrixSvg(double[,] w, int cellSize)
    {
        int rows = w.GetLength(0);
        int cols = w.GetLength(1);
        double min = double.PositiveInfinity;
        double max = double.NegativeInfinity;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double v = w[i, j];
                if (v < min) min = v;
                if (v > max) max = v;
            }
        }

        int width = cols * cellSize;
        int height = rows * cellSize;
        var sb = new StringBuilder(256 + rows * cols * 64);
        sb.Append(CultureInfo.InvariantCulture, $"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\">");
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double v = w[i, j];
                double t = max > min ? (v - min) / (max - min) : 0.5;
                int g = (int)Math.Round(Math.Clamp(t, 0, 1) * 255);
                int x = j * cellSize;
                int y = i * cellSize;
                sb.Append(CultureInfo.InvariantCulture, $"<rect x=\"{x}\" y=\"{y}\" width=\"{cellSize}\" height=\"{cellSize}\" fill=\"rgb({g},{g},{g})\"/>");
            }
        }
        sb.Append("</svg>");
        return sb.ToString();
    }

    [Test]
    public async Task AsciiHeatmap_TwoByOneColumn_LowToHigh_ReadsAsGradient()
    {
        var weights = new double[2, 1] { { -10.0 }, { 10.0 } };
        var layer = MakeLayer(2, 1, weights, new double[1], ActivationKind.Linear);
        var net = MakeNetwork("v", layer);
        string art = RenderWeightMatrixAscii(net.Layers[0].Weights);
        await Assert.That(art).IsEqualTo(" \n@\n");
    }

    [Test]
    public async Task AsciiHeatmap_TwoByTwoKnownMatrix_MatchesGolden()
    {
        // min=-1, max=1 → corners map to palette ends and mid values land in the middle.
        var weights = new double[2, 2] { { 1.0, 0.0 }, { 0.0, -1.0 } };
        var layer = MakeLayer(2, 2, weights, new double[2], ActivationKind.Linear);
        var net = MakeNetwork("v", layer);
        string art = RenderWeightMatrixAscii(net.Layers[0].Weights);
        await Assert.That(art).IsEqualTo("@=\n= \n");
    }

    [Test]
    public async Task SvgHeatmap_OneRowTwoCells_BlackToWhite()
    {
        var weights = new double[1, 2] { { 0.0, 1.0 } };
        var layer = MakeLayer(1, 2, weights, new double[2], ActivationKind.Linear);
        var net = MakeNetwork("v", layer);
        string svg = RenderWeightMatrixSvg(net.Layers[0].Weights, cellSize: 8);
        const string expected =
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"16\" height=\"8\">" +
            "<rect x=\"0\" y=\"0\" width=\"8\" height=\"8\" fill=\"rgb(0,0,0)\"/>" +
            "<rect x=\"8\" y=\"0\" width=\"8\" height=\"8\" fill=\"rgb(255,255,255)\"/>" +
            "</svg>";
        await Assert.That(svg).IsEqualTo(expected);
    }

    [Test]
    public async Task MultiLayerAscii_LogsStackedHeatmaps()
    {
        var l0 = MakeLayer(2, 2, new double[2, 2] { { 0, 2 }, { 2, 0 } }, new double[2], ActivationKind.Tanh);
        var l1 = MakeLayer(2, 1, new double[2, 1] { { -1 }, { 1 } }, new double[1], ActivationKind.Linear);
        var net = MakeNetwork("tiny", l0, l1);
        var sb = new StringBuilder();
        for (int li = 0; li < net.Layers.Length; li++)
        {
            sb.Append("Layer ").Append(li).Append('\n');
            sb.Append(RenderWeightMatrixAscii(net.Layers[li].Weights));
        }
        string stacked = sb.ToString();
        Output(stacked);
        await Assert.That(stacked).Contains("Layer 0");
        await Assert.That(stacked).Contains("Layer 1");
        await Assert.That(stacked).Contains("@");
    }
}
