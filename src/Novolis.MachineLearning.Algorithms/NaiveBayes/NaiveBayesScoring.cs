namespace Novolis.MachineLearning.Algorithms.NaiveBayes;

internal static class NaiveBayesScoring
{
    public static IReadOnlyList<ClassScore<TLabel>> ToClassScores<TLabel>(TLabel[] labels, double[] logScores)
        where TLabel : notnull
    {
        // Log-sum-exp for stable normalization.
        var max = logScores[0];
        for (var i = 1; i < logScores.Length; i++)
        {
            if (logScores[i] > max)
                max = logScores[i];
        }

        var weights = new double[logScores.Length];
        double sum = 0;
        for (var i = 0; i < logScores.Length; i++)
        {
            weights[i] = Math.Exp(logScores[i] - max);
            sum += weights[i];
        }

        var results = new ClassScore<TLabel>[labels.Length];
        for (var i = 0; i < labels.Length; i++)
            results[i] = new ClassScore<TLabel>(labels[i], logScores[i], weights[i] / sum);

        return results;
    }
}
