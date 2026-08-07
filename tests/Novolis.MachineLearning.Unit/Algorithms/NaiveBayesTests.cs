using Novolis.MachineLearning.Algorithms;
using Novolis.MachineLearning.Algorithms.NaiveBayes;

namespace Novolis.MachineLearning.Algorithms.Tests;

public sealed class NaiveBayesTests
{
    private enum PlayDecision
    {
        SitOut,
        Play,
    }

    private enum FlowerSpecies
    {
        Setosa,
        Versicolor,
        Virginica,
    }

    private enum MatchOutcome
    {
        Loss,
        Win,
    }

    [Test]
    public async Task Gaussian_WeatherPlayDecision_PredictsHeldOutRows()
    {
        // Outlook/Temp/Humidity/Wind encoded as continuous proxies — outdoor recreation, not finance.
        var train = new LabeledExample<double, PlayDecision>[]
        {
            new(new Features<double>(0, 85, 85, 0), PlayDecision.SitOut),
            new(new Features<double>(0, 80, 90, 1), PlayDecision.SitOut),
            new(new Features<double>(1, 83, 78, 0), PlayDecision.Play),
            new(new Features<double>(2, 70, 96, 0), PlayDecision.Play),
            new(new Features<double>(2, 68, 80, 0), PlayDecision.Play),
            new(new Features<double>(2, 65, 70, 1), PlayDecision.SitOut),
            new(new Features<double>(1, 64, 65, 1), PlayDecision.Play),
            new(new Features<double>(0, 72, 95, 0), PlayDecision.SitOut),
            new(new Features<double>(0, 69, 70, 0), PlayDecision.Play),
            new(new Features<double>(2, 75, 80, 0), PlayDecision.Play),
            new(new Features<double>(0, 75, 70, 1), PlayDecision.Play),
            new(new Features<double>(1, 72, 90, 1), PlayDecision.Play),
            new(new Features<double>(1, 81, 75, 0), PlayDecision.Play),
        };

        INaiveBayesTrainer<double, PlayDecision> trainer = new GaussianNaiveBayesTrainer<double, PlayDecision>();
        var model = trainer.Fit(train);

        var rainyCool = model.Predict(new Features<double>(2, 71, 80, 0));
        var sunnyHotHumid = model.Predict(new Features<double>(0, 85, 90, 1));

        await Assert.That(rainyCool).IsEqualTo(PlayDecision.Play);
        await Assert.That(sunnyHotHumid).IsEqualTo(PlayDecision.SitOut);
        await Assert.That(model.Classes.Count).IsEqualTo(2);
        await Assert.That(model.FeatureCount).IsEqualTo(4);
    }

    [Test]
    public async Task Gaussian_FlowerSpecies_SeparatesPetalClusters()
    {
        var train = new LabeledExample<float, FlowerSpecies>[]
        {
            new(new Features<float>(5.1f, 3.5f, 1.4f, 0.2f), FlowerSpecies.Setosa),
            new(new Features<float>(4.9f, 3.0f, 1.4f, 0.2f), FlowerSpecies.Setosa),
            new(new Features<float>(4.7f, 3.2f, 1.3f, 0.2f), FlowerSpecies.Setosa),
            new(new Features<float>(7.0f, 3.2f, 4.7f, 1.4f), FlowerSpecies.Versicolor),
            new(new Features<float>(6.4f, 3.2f, 4.5f, 1.5f), FlowerSpecies.Versicolor),
            new(new Features<float>(6.9f, 3.1f, 4.9f, 1.5f), FlowerSpecies.Versicolor),
            new(new Features<float>(6.3f, 3.3f, 6.0f, 2.5f), FlowerSpecies.Virginica),
            new(new Features<float>(5.8f, 2.7f, 5.1f, 1.9f), FlowerSpecies.Virginica),
            new(new Features<float>(7.1f, 3.0f, 5.9f, 2.1f), FlowerSpecies.Virginica),
        };

        var trainer = new GaussianNaiveBayesTrainer<float, FlowerSpecies>();
        var model = trainer.Fit(train);

        await Assert.That(model.Predict(new Features<float>(5.0f, 3.4f, 1.5f, 0.2f)))
            .IsEqualTo(FlowerSpecies.Setosa);
        await Assert.That(model.Predict(new Features<float>(6.7f, 3.1f, 4.7f, 1.4f)))
            .IsEqualTo(FlowerSpecies.Versicolor);
        await Assert.That(model.Predict(new Features<float>(6.5f, 3.0f, 5.8f, 2.2f)))
            .IsEqualTo(FlowerSpecies.Virginica);

        var scores = model.PredictScores(new Features<float>(5.0f, 3.4f, 1.5f, 0.2f));
        var setosa = scores.Single(s => s.Label == FlowerSpecies.Setosa);
        await Assert.That(setosa.Probability).IsGreaterThan(0.5);
        await Assert.That(scores.Sum(s => s.Probability)).IsEqualTo(1.0).Within(1e-9);
    }

    [Test]
    public async Task Bernoulli_MatchTraits_PredictsWinLoss()
    {
        // Binary match traits: home field, scored first, rested midweek — sports, not finance.
        var train = new LabeledExample<bool, MatchOutcome>[]
        {
            new(new Features<bool>(true, true, true), MatchOutcome.Win),
            new(new Features<bool>(true, true, false), MatchOutcome.Win),
            new(new Features<bool>(true, false, true), MatchOutcome.Win),
            new(new Features<bool>(false, true, true), MatchOutcome.Win),
            new(new Features<bool>(false, false, false), MatchOutcome.Loss),
            new(new Features<bool>(false, false, true), MatchOutcome.Loss),
            new(new Features<bool>(false, true, false), MatchOutcome.Loss),
            new(new Features<bool>(true, false, false), MatchOutcome.Loss),
        };

        INaiveBayesTrainer<bool, MatchOutcome> trainer = new BernoulliNaiveBayesTrainer<MatchOutcome>();
        var model = trainer.Fit(train);

        await Assert.That(model.Predict(new Features<bool>(true, true, true))).IsEqualTo(MatchOutcome.Win);
        await Assert.That(model.Predict(new Features<bool>(false, false, false))).IsEqualTo(MatchOutcome.Loss);
    }

    [Test]
    public async Task Features_Empty_Throws()
    {
        var actSpan = () => Features<double>.From(ReadOnlySpan<double>.Empty);
        var actArray = () => new Features<double>(Array.Empty<double>());
        await Assert.That(actSpan).Throws<ArgumentException>();
        await Assert.That(actArray).Throws<ArgumentException>();
    }

    [Test]
    public async Task Features_Equality_IsValueBased()
    {
        var a = new Features<int>(1, 2, 3);
        var b = Features<int>.From([1, 2, 3]);
        var c = new Features<int>(1, 2, 4);

        await Assert.That(a == b).IsTrue();
        await Assert.That(a != c).IsTrue();
        await Assert.That(a.Equals((object)b)).IsTrue();
        await Assert.That(a.Length).IsEqualTo(3);
        await Assert.That(a[1]).IsEqualTo(2);
    }

    [Test]
    public async Task Gaussian_FeatureLengthMismatch_Throws()
    {
        var trainer = new GaussianNaiveBayesTrainer<double, string>();
        var model = trainer.Fit(
        [
            new(new Features<double>(1.0, 2.0), "a"),
            new(new Features<double>(1.5, 2.5), "b"),
        ]);

        var act = () => model.Predict(new Features<double>(1.0));
        await Assert.That(act).Throws<ArgumentException>();
    }

    [Test]
    public async Task Gaussian_InconsistentTrainingLengths_Throws()
    {
        var trainer = new GaussianNaiveBayesTrainer<double, string>();
        var act = () => trainer.Fit(
        [
            new(new Features<double>(1.0, 2.0), "a"),
            new(new Features<double>(1.0), "b"),
        ]);

        await Assert.That(act).Throws<ArgumentException>();
    }

    [Test]
    public async Task Options_NonPositive_Throw()
    {
        await Assert.That(() => new NaiveBayesOptions { VarianceFloor = 0 }.Validate())
            .Throws<ArgumentOutOfRangeException>();
        await Assert.That(() => new NaiveBayesOptions { Smoothing = -1 }.Validate())
            .Throws<ArgumentOutOfRangeException>();
    }

    [Test]
    public async Task Bernoulli_EmptyTraining_Throws()
    {
        var trainer = new BernoulliNaiveBayesTrainer<string>();
        var act = () => trainer.Fit(Array.Empty<LabeledExample<bool, string>>());
        await Assert.That(act).Throws<ArgumentException>();
    }
}
