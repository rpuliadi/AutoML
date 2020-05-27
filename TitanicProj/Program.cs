using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Internal.Utilities;
using System.Data.SqlClient;
using System.IO;
using PLplot;
using Common;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Threading;
using System.Linq;
using System.Collections.Immutable;

namespace Titanic
{
    internal class Program
    {
        private static string BaseDatasetsRelativePath = @"../../../Data";

        // private static string TrainDataRelativePath = $"{BaseDatasetsRelativePath}\titanic.csv";
        // private static string TrainDataPath = GetAbsolutePath(TrainDataRelativePath);
        private static IDataView TrainDataView = null;

        // private static string TestDataRelativePath = $"{BaseDatasetsRelativePath}\\titanic.csv";
        // private static string TestDataPath = GetAbsolutePath(TestDataRelativePath);
        private static IDataView TestDataView = null;

        private static string BaseModelsRelativePath = @"../../../MLModels";
        private static string ModelRelativePath = $"{BaseModelsRelativePath}\\TitanicModel.zip";
        private static string ModelPath = GetAbsolutePath(ModelRelativePath);

        private static string LabelColumnName = "fblnSurvived";

        private static string serverName = "<ServerName>";
        private static string dbName = "<dbName>";

        private static readonly string connectionString = $"Server={serverName}; Database={dbName}; Integrated Security=True";
        private static readonly string commandText = 
            "SELECT [fstrClass] AS [Class]" +
            ", CAST([fblnSurvived] AS BIT) AS Survived" +
            ", [fstrName] AS Name, [fstrGender] AS Gender" +
            ", CAST([flngAge] AS REAL) AS Age" +
            ", CAST([flngSiblingsSpouses] AS REAL) AS SiblingsSpouses" +
            ", CAST([flngParentChildren] AS REAL) AS ParentChildren" +
            ", [fstrTicketNumber] AS TicketNumber" +
            ", CAST([fcurPassengerFare] AS REAL) AS PassengerFare" +
            ", [fstrCabin] AS Cabin" +
            ", [fstrEmbarked] AS Embarked" +
            " FROM[dbo].[tblFE_Ana_Titanic]";

        private static MLContext mlContext;


        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            //Console.WriteLine("Enter the name of the table:");
            //String tblName = Console.ReadLine();
            //String TrainDataRelativePath = ConvertDataToCsv(tblName);
            //Console.WriteLine("Enter the name of the target field:");
            //LabelColumnName = Console.ReadLine();

            String tblName = "tblFE_Ana_Titanic";
            String TrainDataRelativePath = ConvertDataToCsv(tblName);
            LabelColumnName = "fblnSurvived";


            mlContext = new MLContext(seed: 1);
            DatabaseLoader loader = mlContext.Data.CreateDatabaseLoader<TitanicData>();
            DatabaseSource dbSource = new DatabaseSource(providerFactory: SqlClientFactory.Instance, connectionString: connectionString, commandText: commandText);

            //ColumnInferenceResults columnInference = mlContext.Auto().InferColumns();

            var columnInference = InferColumns(mlContext, GetAbsolutePath(TrainDataRelativePath));

            LoadData(mlContext, columnInference, GetAbsolutePath(TrainDataRelativePath), GetAbsolutePath(TrainDataRelativePath));

            var experimentResult = RunAutoMLExperiment(mlContext: mlContext, columnInference: columnInference);

            EvaluateModel(mlContext, experimentResult.BestRun.Model, experimentResult.BestRun.TrainerName);

            SaveModel(mlContext, experimentResult.BestRun.Model);

           // PlotRegressionChart(mlContext, TestDataPath, numberOfRecordsToRead: 100, args);

            var refitModel = RefitBestPipeline(mlContext, experimentResult, columnInference, GetAbsolutePath(TrainDataRelativePath), GetAbsolutePath(TrainDataRelativePath));



        }

        private static string ConvertDataToCsv(String tableName)
        {
            SqlConnection sqlCon = new SqlConnection(connectionString);
            sqlCon.Open();

            SqlCommand sqlCmd = new SqlCommand(
                $"SELECT * FROM [dbo].[{tableName}]", sqlCon);
            SqlDataReader reader = sqlCmd.ExecuteReader();

            string fileName = $"{GetAbsolutePath(BaseDatasetsRelativePath)}\\{tableName}.csv";
            StreamWriter sw = new StreamWriter(fileName);
            object[] output = new object[reader.FieldCount];

            for (int i = 0; i < reader.FieldCount; i++)
                output[i] = reader.GetName(i);

            sw.WriteLine(string.Join(",", output));

            while (reader.Read())
            {
                reader.GetValues(output);
                sw.WriteLine(string.Join(",", output));
            }

            sw.Close();
            reader.Close();
            sqlCon.Close();
            return fileName;
        }

        /// <summary>
        /// Infer columns in the dataset with AutoML.
        /// </summary>
        private static ColumnInferenceResults InferColumns(MLContext mlContext, string TrainDataPath)
        {
            Console.WriteLine("=============== Inferring columns in dataset ===============");
            ColumnInferenceResults columnInference = mlContext.Auto().InferColumns(TrainDataPath, LabelColumnName, groupColumns: false);
            Console.WriteLine(columnInference);
            return columnInference;
        }

        /// <summary>
        /// Load data from files using inferred columns.
        /// </summary>
        private static void LoadData(MLContext mlContext, ColumnInferenceResults columnInference, string TrainDataPath, string TestDataPath)
        {
            TextLoader textLoader = mlContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);
            TrainDataView = textLoader.Load(TrainDataPath);
            TestDataView = textLoader.Load(TrainDataPath);
        }

        private static ExperimentResult<BinaryClassificationMetrics> RunAutoMLExperiment(MLContext mlContext,
            ColumnInferenceResults columnInference)
        {
            // STEP 1: Display first few rows of the training data.
            
            // ConsoleHelper.ShowDataViewInConsole(mlContext, TrainDataView);

            // STEP 2: Build a pre-featurizer for use in the AutoML experiment.
            // (Internally, AutoML uses one or more train/validation data splits to 
            // evaluate the models it produces. The pre-featurizer is fit only on the 
            // training data split to produce a trained transform. Then, the trained transform 
            // is applied to both the train and validation data splits.)
            //IEstimator<ITransformer> preFeaturizer = mlContext.Transforms.Conversion.MapValue("is_cash",
            //    new[] { new KeyValuePair<string, bool>("CSH", true) }, "payment_type");

            //IEstimator<ITransformer> preFeaturizer = mlContext.Transforms.Conversion.MapValue("fstrClassCategory",
            //    new[] { new KeyValuePair<float, String>(1, "First"), new KeyValuePair<float, String>(2, "Second"), new KeyValuePair<float, String>(3, "Third") }, "fstrClass").Append(mlContext.Transforms.Categorical.OneHotEncoding("fstrClassCategory", "fstrClassCategory")).Append(mlContext.Transforms.DropColumns("fstrClass"));

            // STEP 3: Customize column information returned by InferColumns API.
            ColumnInformation columnInformation = columnInference.ColumnInformation;
            columnInformation = CorrectColumnTypes(columnInformation);

            // columnInformation.NumericColumnNames.Remove("fstrClass");
            // columnInformation.CategoricalColumnNames.Add("fstrClass");
            // columnInformation.IgnoredColumnNames.Add("fstrClass");
            // columnInformation.IgnoredColumnNames.


            // STEP 4: Initialize a cancellation token source to stop the experiment.
            var cts = new CancellationTokenSource();

            // STEP 5: Initialize our user-defined progress handler that AutoML will 
            // invoke after each model it produces and evaluates.
            var progressHandler = new BinaryExperimentProgressHandler(); //  RegressionExperimentProgressHandler();

            // STEP 6: Create experiment settings
            var experimentSettings = CreateExperimentSettings(mlContext, cts);

            // STEP 7: Run AutoML Binary Classification experiment.
            var experiment = mlContext.Auto().CreateBinaryClassificationExperiment(experimentSettings);
            ConsoleHelper.ConsoleWriteHeader("=============== Running AutoML experiment ===============");
            Console.WriteLine($"Running AutoML regression experiment...");
            var stopwatch = Stopwatch.StartNew();
            // Cancel experiment after the user presses any key.
            CancelExperimentAfterAnyKeyPress(cts);
            ExperimentResult<BinaryClassificationMetrics> experimentResult = experiment.Execute(trainData: TrainDataView, columnInformation: columnInformation, progressHandler: progressHandler);
            Console.WriteLine($"{experimentResult.RunDetails.Count()} models were returned after {stopwatch.Elapsed.TotalSeconds:0.00} seconds{Environment.NewLine}");

            // Print top models found by AutoML.
            PrintTopModels(experimentResult);
            // var featureNames = columnInformation.CategoricalColumnNames.Concat(columnInformation.ImagePathColumnNames).Concat(columnInformation.NumericColumnNames).Concat(columnInformation.TextColumnNames).ToList();
            // var permutationMetrics = mlContext.BinaryClassification.PermutationFeatureImportance(predictionTransformer: )
            // PrintContributions(featureNames, TrainDataView, experimentResult.RunDetails);

            // DatasetDimensionsUtil.GetTextColumnCardinality();

            return experimentResult;

        }

        private static ColumnInformation CorrectColumnTypes(ColumnInformation columnInformation)
        {
            var modifyColumns = new List<String>();
            foreach (var numCol in columnInformation.NumericColumnNames) 
            {
                if (numCol.Contains("fstr")) 
                {
                    modifyColumns.Add(numCol);
                }
            }

            foreach (var numCol in modifyColumns)
            {
                columnInformation.NumericColumnNames.Remove(numCol);
                columnInformation.CategoricalColumnNames.Add(numCol);
            }

            return columnInformation;
        }

        private static void PrintContributions(List<String> allFeatureNames, IDataView permuteTestData, ImmutableArray<BinaryClassificationMetrics> permutationMetrics)
        {
            var mapFields = new List<String>();

            for (int i = 0; i < allFeatureNames.Count; i++)
            {
                var slotField = new VBuffer<ReadOnlyMemory<Char>>();
                if (permuteTestData.Schema[allFeatureNames[i]].HasSlotNames())
                {
                    permuteTestData.Schema[allFeatureNames[i]].GetSlotNames(ref slotField);
                    for (int j = 0; j < slotField.Length; j++)
                    {
                        var slotFieldInd = slotField.GetIndices();
                        var slotFieldVal = slotField.GetValues().ToArray()[j];
                        var CategoryValue = $"{allFeatureNames[i]}\t- {slotField.Items().ElementAtOrDefault(j)}";
                        mapFields.Add($"{allFeatureNames[i]}\t- {slotField.GetValues().ToArray()[j]}");
                    }
                }
                else
                {
                    mapFields.Add(allFeatureNames[i]);
                }
            }

            var sortedIndices = permutationMetrics.Select((metrics, index) => new { index, metrics.AreaUnderRocCurve})
                .OrderByDescending(feature => Math.Abs(feature.AreaUnderRocCurve));


            using (System.IO.StreamWriter file =
            new System.IO.StreamWriter($"{GetAbsolutePath(BaseModelsRelativePath)}\\TitanicResult.txt"))
            {
                foreach (var feature in sortedIndices)
                {
                    Console.WriteLine($"{mapFields[feature.index],-40}|\t{Math.Abs(feature.AreaUnderRocCurve):F6}");
                    file.WriteLine($"{mapFields[feature.index],-40}|\t{Math.Abs(feature.AreaUnderRocCurve):F6}");
                }
            }

        }

        /// <summary>
        /// Create AutoML Binary Classification experiment settings.
        /// </summary>
        private static BinaryExperimentSettings CreateExperimentSettings(MLContext mlContext,
            CancellationTokenSource cts)
        {
            var experimentSettings = new BinaryExperimentSettings();
            experimentSettings.MaxExperimentTimeInSeconds = 3600;
            experimentSettings.CancellationToken = cts.Token;

            // Set the metric that AutoML will try to optimize over the course of the experiment.
            experimentSettings.OptimizingMetric = BinaryClassificationMetric.Accuracy;

            // Set the cache directory to null.
            // This will cause all models produced by AutoML to be kept in memory 
            // instead of written to disk after each run, as AutoML is training.
            // (Please note: for an experiment on a large dataset, opting to keep all 
            // models trained by AutoML in memory could cause your system to run out 
            // of memory.)
            experimentSettings.CacheDirectory = null;

            // Don't use LbfgsPoissonRegression and OnlineGradientDescent trainers during this experiment.
            // (These trainers sometimes underperform on this dataset.)
            // experimentSettings.Trainers.Remove(BinaryClassificationTrainer.LbfgsLogisticRegression);
            // experimentSettings.Trainers.Remove(BinaryClassificationTrainer.SymbolicSgdLogisticRegression);

            return experimentSettings;
        }

        /// <summary>
        /// Print top models from AutoML experiment.
        /// </summary>
        private static void PrintTopModels(ExperimentResult<BinaryClassificationMetrics> experimentResult)
        {
            // Get top few runs ranked by root mean squared error.
            var topRuns = experimentResult.RunDetails
                .Where(r => r.ValidationMetrics != null && !double.IsNaN(r.ValidationMetrics.Accuracy))
                .OrderByDescending(r => r.ValidationMetrics.Accuracy).Take(3);

            Console.WriteLine("Top models ranked by Accuracy --");
            ConsoleHelper.PrintBinaryClassificationMetricsHeader();
            for (var i = 0; i < topRuns.Count(); i++)
            {
                var run = topRuns.ElementAt(i);
                ConsoleHelper.PrintIterationMetrics(i + 1, run.TrainerName, run.ValidationMetrics, run.RuntimeInSeconds);
            }

            var bestRun = topRuns.ElementAt(0);
            
        }

        /// <summary>
        /// Re-fit best pipeline on all available data.
        /// </summary>
        private static ITransformer RefitBestPipeline(MLContext mlContext, ExperimentResult<BinaryClassificationMetrics> experimentResult,
            ColumnInferenceResults columnInference, string TrainDataPath, string TestDataPath)
        {
            ConsoleHelper.ConsoleWriteHeader("=============== Re-fitting best pipeline ===============");
            var textLoader = mlContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);
            var combinedDataView = textLoader.Load(new MultiFileSource(TrainDataPath, TestDataPath));
            RunDetail<BinaryClassificationMetrics> bestRun = experimentResult.BestRun;
            return bestRun.Estimator.Fit(combinedDataView);
        }

        /// <summary>
        /// Evaluate the model and print metrics.
        /// </summary>
        private static void EvaluateModel(MLContext mlContext, ITransformer model, string trainerName)
        {
            ConsoleHelper.ConsoleWriteHeader("===== Evaluating model's accuracy with test data =====");
            IDataView predictions = model.Transform(TestDataView);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: LabelColumnName, scoreColumnName: "Score", probabilityColumnName: "Probability");
            ConsoleHelper.PrintBinaryClassificationMetrics(trainerName, metrics);
        }

        /// <summary>
        /// Save/persist the best model to a .ZIP file
        /// </summary>
        private static void SaveModel(MLContext mlContext, ITransformer model)
        {
            ConsoleHelper.ConsoleWriteHeader("=============== Saving the model ===============");
            mlContext.Model.Save(model, TrainDataView.Schema, ModelPath);
            Console.WriteLine($"The model is saved to {ModelPath}");
        }

        private static void CancelExperimentAfterAnyKeyPress(CancellationTokenSource cts)
        {
            Task.Run(() =>
            {
                Console.WriteLine("Press any key to stop the experiment run...");
                Console.ReadKey();
                cts.Cancel();
            });
        }

        //private static void TestSinglePrediction(MLContext mlContext)
        //{
        //    ConsoleHelper.ConsoleWriteHeader("=============== Testing prediction engine ===============");

        //    // Sample: 
        //    // vendor_id,rate_code,passenger_count,trip_time_in_secs,trip_distance,payment_type,fare_amount
        //    // VTS,1,1,1140,3.75,CRD,15.5

        //    var taxiTripSample = new TaxiTrip()
        //    {
        //        VendorId = "VTS",
        //        RateCode = 1,
        //        PassengerCount = 1,
        //        TripTime = 1140,
        //        TripDistance = 3.75f,
        //        PaymentType = "CRD",
        //        FareAmount = 0 // To predict. Actual/Observed = 15.5
        //    };

        //    ITransformer trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);

        //    // Create prediction engine related to the loaded trained model.
        //    var predEngine = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(trainedModel);

        //    // Score.
        //    var predictedResult = predEngine.Predict(taxiTripSample);

        //    Console.WriteLine("**********************************************************************");
        //    Console.WriteLine($"Predicted fare: {predictedResult.FareAmount:0.####}, actual fare: 15.5");
        //    Console.WriteLine("**********************************************************************");
        //}

        //private static void PlotRegressionChart(MLContext mlContext,
        //                                        string testDataSetPath,
        //                                        int numberOfRecordsToRead,
        //                                        string[] args)
        //{
        //    ITransformer trainedModel;
        //    using (var stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
        //    {
        //        trainedModel = mlContext.Model.Load(stream, out var modelInputSchema);
        //    }

        //    // Create prediction engine related to the loaded trained model
        //    var predFunction = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(trainedModel);

        //    string chartFileName = "";
        //    using (var pl = new PLStream())
        //    {
        //        // use SVG backend and write to SineWaves.svg in current directory.
        //        if (args.Length == 1 && args[0] == "svg")
        //        {
        //            pl.sdev("svg");
        //            chartFileName = "TaxiRegressionDistribution.svg";
        //            pl.sfnam(chartFileName);
        //        }
        //        else
        //        {
        //            pl.sdev("pngcairo");
        //            chartFileName = "TaxiRegressionDistribution.png";
        //            pl.sfnam(chartFileName);
        //        }

        //        // use white background with black foreground.
        //        pl.spal0("cmap0_alternate.pal");

        //        // Initialize plplot.
        //        pl.init();

        //        // Set axis limits.
        //        const int xMinLimit = 0;
        //        const int xMaxLimit = 35; // Rides larger than $35 are not shown in the chart.
        //        const int yMinLimit = 0;
        //        const int yMaxLimit = 35;  // Rides larger than $35 are not shown in the chart.
        //        pl.env(xMinLimit, xMaxLimit, yMinLimit, yMaxLimit, AxesScale.Independent, AxisBox.BoxTicksLabelsAxes);

        //        // Set scaling for mail title text 125% size of default.
        //        pl.schr(0, 1.25);

        //        // The main title.
        //        pl.lab("Measured", "Predicted", "Distribution of Taxi Fare Prediction");

        //        // plot using different colors
        //        // see http://plplot.sourceforge.net/examples.php?demo=02 for palette indices
        //        pl.col0(1);

        //        int totalNumber = numberOfRecordsToRead;
        //        //var testData = new TaxiTripCsvReader().GetDataFromCsv(testDataSetPath, totalNumber).ToList();

        //        // This code is the symbol to paint
        //        var code = (char)9;

        //        // plot using other color
        //        //pl.col0(9); //Light Green
        //        //pl.col0(4); //Red
        //        pl.col0(2); //Blue

        //        double yTotal = 0;
        //        double xTotal = 0;
        //        double xyMultiTotal = 0;
        //        double xSquareTotal = 0;

        //        for (int i = 0; i < testData.Count; i++)
        //        {
        //            var x = new double[1];
        //            var y = new double[1];

        //            // Make Prediction.
        //            var farePrediction = predFunction.Predict(testData[i]);

        //            x[0] = testData[i].FareAmount;
        //            y[0] = farePrediction.FareAmount;

        //            // Paint a dot
        //            pl.poin(x, y, code);

        //            xTotal += x[0];
        //            yTotal += y[0];

        //            double multi = x[0] * y[0];
        //            xyMultiTotal += multi;

        //            double xSquare = x[0] * x[0];
        //            xSquareTotal += xSquare;

        //            double ySquare = y[0] * y[0];

        //            Console.WriteLine("-------------------------------------------------");
        //            Console.WriteLine($"Predicted : {farePrediction.FareAmount}");
        //            Console.WriteLine($"Actual:    {testData[i].FareAmount}");
        //            Console.WriteLine("-------------------------------------------------");
        //        }

        //        // Regression Line calculation explanation:
        //        // https://www.khanacademy.org/math/statistics-probability/describing-relationships-quantitative-data/more-on-regression/v/regression-line-example

        //        double minY = yTotal / totalNumber;
        //        double minX = xTotal / totalNumber;
        //        double minXY = xyMultiTotal / totalNumber;
        //        double minXsquare = xSquareTotal / totalNumber;

        //        double m = ((minX * minY) - minXY) / ((minX * minX) - minXsquare);

        //        double b = minY - (m * minX);

        //        // Generic function for Y for the regression line
        //        // y = (m * x) + b;

        //        double x1 = 1;

        //        // Function for Y1 in the line
        //        double y1 = (m * x1) + b;

        //        double x2 = 39;

        //        // Function for Y2 in the line
        //        double y2 = (m * x2) + b;

        //        var xArray = new double[2];
        //        var yArray = new double[2];
        //        xArray[0] = x1;
        //        yArray[0] = y1;
        //        xArray[1] = x2;
        //        yArray[1] = y2;

        //        pl.col0(4);
        //        pl.line(xArray, yArray);

        //        // End page (writes output to disk)
        //        pl.eop();

        //        // Output version of PLplot
        //        pl.gver(out var verText);
        //        Console.WriteLine("PLplot version " + verText);

        //    } // The pl object is disposed here

        //    // Open chart file in Microsoft Photos App (or default app for .svg or .png, like browser)

        //    Console.WriteLine("Showing chart...");
        //    var p = new Process();
        //    string chartFileNamePath = @".\" + chartFileName;
        //    p.StartInfo = new ProcessStartInfo(chartFileNamePath)
        //    {
        //        UseShellExecute = true
        //    };
        //    p.Start();
        //}

        public static string GetAbsolutePath(string relativePath)
        {
            var _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}
