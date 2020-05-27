using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace AutoMLEx.DataStructures
{
    class TitanicPrediction
    {
        [ColumnName("Score")]
        public Boolean Survived;

        [ColumnName("Probability")]
        public float ProbabilitySurvived;
    }
}
