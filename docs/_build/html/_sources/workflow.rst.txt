Project Workflow
================

Dataset download
----------------

The project downloads the dataset from Hugging Face using the following
source:

.. code-block:: python

   DATASET_URI = (
       "hf://datasets/juliensimon/sentry-impact-risk/"
       "data/sentry_impact_risk.parquet"
   )

A sample of up to 500 rows is selected and stored in:

.. code-block:: text

   data/raw/dataset.csv

The dataset is then cleaned by removing rows containing missing values.
The cleaned dataset is stored in:

.. code-block:: text

   data/processed/dataset.csv

Feature engineering
-------------------

The ``features.py`` module reads the processed dataset and removes missing
values if necessary.

The target variable is transformed using a base-10 logarithm:

.. math::

   y = \log_{10}\left(\text{impact\_probability}\right)

All numerical columns are selected as model features except:

* ``impact_probability``
* ``log_impact_probability``

The resulting files are:

* ``features.csv``: model input variables.
* ``labels.csv``: transformed target variable.

Model architecture
------------------

The neural network is defined in ``modeling/train.py``.

It contains:

* An input layer.
* A dense layer with 64 neurons and ReLU activation.
* A dense layer with 32 neurons and ReLU activation.
* A dense layer with 16 neurons and ReLU activation.
* An output layer with one neuron.

The model uses:

* The Adam optimizer.
* A learning rate of ``0.001``.
* Mean squared error as the loss function.
* Mean absolute error as a metric.

Data splitting and scaling
--------------------------

The dataset is divided into training and test data using an 80/20 split.

The features are standardized with ``StandardScaler`` before training:

.. code-block:: python

   scaler = StandardScaler()

The scaler is saved to:

.. code-block:: text

   models/feature_scaler.joblib

Training
--------

The model is trained for a maximum of 100 epochs with a batch size of 32.

Early stopping is used to reduce overfitting. Training stops when the
validation loss does not improve for 15 consecutive epochs.

Evaluation
----------

The model is evaluated using:

* MAE: Mean Absolute Error.
* RMSE: Root Mean Squared Error.
* R²: Coefficient of Determination.

The results are saved to:

.. code-block:: text

   reports/model_metrics.csv

The training history is saved to:

.. code-block:: text

   reports/training_history.csv

Prediction
----------

The prediction module loads:

* The trained Keras model.
* The saved feature scaler.
* The processed feature data.

The model produces a prediction in logarithmic form. The original scale is
then recovered using:

.. math::

   \text{impact probability} = 10^{\text{log prediction}}