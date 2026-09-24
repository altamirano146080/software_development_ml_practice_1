Results
=======

Model metrics
-------------

The model is evaluated using the following metrics:

* Mean Absolute Error.
* Root Mean Squared Error.
* R² score.

The metrics are stored in:

.. code-block:: text

   reports/model_metrics.csv

Training history
----------------

The training history is stored in:

.. code-block:: text

   reports/training_history.csv

This file contains information about the training and validation loss during
the training process.

Generated visualizations
------------------------

The project generates several visualizations in
``reports/figures/``.

Impact probability distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../reports/figures/impact_probability_distribution.png
   :alt: Distribution of impact probability
   :width: 700px

Training history
~~~~~~~~~~~~~~~~

.. image:: ../reports/figures/training_history.png
   :alt: Training history
   :width: 700px

Predictions versus observations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../reports/figures/predictions_vs_observations.png
   :alt: Predictions versus observations
   :width: 700px

Correlation heatmap
~~~~~~~~~~~~~~~~~~~

.. image:: ../reports/figures/correlation_heatmap.png
   :alt: Correlation heatmap
   :width: 700px

Potential impact timeline
~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../reports/figures/potential_impact_timeline.png
   :alt: Potential impact timeline
   :width: 700px

Magnitude versus Palermo scale
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../reports/figures/magnitude_vs_palermo.png
   :alt: Magnitude versus Palermo scale
   :width: 700px

Velocity versus Palermo scale
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../reports/figures/velocity_vs_palermo.png
   :alt: Velocity versus Palermo scale
   :width: 700px