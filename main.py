import data_simulation
import measure_improvement
import plot_measure_improvements
import influence_calculation
import plot_influences
import read_csv

# Create data directory and plot directory if they do not exist
read_csv.create_directory('data')
read_csv.create_directory('plots')

# Simulate data (x,y)
data_simulation.simulate_x()
data_simulation.simulate_y()

# Test parts of influence measures
measure_improvement.pd_rf_to_csv()
measure_improvement.pd_compare_to_csv()
measure_improvement.perm_to_csv()
measure_improvement.perm_rf_to_csv()
measure_improvement.shap_to_csv()

# Plot tests
plot_measure_improvements.pd_rf_plot_to_jpg()
plot_measure_improvements.pd_compare_plot_to_jpg()
plot_measure_improvements.perm_plot_to_jpg()
plot_measure_improvements.perm_rf_plot_to_jpg()
plot_measure_improvements.shap_plot_to_jpg()

# Evaluation of final influence measures
influence_calculation.linear_independent_to_csv()
influence_calculation.linear_correlated_to_csv()
influence_calculation.linear_extrapolation_to_csv()
influence_calculation.non_linear_independent_to_csv()
influence_calculation.non_linear_correlated_to_csv()
influence_calculation.non_linear_extrapolation_to_csv()
influence_calculation.true_model_influence_to_csv()

# Plot influences
plot_influences.linear_plot_to_jpg()
plot_influences.independence_plot_to_jpg()
plot_influences.correlation_plot_to_jpg()
plot_influences.extrapolation_plot_to_jpg()
