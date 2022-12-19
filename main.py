import data_simulation
import measure_improvement
import plot_measure_improvements
import influence_evaluation
import plot_influences

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
influence_evaluation.linear_independent_to_csv()
influence_evaluation.linear_correlated_to_csv()
influence_evaluation.linear_extrapolation_to_csv()
influence_evaluation.non_linear_independent_to_csv()
influence_evaluation.non_linear_correlated_to_csv()
influence_evaluation.non_linear_extrapolation_to_csv()
influence_evaluation.true_model_influence_to_csv()

# Plot influences
plot_influences.linear_plot_to_jpg()
plot_influences.independence_plot_to_jpg()
plot_influences.correlation_plot_to_jpg()
plot_influences.extrapolation_plot_to_jpg()
